from copy import deepcopy
from collections import namedtuple
import datetime
import io
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# Must be before torch.
import pydrake
import numpy as np
import scipy as sp
import scipy.stats
import time

from tensorboardX import SummaryWriter

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

from pyro import poutine
from pyro.infer import (
    config_enumerate,
    Trace_ELBO, TraceGraph_ELBO,
    SVI
)
from pyro.contrib.autoguide import (
    AutoDelta, AutoDiagonalNormal,
    AutoMultivariateNormal, AutoGuideList
)
import torch
import torch.distributions.constraints as constraints

import scene_generation.data.dataset_utils as dataset_utils
import scene_generation.differentiable_nlp as diff_nlp


class ProjectToFeasibilityDist(dist.TorchDistribution):
    has_rsample = True

    def __init__(self, pre_projection_params, class_i,
                 object_i, context, new_class, generated_data,
                 base_environment_type):
        batch_shape = pre_projection_params.shape[:-1]
        event_shape = pre_projection_params.shape[-1]
        dtype = pre_projection_params.dtype
        variance = torch.tensor(0.05, dtype=dtype)
        # TODO(gizatt) Handle when len(batch_shape) > 0?
        if len(batch_shape) > 1:
            raise NotImplementedError("Don't know how to handle"
                                      "multidimensional batch.")

        ones = np.ones(batch_shape[0], 1)
        tentative_generated_data = VectorizedEnvironments(
            batch_size=generated_data.batch_size,
            keep_going=np.hstack([generated_data.keep_going, ones]),
            classes=np.hstack([generated_data.classes[k, :], ones*class_i]),
            params_by_class=deepcopy(generated_data.params_by_class))
        tentative_generated_data.params_by_class[class_i] = np.hstack(
            [tentative_generated_data.params_by_class[class_i],
             pre_projection_params])

        all_params = []
        all_params_derivatives = []
        for k in range(batch_shape[0]):
            # Short circuit if class_i and new_class don't match.
            # This will likely happen in batching quite frequently,
            # and saves lots of unnecessary projections. What is produced
            # doesn't matter, as evaluated probabilities will be masked out.
            new_params = torch.tensor(pre_projection_params[k, :], dtype=dtype)
            new_params_derivs = torch.eye([event_shape, event_shape], dtype=dtype)
            if class_i == new_class[k]:
                env = tentative_generated_data.subsample([k]).convert_to_yaml()[0]
                builder, mbp, scene_graph, q0 = dataset_utils.BuildMbpAndSgFromYamlEnvironment(
                    env, base_environment_type)
                diagram = builder.Build()

                diagram_context = diagram.CreateDefaultContext()
                mbp_context = diagram.GetMutableSubsystemContext(
                    mbp, diagram_context)

                # Pre-compute the "active" decision variable indices
                if base_environment_type in ["planar_bin", "planar_tabletop"]:
                    x_index = mbp.GetJointByName(
                        "body_{}_x".format(object_i)).position_start()
                    z_index = mbp.GetJointByName(
                        "body_{}_z".format(object_i)).position_start()
                    t_index = mbp.GetJointByName(
                        "body_{}_theta".format(object_i)).position_start()
                    inds = [x_index, z_index, t_index]
                else:
                    raise NotImplementedError("Unsupported base environment type.")

                # Do projection
                q_min = q0.copy()
                q_max = q0.copy()
                q_min[inds] = -np.infty
                q_max[inds] = np.infty

                results = ProjectMBPToFeasibility(
                    q0, mbp, mbp_context,
                    [diff_nlp.SetArguments(diff_nlp.AddMinimumDistanceConstraint, minimum_distance=0.01),
                     diff_nlp.SetArguments(diff_nlp.AddJointPositionBounds(q_min=q_min, q_max=q_max))],
                    compute_gradients_at_solution=True, verbose=1)

                new_params[:len(inds)] = results.qf[inds]
                for i, ind in enumerate(inds):
                    new_params_derivs[i, :len(inds)] = results.dqf_dq0[ind, inds]

            all_params.append(new_params)
            all_params_derivatives.append(new_params_derivs)

        all_params_tensor = torch.stack(all_params)
        all_params_derivatives_tensor = torch.stack(all_params_derivatives)

        self._rsample = diff_nlp.PassthroughWithGradient.apply(
            pre_projection_params, all_params_tensor,
            all_params_derivatives_tensor)
        self._distrib = dist.Normal(
            self._rsample,
            variance.expand(event_shape)).to_event(1)

        super(ProjectToFeasibilityDist, self).__init__(
            batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(
            ProjectToFeasibilityDist, _instance)
        batch_shape = torch.Size(batch_shape)
        new._rsample = self._rsample.expand(batch_shape + self.event_shape)
        new._distrib = self._distrib.expand(batch_shape)
        super(ProjectToFeasibilityDist, new).__init__(
            batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @torch.distributions.constraints.dependent_property
    def support(self):
        return torch.distributions.constraints.real

    def log_prob(self, value):
        assert value.shape[-1] == self.event_shape[0]
        if value.dim() > 1:
            assert value.shape[0] == self.batch_shape[0]

        # Difference of each new value from the projected point
        diff_values = value - self._rsample
        # Project that into the infeasible cone -- we're moving out
        # towards local infeasible space if any of these inner products
        # is positive.
        # I'll use a "large" threshold of violation for now...
        # TODO(gizatt) What's a good val for eps?
        return self._distrib.log_prob(value)

    def rsample(self, sample_shape=torch.Size()):
        return self._rsample.expand(sample_shape + self.batch_shape + self.event_shape)


class FixedEnvironmentWithProjectionModel():
    ''' Model the generation of a prototype environment where
    the # of objects and their classes don't change, but
    their params do. '''

    def __init__(self, dataset, use_projection=False):
        assert(isinstance(dataset, dataset_utils.ScenesDatasetVectorized))
        self.dataset = dataset
        self.prototype_environment = dataset.get_subsample_dataset([0]).convert_to_yaml()[0]
        self.context_size = 50
        self.class_general_encoded_size = 50
        self.num_objects = self.prototype_environment["n_objects"]
        self.num_classes = dataset.get_num_classes()
        self.num_params_by_class = dataset.get_num_params_by_class()
        self.use_projection = use_projection

        # Class-specific encoders and generators
        self.class_means_generators = []
        self.class_vars_generators = []
        self.class_encoders = []
        for class_i in range(self.num_classes):
            # Generator
            input_size = self.context_size
            output_size = self.num_params_by_class[class_i]
            generator_H = 50
            self.class_means_generators.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, output_size),
                )
            )
            self.class_vars_generators.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, output_size),
                    torch.nn.Softplus()
                )
            )
            # Encoder
            input_size = self.num_params_by_class[class_i]
            output_size = self.context_size
            encoder_H = 50
            self.class_encoders.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, encoder_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_H, encoder_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_H, output_size),
                )
            )

        self.context_updater = torch.nn.GRU(
            input_size=self.context_size,
            hidden_size=50)

    def _create_empty_context(self, minibatch_size):
        return torch.zeros(minibatch_size, self.context_size)

    def _extract_and_check_class(self, data, object_i):
        new_class = torch.tensor(
            self.dataset.get_class_id_from_name(
                self.prototype_environment["obj_%04d" % object_i]["class"]))
        if data is None:
            return new_class
        assert(np.all(data.classes[..., object_i] == new_class))
        return new_class.expand(data.shape[0])

    def _extract_params(self, data, object_i):
        if data is None:
            return [None]*self.num_classes
        params = [p[..., object_i, :] for p in data.params_by_class]
        return params

    def _sample_class_specific_generators(
            self, object_i, minibatch_size, context, new_class,
            observed_params, generated_data):
        # To operate in batch, this needs to sample from every
        # class-specific generator, but mask off the ones that weren't
        # actually selected.
        # Unfortunately, that's pretty ugly and super-wasteful, but I
        # suspect the batch speedup will still be worth it.

        sampled_params_components = []
        # TODO(gizatt) Use plate here?
        for class_i in range(self.num_classes):

            if self.use_projection:
                def sample_pre_projection_params():
                    params_means = self.class_means_generators[class_i](context).view(minibatch_size, self.num_params_by_class[class_i])
                    params_vars = self.class_vars_generators[class_i](context).view(minibatch_size, self.num_params_by_class[class_i])

                    pre_projection_params = pyro.sample(
                        "pre_projection_params_{}_{}".format(object_i, class_i),
                        dist.Normal(params_means, params_vars).to_event(1))

                    projection_dist = ProjectToFeasibilityDist(
                        pre_projection_params, class_i,
                        object_i, context, new_class, generated_data)

                    pyro.sample("params_{}_{}".format(object_i, class_i),
                                projection_dist, obs=observed_params[class_i])

            else:
                def sample_params():
                    params_means = self.class_means_generators[class_i](context).view(minibatch_size, self.num_params_by_class[class_i])
                    params_vars = self.class_vars_generators[class_i](context).view(minibatch_size, self.num_params_by_class[class_i])
                    return pyro.sample(
                        "params_{}_{}".format(object_i, class_i),
                        dist.Normal(params_means, params_vars).to_event(1),
                        obs=observed_params[class_i])

                # Sample everything in batch -- meaning we'll sample
                # every class even though we know what class we wanted to sample.
                # Mask so that only the appropriate ones show up in the objective.
                sampled_params_components.append(
                    poutine.mask(sample_params, new_class == class_i)())

        return sampled_params_components

    def _apply_class_specific_encoders(
            self, context, new_class, params):
        # No masking is needed because the encoders are deterministic.
        # Some clever splitting off to each encoder is still needed,
        # though...*

        encoded_components = [
            self.class_encoders[class_i](params[class_i])
            for class_i in range(self.num_classes)
        ]

        print("New class: ", new_class)
        print("New class shape: ", new_class.shape)
        one_hot = torch.zeros(new_class.shape + (self.num_classes,))
        one_hot.scatter_(1, new_class.unsqueeze(1), 1)
        one_hot = one_hot.view(-1, 1, self.num_classes)
        stacked_components = torch.stack(encoded_components, dim=1)
        collapsed_components = one_hot.matmul(stacked_components).view(
            new_class.shape[0], self.context_size)
        return collapsed_components

    def _update_context(self, encoded_params, context):
        return self.context_updater(
            encoded_params.view(1, -1, self.class_general_encoded_size),
            context.view(1, -1, self.context_size))[-1]

    def _sample_single_object(self, object_i, data, batch_size,
                              context, generated_data):
        # Sample the new object type
        new_class = self._extract_and_check_class(data, object_i)

        # Generate an object of that type.
        observed_params = self._extract_params(
            data, object_i)
        sampled_params = self._sample_class_specific_generators(
            object_i, batch_size, context, new_class,
            observed_params, generated_data)
        # Update the context by encoding the new params
        # into a fixed-size vector through a class-specific encoder.
        encoded_params = self._apply_class_specific_encoders(
            context, new_class, sampled_params)
        context = self._update_context(
            encoded_params, context)

        # Keep going, after this?
        observed_keep_going = self._extract_keep_going(data, object_i)
        keep_going = self._sample_keep_going(
            object_i, batch_size, context, observed_keep_going)

        return keep_going, new_class, sampled_params, encoded_params, context

    def model(self, data=None, subsample_size=None):
        pyro.module("context_updater_module", self.context_updater, update_module_params=True)
        for class_i in range(self.num_classes):
            pyro.module("class_means_generator_module_{}".format(class_i),
                        self.class_means_generators[class_i], update_module_params=True)
            pyro.module("class_vars_generator_module_{}".format(class_i),
                        self.class_vars_generators[class_i], update_module_params=True)
            pyro.module("class_encoder_module_{}".format(class_i),
                        self.class_encoders[class_i], update_module_params=True)
        if data is None:
            data_batch_size = 1
        else:
            data_batch_size = data.batch_size

        generated_keep_going = []
        generated_classes = []
        generated_params_by_class = [[] for i in range(self.num_classes)]
        generated_encodings = []
        generated_contexts = []

        with pyro.plate('data', size=data_batch_size) as subsample_inds:
            if data is not None:
                data_sub = data.subsample(subsample_inds)
            minibatch_size = len(subsample_inds)
            context = self._create_empty_context(minibatch_size)

            generated_data = None

            for object_i in range(self.num_objects):
                # Do a generation step
                keep_going, new_class, sampled_params, encoded_params, context = \
                    self._sample_single_object(
                            object_i, data, minibatch_size,
                            context, generated_data)

                generated_keep_going.append(keep_going)
                generated_classes.append(new_class)
                for k in range(self.num_classes):
                    generated_params_by_class[k].append(sampled_params[k])
                generated_encodings.append(encoded_params)
                generated_contexts.append(context)

            # Generated environments so far.
            generated_data = dataset_utils.VectorizedEnvironments(
                batch_size=minibatch_size,
                keep_going=torch.ones(minibatch_size, object_i+1),
                classes=torch.stack(generated_classes),
                params_by_class=[
                    torch.stack(p, -2) for p in generated_params_by_class],
                dataset=self.dataset)

        return (generated_data,
                torch.stack(generated_encodings, -1),
                torch.stack(generated_contexts, -1))


if __name__ == "__main__":
    pyro.enable_validation(True)

    file = "../data/planar_bin/planar_bin_static_scenes.yaml"
    scenes_dataset = dataset_utils.ScenesDatasetVectorized(file)
    data = scenes_dataset.get_full_dataset()

    model = FixedEnvironmentWithProjectionModel(scenes_dataset)

    log_dir = "runs/pmomc2/" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%m-%s")
    writer = SummaryWriter(log_dir)

    start = time.time()
    pyro.clear_param_store()
    generated_data, generated_encodings, generated_contexts = model.model()

    # Convert that data back to a YAML environment, which is easier to
    # handle.
    scene_yaml = scenes_dataset.convert_vectorized_environment_to_yaml(
        generated_data)
    dataset_utils.DrawYamlEnvironment(scene_yaml[0], "planar_bin")
    end = time.time()
    print "Time to generate and draw one scene: %fs" % (end - start)

    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace()
    trace.compute_log_prob()
    end = time.time()
    #print(trace.format_shapes())
    print "Time to run and do log probs with no args: %fs" % (end - start)
#
    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace(
        dataset_utils.SubsampleVectorizedEnvironments(
            data, [0, 1, 2]))
    trace.compute_log_prob()
    end = time.time()
    print "Time to run and do log probs with 3 datapoints: %fs" % (end - start)
    #print(trace.format_shapes())
#

    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace(data)
    trace.compute_log_prob()
    end = time.time()
    print "Time to run and do log probs with %d datapoints: %fs" % (
        data.batch_size, end - start)

    #pyro.clear_param_store()
    #trace = poutine.trace(model.generation_guide).get_trace(data, subsample_size=5)
    #trace.compute_log_prob()
    #print "GUIDE WITH ARGS RUN SUCCESSFULLY"
    ##print(trace.format_shapes())
#