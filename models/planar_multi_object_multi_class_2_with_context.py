from collections import namedtuple
from copy import deepcopy
import datetime
import io
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import multiprocessing
try:  # Python 2
    import Queue as queue
except ImportError:  # Python 3
    import queue
# Must be before torch.
import pydrake
import numpy as np
import sys
import time
import traceback

#from tensorboardX import SummaryWriter

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
from pyro.nn import AutoRegressiveNN
import torch
import torch.distributions.constraints as constraints

import scene_generation.data.dataset_utils as dataset_utils
import scene_generation.differentiable_nlp as diff_nlp


class ProjectionWorker(object):
    """Multiprocess worker."""

    def __init__(self, input_queue, output_queue,
                 termination_event, error_queue=None,
                 no_constraints=False):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.termination_event = termination_event
        self.error_queue = error_queue
        self.no_constraints = no_constraints

    def _do_projection_inner_work(
            self, env, object_i, base_environment_type, new_params):
        builder, mbp, scene_graph, q0 = dataset_utils.BuildMbpAndSgFromYamlEnvironment(
            env, base_environment_type)
        diagram = builder.Build()

        diagram_context = diagram.CreateDefaultContext()
        mbp_context = diagram.GetMutableSubsystemContext(
            mbp, diagram_context)

        # Pre-compute the "active" decision variable indices
        if base_environment_type in ["planar_bin", "planar_tabletop", "table_setting"]:
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

        if self.no_constraints:
            constraints = [
                diff_nlp.SetArguments(diff_nlp.AddJointPositionBounds, q_min=q_min, q_max=q_max)
            ]
        else:
            constraints = [
                diff_nlp.SetArguments(diff_nlp.AddMinimumDistanceConstraint, minimum_distance=0.01),
                diff_nlp.SetArguments(diff_nlp.AddJointPositionBounds, q_min=q_min, q_max=q_max)
            ]
        results = diff_nlp.ProjectMBPToFeasibility(
            q0, mbp, mbp_context, constraints,
            compute_gradients_at_solution=True, verbose=0)

        new_params[:len(inds)] = results.qf[inds]
        new_params_derivs = np.eye(len(new_params))
        for i, ind in enumerate(inds):
            new_params_derivs[i, :len(inds)] = results.dqf_dq0[ind, inds]
        return new_params, new_params_derivs

    def __call__(self, worker_index):
        while ((not self.input_queue.empty()) or
               (not self.termination_event.is_set())):
            try:
                new_data = None
                try:
                    new_data = self.input_queue.get(False)
                except queue.Empty:
                    pass

                if new_data is None:
                    time.sleep(0)
                    continue

                k, env, object_i, base_environment_type, new_params = new_data
                new_params, new_params_derivs = self._do_projection_inner_work(
                    env, object_i, base_environment_type, new_params)
                self.output_queue.put((k, new_params, new_params_derivs))

            except Exception as e:
                if self.error_queue:
                    self.error_queue.put((worker_index, e))
                else:
                    print("Unhandled exception in ProjectionWorker #%d" % worker_index)
                    traceback.print_exc()


class ProjectToFeasibilityDist(dist.TorchDistribution):
    has_rsample = True
    arg_constraints = {"pre_projection_params": torch.distributions.constraints.real}

    def __init__(self, pre_projection_params, class_i,
                 object_i, context, new_class, generated_data,
                 base_environment_type, no_constraints=False,
                 worker_pool=None, variance=None):
        batch_shape = pre_projection_params.shape[:-1]
        event_shape = pre_projection_params.shape[-1:]
        dtype = pre_projection_params.dtype
        if variance is None:
            variance = torch.tensor(0.05, dtype=dtype)
        # TODO(gizatt) Handle when len(batch_shape) > 1?
        if len(batch_shape) > 1:
            raise NotImplementedError("Don't know how to handle"
                                      "multidimensional batch.")

        ones = np.ones([batch_shape[0], 1])

        tentative_generated_data = dataset_utils.VectorizedEnvironments(
            batch_size=generated_data.batch_size,
            keep_going=np.hstack([generated_data.keep_going.cpu().detach().numpy(), ones*0]),
            classes=np.hstack([generated_data.classes.cpu().detach().numpy(), ones*class_i]).astype(np.int),
            params_by_class=[p.cpu().detach().numpy() for p in generated_data.params_by_class],
            dataset=generated_data.dataset)
        # Cram the generated parameters onto the appropriate params_by_class
        # element. Don't bother updating the other class params, as they won't
        # be read.
        # params_by_class[class_i].shape() =
        #   [minibatch_size, #_object_so_far, #_params_for_this_class]
        tentative_generated_data.params_by_class[class_i] = np.concatenate(
            [tentative_generated_data.params_by_class[class_i],
             pre_projection_params.cpu().detach().numpy().reshape(batch_shape[0], 1, -1)], axis=1)

        all_params = [None] * batch_shape[0]
        all_params_derivatives = [None] * batch_shape[0]

        if worker_pool:
            worker_manager = multiprocessing.Manager()
            input_queue = worker_manager.Queue()
            output_queue = worker_manager.Queue()
            termination_event = worker_manager.Event()
            result = worker_pool.map_async(
                ProjectionWorker(input_queue=input_queue,
                                 output_queue=output_queue,
                                 termination_event=termination_event,
                                 no_constraints=no_constraints),
                range(min(worker_pool._processes, batch_shape[0])))
            for k in range(batch_shape[0]):
                # Short circuit if class_i and new_class don't match,
                # or if keep_going has previously been zero.
                # This will likely happen in batching quite frequently,
                # and saves lots of unnecessary projections. What is produced
                # doesn't matter, as evaluated probabilities will be masked out.
                new_params = pre_projection_params[k, :].clone()
                if (not no_constraints and class_i == new_class[k] and (
                        object_i == 0 or
                        np.all(generated_data.keep_going[k, :].cpu().detach().numpy() != 0.))):
                    env = tentative_generated_data.subsample([k]).convert_to_yaml()[0]
                    input_queue.put((k, env, object_i, base_environment_type, new_params.cpu().detach().numpy()))

                else:
                    all_params[k] = new_params
                    all_params_derivatives[k] = torch.eye(event_shape[0], dtype=dtype)

            termination_event.set()
            while (True):
                if result.ready() and output_queue.empty():
                    break
                if not output_queue.empty():
                    index, new_params, new_params_derivatives = output_queue.get(
                        timeout=0)
                    all_params[index] = torch.tensor(new_params, dtype=dtype)
                    all_params_derivatives[index] = torch.tensor(new_params_derivatives, dtype=dtype)
                time.sleep(0)

        else:
            dummy_worker = ProjectionWorker(None, None, None, no_constraints=no_constraints)
            for k in range(batch_shape[0]):
                new_params = pre_projection_params[k, :].clone()
                if (not no_constraints and
                        class_i == new_class[k] and
                        (object_i == 0 or
                         np.all(generated_data.keep_going[k, :].cpu().detach().numpy() != 0.))):
                    env = tentative_generated_data.subsample([k]).convert_to_yaml()[0]
                    new_params, new_params_derivs = dummy_worker._do_projection_inner_work(
                         env, object_i, base_environment_type, new_params.cpu().detach().numpy())
                    all_params[k] = torch.tensor(new_params, dtype=dtype)
                    all_params_derivatives[k] = torch.tensor(new_params_derivs, dtype=dtype)
                else:
                    all_params[k] = new_params
                    all_params_derivatives[k] = torch.eye(event_shape[0], dtype=dtype)

        all_params_tensor = torch.stack(all_params)
        all_params_derivatives_tensor = torch.stack(all_params_derivatives)

        self._sample_mean = diff_nlp.PassthroughWithGradient.apply(
            pre_projection_params, all_params_tensor,
            all_params_derivatives_tensor)
        self._distrib = dist.Normal(
            self._sample_mean,
            variance).to_event(1)

        super(ProjectToFeasibilityDist, self).__init__(
            batch_shape, event_shape, validate_args=False)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(
            ProjectToFeasibilityDist, _instance)
        batch_shape = torch.Size(batch_shape)
        new._sample_mean = self._sample_mean.expand(batch_shape + self.event_shape)
        new._distrib = self._distrib.expand(batch_shape)
        super(ProjectToFeasibilityDist, new).__init__(
            batch_shape, self.event_shape, validate_args=False)
        return new

    @torch.distributions.constraints.dependent_property
    def support(self):
        return torch.distributions.constraints.real

    def log_prob(self, value):
        assert value.shape[-1] == self.event_shape[0]
        if value.dim() > 1:
            assert value.shape[0] == self.batch_shape[0]

        # Difference of each new value from the projected point
        diff_values = value - self._sample_mean
        # Project that into the infeasible cone -- we're moving out
        # towards local infeasible space if any of these inner products
        # is positive.
        # I'll use a "large" threshold of violation for now...
        # TODO(gizatt) What's a good val for eps?
        return self._distrib.log_prob(value)

    def rsample(self, sample_shape=torch.Size()):
        return self._distrib.rsample(sample_shape=sample_shape)


class MultiObjectMultiClassModelWithContext():
    def __init__(self, dataset, use_projection=False, n_processes=20):
        assert(isinstance(dataset, dataset_utils.ScenesDatasetVectorized))
        self.use_projection = use_projection
        self.dataset = dataset
        self.base_environment_type = dataset.base_environment_type
        self.context_size = 50
        self.class_general_encoded_size = 50
        self.max_num_objects = dataset.get_max_num_objects()
        self.num_classes = dataset.get_num_classes()
        self.num_params_by_class = dataset.get_num_params_by_class()

        self.worker_pool = None # multiprocessing.Pool(processes=n_processes)

        # Class-specific encoders and generators
        self.class_flows = []
        self.class_tf_dists = []
        self.class_encoders = []
        self.class_param_controllers = []
        # Must seed RNG carefully during creation of AutoRegNNs.
        torch.manual_seed(42)
        for class_i in range(self.num_classes):
            # Generator
            input_size = self.context_size
            output_size = self.num_params_by_class[class_i]
            
            generator_H = 100
            self.class_param_controllers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, generator_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(generator_H, output_size*2), # Mean and Var
                )
            )
            
            flow_H = [50, 50]
            flow_layers = 16

            base_dist = dist.Normal(torch.zeros(output_size),
                                    torch.ones(output_size)).to_event(1)
            flows = []
            for flow_i in range(flow_layers):
                flows.append(
                    dist.MaskedAutoregressiveFlow(
                        AutoRegressiveNN(output_size,
                                         flow_H,
                                         observed_dim=input_size)))
            tf_dist = dist.TransformedDistribution(base_dist, flows)
            self.class_flows.append(flows)
            self.class_tf_dists.append(tf_dist)

            # Encoder
            input_size = self.num_params_by_class[class_i]
            output_size = self.class_general_encoded_size
            encoder_H = 100
            self.class_encoders.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, encoder_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_H, encoder_H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(encoder_H, output_size),
                )
            )

        torch.manual_seed(time.time()*1000)
        self.context_updater = torch.nn.GRU(
            input_size=self.class_general_encoded_size,
            hidden_size=self.context_size)

        # Keep going predictor:
        # regresses bernoulli keep_going weight
        # from current context
        keep_going_H = 50
        self.keep_going_controller = torch.nn.Sequential(
            torch.nn.Linear(self.context_size, keep_going_H),
            torch.nn.ReLU(),
            torch.nn.Linear(keep_going_H, keep_going_H),
            torch.nn.ReLU(),
            torch.nn.Linear(keep_going_H, 1),
            torch.nn.Sigmoid()
        )
        # Class predictor:
        # regresses categorical weights
        # from current context
        class_H = 50
        self.class_controller = torch.nn.Sequential(
            torch.nn.Linear(self.context_size, class_H),
            torch.nn.ReLU(),
            torch.nn.Linear(keep_going_H, class_H),
            torch.nn.ReLU(),
            torch.nn.Linear(keep_going_H, self.num_classes),
            torch.nn.Softmax()
        )

        # Param inversion for guide:
        # Predicts pre-projection params from post-projection params.
        self.class_guides_flows = []
        self.class_guides_tf_dists = []
        self.class_guides = []
        for class_i in range(self.num_classes):
            # Generator
            input_size = self.context_size + self.num_params_by_class[class_i]
            output_size = self.num_params_by_class[class_i]*2 # Mean + var

            H = 100
            self.class_guides.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H, H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H, output_size),
                )
            )


            #flow_H = [100, 100]
            #flow_layers = 16
#
            #base_dist = dist.Normal(torch.zeros(output_size),
            #                        torch.ones(output_size)).to_event(1)
            #flows = []
            #for flow_i in range(flow_layers):
            #    # TODO(gizatt) Flip to the other flow, since this is in the guide.
            #    flows.append(
            #        dist.MaskedAutoregressiveFlow(
            #            AutoRegressiveNN(output_size,
            #                             flow_H,
            #                             observed_dim=input_size)))
            #tf_dist = dist.TransformedDistribution(base_dist, flows)
            #self.class_guides_flows.append(flows)
            #self.class_guides_tf_dists.append(tf_dist)

    def _create_empty_context(self, minibatch_size):
        return torch.zeros(minibatch_size, self.context_size)

    def _extract_keep_going(self, data, object_i):
        if data is None:
            return None
        return data.keep_going[..., object_i]

    def _extract_new_class(self, data, object_i):
        if data is None:
            return None
        return data.classes[..., object_i]

    def _extract_params(self, data, object_i):
        if data is None:
            return [None]*self.num_classes
        params = [p[..., object_i, :] for p in data.params_by_class]
        return params

    def _sample_keep_going(self, object_i, minibatch_size, context,
                           observed_keep_going):
        # This sampling strategy supports a geometric distribution
        # over # of objects.
        #keep_going_params = pyro.param(
        #    "keep_going_weights", torch.ones(1)*0.5,
        #    constraint=constraints.interval(0, 1))
        #keep_going_params = pyro.param(
        #    "keep_going_weights".format(object_i),
        #    torch.ones(self.max_num_objects)*0.9,
        #    constraint=constraints.interval(0, 1))[object_i]
        keep_going_params = self.keep_going_controller(context).view(minibatch_size)
        #keep_going_params = pyro.param(
        #    "keep_going_weights", torch.ones(1)*0.5,
        #    constraint=constraints.interval(0, 1))
        return pyro.sample("keep_going_%d" % object_i,
                           dist.Bernoulli(keep_going_params),
                           obs=observed_keep_going) == 1.

    def _sample_new_class(self, object_i, minibatch_size, context,
                          observed_new_class):
        # TODO(gizatt) An alternative generator could carry around
        # the predicted class weights on their own, and use them
        # to collect the results of the encoders into an
        # updated context to avoid the cast-to-int that has
        # to happen here. The actual thing "generated" would no
        # longer be clear, but that's irrelevant in the context
        # of training this thing, isn't it?
        #new_class_params = pyro.param(
        #    "new_class_weights",
        #    torch.ones(self.num_classes)/self.num_classes,
        #    constraint=constraints.simplex)
        new_class_params = self.class_controller(context)[0]
        return pyro.sample("new_class_%d" % object_i,
                           dist.Categorical(new_class_params),
                           obs=observed_new_class)

    def _sample_class_specific_generators(
            self, object_i, minibatch_size, context, new_class,
            observed_params, generated_data):
        # To operate in batch, this needs to sample from every
        # class-specific generator, but mask off the ones that weren't
        # actually selected.
        # Unfortunately, that's pretty ugly and super-wasteful, but I
        # suspect the batch speedup will still be worth it.

        sampled_params_components = []
        for class_i in range(self.num_classes):
            if self.use_projection:
                def sample_params():
                    #for flow in self.class_flows[class_i]:
                    #    flow.set_z(context)
                    #tf_dist = self.class_tf_dists[class_i]
                    param_mean_and_var = self.class_param_controllers[class_i](context)
                    n = self.num_params_by_class[class_i]
                    #tf_dist = dist.Normal(param_mean_and_var[:, :n],
                    #                      torch.nn.Softplus()(param_mean_and_var[:, n:])).to_event(1)
                    #pre_projection_params = pyro.sample(
                    #    "pre_projection_params_{}_{}".format(
                    #        object_i, class_i), tf_dist)

                    projection_dist = ProjectToFeasibilityDist(
                        param_mean_and_var[:, :n], class_i,
                        object_i, context, new_class, generated_data,
                        self.base_environment_type,
                        no_constraints=False,
                        worker_pool=self.worker_pool,
                        variance=torch.nn.Softplus()(param_mean_and_var[:, n:]))

                    return pyro.sample("params_{}_{}".format(object_i,
                                                             class_i),
                                       projection_dist,
                                       obs=observed_params[class_i])
            else:
                def sample_params():
                    #for flow in self.class_flows[class_i]:
                    #    flow.set_z(context)
                    #tf_dist = self.class_tf_dists[class_i]
                    param_mean_and_var = self.class_param_controllers[class_i](context)
                    n = self.num_params_by_class[class_i]
                    tf_dist = dist.Normal(param_mean_and_var[:, :n],
                                          torch.nn.Softplus()(param_mean_and_var[:, n:])).to_event(1)
                    return pyro.sample(
                        "params_{}_{}".format(object_i, class_i),
                        tf_dist, obs=observed_params[class_i])

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
            context.view(1, -1, self.context_size))[-1].view(-1, self.context_size)

    def _sample_single_object(self, object_i, data, batch_size,
                              context, generated_data):
        # Sample the new object type
        observed_new_class = self._extract_new_class(data, object_i)
        new_class = self._sample_new_class(
            object_i, batch_size, context, observed_new_class)

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

    def declare_all_modules(self):
        pyro.module("context_updater_module", self.context_updater, update_module_params=True)
        pyro.module("keep_going_controller_module", self.keep_going_controller, update_module_params=True)
        pyro.module("class_controller_module", self.class_controller, update_module_params=True)
        for class_i in range(self.num_classes):
            #for flow_i, flow in enumerate(self.class_flows[class_i]):
            #    pyro.module("class_{}_flow_{}".format(class_i, flow_i), flow, update_module_params=True)
            pyro.module("class_encoder_module_{}".format(class_i),
                        self.class_encoders[class_i], update_module_params=True)
            pyro.module("class_param_controller_{}".format(class_i),
                        self.class_param_controllers[class_i], update_module_params=True)
            if self.use_projection:
                #for flow_i, flow in enumerate(self.class_guides_flows[class_i]):
                #    pyro.module("class_guide_module_{}_flow_{}".format(class_i, flow_i),
                #                flow, update_module_params=True)
                pyro.module("class_guide_module_{}".format(class_i),
                            self.class_guides[class_i], update_module_params=True)


    def model(self, data=None, subsample_size=None):
        self.declare_all_modules()
        if data is None:
            data_batch_size = 50
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
            else:
                data_sub = None
            minibatch_size = len(subsample_inds)
            context = self._create_empty_context(minibatch_size)

            # Because any given row in the batch might produce
            # all of the objects, we must iterate over all of the
            # generation steps and mask if we're on a stop
            # where the generator said to stop.
            not_terminated = torch.ones(minibatch_size) == 1.
            generated_data = dataset_utils.VectorizedEnvironments(
                batch_size=minibatch_size,
                keep_going=torch.empty(minibatch_size, 0),
                classes=torch.empty(minibatch_size, 0),
                params_by_class=[torch.empty(minibatch_size, 0, p) for p in self.num_params_by_class],
                dataset=self.dataset)
            for object_i in range(self.max_num_objects):
                # Do a generation step
                keep_going, new_class, sampled_params, encoded_params, context = \
                    poutine.mask(
                        lambda: self._sample_single_object(
                            object_i, data_sub, minibatch_size,
                            context, generated_data),
                        not_terminated)()

                not_terminated = not_terminated * keep_going
                generated_keep_going.append(keep_going)
                generated_classes.append(new_class)
                for k in range(self.num_classes):
                    generated_params_by_class[k].append(sampled_params[k])
                generated_encodings.append(encoded_params)
                generated_contexts.append(context)

                # Reassemble the output VectorizedEnvironments
                generated_data = dataset_utils.VectorizedEnvironments(
                    batch_size=minibatch_size,
                    keep_going=torch.stack(generated_keep_going, -1),
                    classes=torch.stack(generated_classes, -1),
                    params_by_class=[
                        torch.stack(p, -2) for p in generated_params_by_class],
                    dataset=self.dataset)

        return (generated_data,
                torch.stack(generated_encodings, -1),
                torch.stack(generated_contexts, -1))

    def guide(self, data, subsample_size=None):
        self.declare_all_modules()

        if not data:
            raise InvalidArgumentError("Guide must be handed data.")
        data_batch_size = data.batch_size
        if subsample_size:
            minibatch_size = subsample_size
        else:
            minibatch_size = data_batch_size

        with pyro.plate('data', size=data_batch_size, subsample_size=subsample_size) as subsample_inds:
            if 0 and self.use_projection is True:
                # NOTE: Can't use AMORTIZED = False on test data.
                data_sub = data.subsample(subsample_inds)
                context = self._create_empty_context(minibatch_size)
                for object_i in range(self.max_num_objects):
                    real_params_components = []
                    real_class = self._extract_new_class(data_sub, object_i).type(torch.LongTensor)
                    for class_i in range(self.num_classes):
                        real_params = data_sub.params_by_class[class_i][:, object_i, :]
                        real_params_components.append(real_params)
                        #for flow in self.class_guides_flows[class_i]:
                        #    flow.set_z(real_params)
                        #tf_dist = self.class_guides_tf_dists[class_i]
                        n_params = self.num_params_by_class[class_i]
                        preproj_pose_mean_var = self.class_guides[class_i](
                            torch.cat([context, real_params], dim=-1))
                        estimate_mean = preproj_pose_mean_var[:, :n_params]
                        estimate_var = torch.nn.Softplus()(preproj_pose_mean_var[:, n_params:])
                        #estimate_var[:, 3:] = real_params[:, 3:]
                        #estimate_var[:, 3:] = 1E-5
                        preproj_param_dist = dist.Normal(estimate_mean, estimate_var).to_event(1)
                        poutine.mask(
                            lambda:
                                pyro.sample(
                                    "pre_projection_params_{}_{}".format(object_i, class_i),
                                    preproj_param_dist),
                            real_class == class_i
                        )()
                    encoded_real_params = self._apply_class_specific_encoders(
                        context, real_class, real_params_components)
                    context = self._update_context(
                        encoded_real_params, context)



if __name__ == "__main__":
    pyro.enable_validation(True)

    DATA_DIR_TRAIN = "/home/gizatt/projects/scene_generation/data/planar_tabletop/planar_tabletop_lines_scenes_train/"

    scenes_dataset = dataset_utils.ScenesDatasetVectorized(
        DATA_DIR_TRAIN, max_num_objects=5,
        base_environment_type="planar_tabletop")
    data = scenes_dataset.get_full_dataset()

    model = MultiObjectMultiClassModelWithContext(scenes_dataset, use_projection=True)

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
        data.subsample([0, 1, 2]))
    trace.compute_log_prob()
    end = time.time()
    print "Time to run and do log probs with 3 datapoints: %fs" % (end - start)
    #print(trace.format_shapes())
#

    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace(data.subsample(range(100)))
    trace.compute_log_prob()
    end = time.time()
    print "Time to run and do log probs with %d datapoints: %fs" % (
        100, end - start)

    #pyro.clear_param_store()
    #trace = poutine.trace(model.generation_guide).get_trace(data, subsample_size=5)
    #trace.compute_log_prob()
    #print "GUIDE WITH ARGS RUN SUCCESSFULLY"
    ##print(trace.format_shapes())
#