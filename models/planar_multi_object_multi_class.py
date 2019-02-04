from collections import namedtuple
import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

import scene_generation.data.planar_scene_arrangement_utils as psa_utils


class DataWrapperForObs:
    # Convenience wrapper on data:
    # If data's batch dimension is length-0,
    # we must be running the model in generative mode,
    # so any slicing returns None.
    # Otherwise, pass through slicing to the real data.
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if self.data.shape[0] > 0:
            return self.data[key]
        else:
            return None


def expand_partial_pose_to_full_pose(this_pose):
    full_poses = torch.zeros(this_pose.shape[0], 6,
                             requires_grad=False,
                             dtype=this_pose.dtype)
    full_poses[:, 0] = this_pose[:, 0]
    full_poses[:, 1] = this_pose[:, 1]
    full_poses[:, 5] = this_pose[:, 2]
    return full_poses


def collapse_full_pose_to_partial_pose(pose):
    partial_pose = torch.zeros(pose.shape[0], 3, requires_grad=False,
                               dtype=pose.dtype)
    partial_pose[:, 0] = pose[:, 0]
    partial_pose[:, 1] = pose[:, 1]
    partial_pose[:, 2] = pose[:, 5]
    return partial_pose


VectorizedEnvironments = namedtuple(
    "VectorizedEnvironments",
    ["idents", "poses", "present", "n_samples"], verbose=False)


class ObjectWorldPriorDescription:
    ''' Prior distribution container for each object class over the ground '''
    def __init__(self, class_name,
                 prior_means_by_class,
                 prior_vars_by_class):
        # TODO(gizatt) Switch this to Normal-inverse-Wishart_distribution
        # if/when that becomes supported by Pyro.
        self.mean = pyro.sample(
            '%s_mean' % class_name,
            dist.Normal(prior_means_by_class[class_name]["mean"],
                        prior_means_by_class[class_name]["var"]).to_event(1))
        self.var = pyro.sample(
            '%s_var' % class_name,
            dist.LogNormal(
                prior_vars_by_class[class_name]["mean"],
                prior_vars_by_class[class_name]["var"]).to_event(1))
        self.dist = dist.MultivariateNormal(self.mean, torch.diag(self.var))


class MultiObjectMultiClassModel():
    def __init__(self, use_projection=True, noisy_projection=False,
                 use_amortization=True,
                 max_num_objects=10, min_num_objects=0):
        self.use_projection = use_projection
        self.noisy_projection = noisy_projection
        self.use_amortization = use_amortization
        self.max_num_objects = max_num_objects
        self.min_num_objects = min_num_objects
        self._prepareObjectClasses()
        self._prepareInferenceModule()
        self.rbts_cache = {}

    def _prepareObjectClasses(self):
        self.object_classes = ["small_box"]
        self.object_class_to_index = {}
        for i, obj in enumerate(self.object_classes):
            self.object_class_to_index[obj] = i
        self.n_object_classes = len(self.object_classes)

    def _prepareInferenceModule(self):
        H = 10
        self.inference_modules = {}
        for class_name in self.object_classes:
            self.inference_modules[class_name] = torch.nn.Sequential(
              torch.nn.Linear(3, H),
              torch.nn.ReLU(),
              torch.nn.Linear(H, 3),
            )

    def vectorizeEnvironments(self, envs):
        # Vectorization into a
        # poses (order x y theta) and
        # idents (ci) per object
        n_environments = len(envs)
        poses = torch.Tensor(n_environments, self.max_num_objects*3)
        poses[:, :] = 0.
        idents = torch.LongTensor(n_environments, self.max_num_objects)
        idents[:, :] = 0
        present = torch.Tensor(n_environments, self.max_num_objects)
        present[:, :] = 0
        n_samples = torch.Tensor(n_environments)
        n_samples[:] = 0
        for i, env in enumerate(envs):
            n_samples[i] = env["n_objects"]
            for k in range(self.n_object_classes):
                if k < env["n_objects"]:
                    obj = env["obj_%04d" % k]
                    idents[i, k] = self.object_class_to_index[obj["class"]]
                    poses[i, (k*3):(k*3+3)] = torch.Tensor(obj["pose"])
                    present[i, k] = 1
        return VectorizedEnvironments(
            idents=idents, poses=poses,
            present=present, n_samples=n_samples)

    def devectorizeEnvironments(self, data):
        if not isinstance(data, VectorizedEnvironments):
            raise ValueError("Expected VectorizedEnvironments input")
        envs = []
        for i in range(data.idents.shape[0]):
            env = {}
            n_objects = 0
            for k in range(data.n_samples[i]):
                obj = {
                    "pose": data.poses[i, (k*3):(k*3+3)].cpu().detach().numpy(),
                    "class": self.object_classes[data.idents[i, k].cpu().detach().item()]
                }
                env["obj_%04d" % k] = obj
            env["n_objects"] = data.n_samples[i]
            envs.append(env)
        return envs

    def _buildRbtFromGeneratedRowAndNewObject(
            self, generated_data, row_i, iter_i, ci):
        # Build a descriptor string to check into the cache
        # "<obj 0 class>_<obj 1 class>_<...>
        present = generated_data.present[row_i, 0:iter_i+1].cpu().detach().numpy()
        present[-1] = 1.
        if not np.allclose(present, 1.):
            return None
        previous_object_classes = generated_data.idents[row_i, 0:iter_i+1].cpu().detach().numpy()
        previous_object_classes[-1] = ci[row_i]
        class_string = "_".join([object_classes[cj] for cj in previous_object_classes])

        if class_string not in self.rbts_cache.keys():
            # Cache miss, generate the RBT
            env = {"n_objects": iter_i+1}
            for iter_j in range(iter_i+1):
                env["obj_%04d" % iter_j] = {
                    "class": object_classes[previous_object_classes[iter_j]],
                    "pose": np.zeros(3)
                }
            new_rbt, _ = psa_utils.build_rbt_from_summary(env)
            self.rbts_cache[class_string] = new_rbt

        return self.rbts_cache[class_string]

    def _SampleObjectClass(self, generated_data, i, keep_going, obs=None):
        ''' Given the currently generated data, the current object number,
            and the keep_going mask, decide what class to spawn next.

            I'm focusing for now on learning properties of classes rather than
            what class to spawn. In future, needs dependency on the environment
            or at least parameterization of this underlying distribution. '''
        return poutine.mask(lambda: pyro.sample("%d_class_assignment" % i,
                            dist.Categorical(torch.Tensor([1.0]).expand(
                                self.n_object_classes)),
                            obs=obs), keep_going)()

    def _SampleObjectPlacement(self, ci, generated_data, i,
                               keep_going, object_world_priors,
                               obs=None):
        ''' Given the current object class, the current generated data,
            the current object number, a keep_going mask, and the
            keep_going mask, sample a placement for the object. '''
        assert(ci.dim() == 1)

        pre_poses_by_class = []

        # Sample a pre-pose for each class so we can create the
        # mixture over classes. We're using discrete class
        # choices per sample, but this at least batches the sampling
        # part.
        # Not using a plate here so I can use AutoGuideList;
        # AutoGuideList does not support sequential pyro.plate.
        for k in pyro.plate("class_prior_mixture_%d" % (i),
                            self.n_object_classes):
            pre_poses_part = pyro.sample('pre_poses_%d_%d' % (i, k),
                                         object_world_priors[k].dist)
            pre_poses_by_class.append(pre_poses_part)

        # Turn ci indices into a one-hot so we can select out the poses.
        one_hot = torch.zeros(ci.shape + (self.n_object_classes,))
        one_hot.scatter_(1, ci.unsqueeze(1), 1)
        one_hot = one_hot.view(-1, 1, self.n_object_classes)
        pre_poses = one_hot.matmul(
            torch.stack(pre_poses_by_class, dim=1)).view(ci.shape[0], 3)

        # Replace projection with a fixed-variance operation that
        # doesn't move the pose far, with the same site name.
        #print ci, i, keep_going, pre_poses
        if not self.use_projection:
            new_pose = poutine.mask(
                lambda: pyro.sample(
                    "post_poses_%d" % i,
                    dist.Normal(
                        pre_poses,
                        scale=0.01*torch.ones(pre_poses.shape)
                    ).to_event(1),
                    obs=obs),
                keep_going)()
            return new_pose
        else:
            # Load in the previous generated poses as poses to be fixed
            # in the projection.
            if i > 0:
                q0_fixed = torch.cat([
                    expand_partial_pose_to_full_pose(
                        generated_data.poses[:, (k*3):(k*3+3)])
                    for k in range(i)], dim=-1)
            else:
                q0_fixed = None

            # Build an RBT for each row in the batch...
            rbts = [build_rbt_from_generated_row_and_new_object(
                        generated_data, k, i, ci)
                    for k in range(generated_data.poses.shape[0])]

            # Constrain free poses to have y,z,roll,pitch constant
            ik_constraints = [
                diff_nlp.object_at_specified_pose_constraint_constructor_factory(
                    i, np.array([0., 0., 0.5, 0., 0., 0.]),
                    np.array([1., 1., 0.5, 0., 0., 2*np.pi]))]
            projection_dist = diff_nlp.ProjectToFeasibilityWithIKAsDistribution(
                rbts, expand_partial_pose_to_full_pose(pre_poses),
                ik_constraints, 0.05, 0.01, noisy_projection=False,
                q0_fixed=q0_fixed, event_select_inds=torch.tensor([0, 1, 5]))

            projected_pose = poutine.mask(
                lambda: pyro.sample(
                    "post_poses_%d" % (i), projection_dist, obs=obs),
                keep_going)()
            return projected_pose[:, :]

    def model(self, data=None, subsample_size=None):
        # Instantiate priors on object means.
        prior_means_by_class = {}
        prior_vars_by_class = {}
        for class_name in self.object_classes:
            prior_means_by_class[class_name] = {
                "mean": torch.Tensor([0.5, 0.5, np.pi/2.]),
                "var": torch.Tensor([0.5, 0.5, np.pi/2.])
            }
            # Remember that these are LogNormal means
            prior_vars_by_class[class_name] = {
                "mean": torch.Tensor([0., 0., 0.]),
                "var": torch.Tensor([2., 2., 2.])
            }

        object_world_priors = []
        for class_name in self.object_classes:
            object_world_priors.append(
                ObjectWorldPriorDescription(
                    class_name, prior_means_by_class,
                    prior_vars_by_class))

        # Sample rates for the total number of objects we'll spawn.
        # (Includes an entry for 0 objects.)
        num_object_choices = self.max_num_objects - self.min_num_objects + 1
        sample_rates = pyro.sample(
            'num_objects_weights',
            dist.Dirichlet(torch.ones(
                num_object_choices)))
        sample_distribution = dist.Categorical(sample_rates)

        # Generate in vectorized form for easier batch conversion at the end
        data_batch_size = 1
        if data is not None:
            if not isinstance(data, VectorizedEnvironments):
                raise ValueError("Expected VectorizedEnvironments input")
            if (data.idents.shape[1] != self.max_num_objects and
               data.poses.shape[1] != self.max_num_objects*3):
                raise ValueError("Got unexpected data shape.")
            data_batch_size = data.idents.shape[0]

        with pyro.plate('data', size=data_batch_size) as subsample_inds:
            subsample_size = subsample_inds.shape[0]
            generated_data = VectorizedEnvironments(
                idents=torch.LongTensor(subsample_size, self.max_num_objects),
                poses=torch.Tensor(subsample_size, self.max_num_objects*3),
                present=torch.Tensor(subsample_size, self.max_num_objects),
                n_samples=torch.Tensor(subsample_size))
            generated_data.idents[:, :] = -1
            generated_data.poses[:, :] = 0
            generated_data.present[:, :] = 0
            generated_data.n_samples[:] = 0

            # Sample actual number of samples immediately
            # (since we can directly observe this from data)
            gt_n_samples = None
            if data is not None:
                gt_n_samples = data.n_samples[subsample_inds]
            num_samples = pyro.sample("num_samples", sample_distribution,
                                      obs=gt_n_samples)
            generated_data.n_samples[:] = num_samples

            # Go and spawn each object in order!
            for i in range(self.max_num_objects):
                gt_class = None
                gt_pose = None
                gt_keep_going = None
                if data is not None:
                    gt_class = data.idents[subsample_inds, i]
                    gt_pose = data.poses[subsample_inds, (i*3):(i*3+3)]
                    gt_keep_going = data.present[subsample_inds, i]

                keep_going = (i < num_samples)
                ci = self._SampleObjectClass(generated_data, i,
                                             keep_going, gt_class)
                pose = self._SampleObjectPlacement(
                    ci, generated_data, i, keep_going,
                    object_world_priors, gt_pose)

                # Fill in generated data appropriately
                generated_data.idents[:, i] = (
                    ci.view(-1).type(torch.long)*keep_going.type(torch.long))
                for k in range(3):
                    generated_data.poses[:, 3*i+k] = (
                        torch.Tensor(pose[:, k])
                        * keep_going.type(torch.float))
                generated_data.present[:, i] = keep_going
        return generated_data

    def generation_guide(self, data, subsample_size=None):
        for class_name in self.object_classes:
            pyro.module(class_name + "_inference_module",
                        self.inference_modules[class_name])

        # Instantiate priors on object means.
        prior_means_by_class = {}
        prior_vars_by_class = {}
        for class_name in self.object_classes:

            prior_means_by_class[class_name] = {
                "mean": pyro.param("auto_%s_mean_mean" % class_name,
                                   torch.rand(3)),
                "var": pyro.param("auto_%s_mean_var" % class_name,
                                  torch.rand(3),
                                  constraint=constraints.positive)
            }
            # Remember that these are LogNormal means
            prior_vars_by_class[class_name] = {
                "mean": pyro.param("auto_%s_var_mean" % class_name,
                                   torch.Tensor([0., 0., 0.])),
                "var": pyro.param("auto_%s_var_var" % class_name,
                                  torch.Tensor([2., 2., 2.]),
                                  constraint=constraints.positive),
            }

        object_world_priors = []
        for class_name in self.object_classes:
            object_world_priors.append(
                ObjectWorldPriorDescription(
                    class_name, prior_means_by_class,
                    prior_vars_by_class))

        # Sample rates for the total number of objects we'll spawn.
        # (Includes an entry for 0 objects.)
        num_object_choices = self.max_num_objects - self.min_num_objects + 1
        sample_rates = pyro.sample(
            'num_objects_weights',
            dist.Delta(
                pyro.param("auto_num_objects_weights",
                           torch.ones(num_object_choices)/num_object_choices,
                           constraint=constraints.simplex)).to_event(1))
        sample_distribution = dist.Categorical(sample_rates)

        # Generate in vectorized form for easier batch conversion at the end
        data_batch_size = 1
        if not isinstance(data, VectorizedEnvironments):
            raise ValueError("Expected VectorizedEnvironments input")
        if (data.idents.shape[1] != self.max_num_objects and
           data.poses.shape[1] != self.max_num_objects*3):
            raise ValueError("Got unexpected data shape.")
        data_batch_size = data.idents.shape[0]

        projection_var = pyro.param(
            "projection_var", torch.tensor([0.05, 0.05, 0.05]),
            constraint=constraints.positive)
        with pyro.plate('data', size=data_batch_size,
                        subsample_size=subsample_size) as subsample_inds:
            # Go and spawn each object in order!
            for i in range(self.max_num_objects):
                ci = data.idents[subsample_inds, i]
                pose = data.poses[subsample_inds, (i*3):(i*3+3)]
                keep_going = data.present[subsample_inds, i] == 1.

                # TODO: Put a sampled prior over the object classes
                # + make a param for it here.

                pre_poses_by_class = []

                # Sample a pre-pose for each class so we can create the
                # mixture over classes. We're using discrete class
                # choices per sample, but this at least batches the sampling
                # part.
                for k in pyro.plate("class_prior_mixture_%d" % (i),
                                    self.n_object_classes):
                    # Predict prior pose from appropriate network
                    pred_pose = self.inference_modules[self.object_classes[k]](
                        pose)
                    pre_poses_part = poutine.mask(lambda: pyro.sample(
                            'pre_poses_%d_%d' % (i, k),
                            dist.Normal(pred_pose, projection_var).to_event(1)),
                        keep_going)()

    def setup_complete_guide(self):
        # Doesn't work right now, because I don't know how to hide
        # the sequential plate + plate('data') from the autoguide
        # model -- they get through the block even though all samples sites
        # inside of those plates are blocked.
        full_guide = AutoGuideList(self.model)

        auto_guide_sites = ["num_objects_weights"]
        for object_name in self.object_classes:
            auto_guide_sites.append("%s_mean" % object_name)
            auto_guide_sites.append("%s_var" % object_name)
        print "AUTO GUIDE SITES: ", auto_guide_sites

        print "AutoGuide model trace: ", poutine.trace(poutine.block(self.model, expose=auto_guide_sites)).get_trace().format_shapes()
        #print "Gen guide trace: ", poutine.trace(poutine.block(self.generation_guide))
        auto_guide = AutoDiagonalNormal(
            poutine.block(self.model, expose=auto_guide_sites))
        print auto_guide
        full_guide.add(auto_guide)
        print "Added AutoDiagonalNormal"
        full_guide.add(self.generation_guide)
        print "Added gen guide"
        return full_guide


def draw_rbt(ax, rbt, q):
    psa_utils.draw_board_state(ax, rbt, q)

    patch = patches.Rectangle([0., 0.], 1., 1., fill=True, color=[0., 1., 0.],
                              linestyle='solid', linewidth=2, alpha=0.3)
    ax.add_patch(patch)


def draw_environment(environment, ax):
    rbt, q = psa_utils.build_rbt_from_summary(environment)
    draw_rbt(ax, rbt, q)


if __name__ == "__main__":
    pyro.enable_validation(True)
    writer = SummaryWriter()

    # These scenes include normally randomly distributed nonpenetrating
    # object arrangements with mu = 0.5, 0.5, pi and sigma=0.1, 0.1, pi/2
    DATA_BASE = "../data/single_planar_box_arrangements/"\
                "normal_random/fixed_2_objects"
    environments = psa_utils.load_environments(DATA_BASE)

    max_num_objects = 2
    model = MultiObjectMultiClassModel(
        use_projection=False,
        use_amortization=True,
        max_num_objects=max_num_objects,
        min_num_objects=0)

    plt.figure().set_size_inches(12, 12)
    print "Selection of environments from prior / generative model"
    N = 3

    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, i*N+j+1)
            draw_environment(model.devectorizeEnvironments(
                model.model())[0], plt.gca())
            plt.grid(True)
    plt.tight_layout()
    writer.add_figure('GeneratedEnvsSample', plt.gcf())
    plt.close()


    data = model.vectorizeEnvironments(environments["train"])

    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace()
    trace.compute_log_prob()
    print "MODEL WITH NO ARGS RUN SUCCESSFULLY"
    #print(trace.format_shapes())

    pyro.clear_param_store()
    trace = poutine.trace(model.model).get_trace(data)
    trace.compute_log_prob()
    print "MODEL WITH ARGS RUN SUCCESSFULLY"
    #print(trace.format_shapes())

    pyro.clear_param_store()
    trace = poutine.trace(model.generation_guide).get_trace(data, subsample_size=5)
    trace.compute_log_prob()
    print "GUIDE WITH ARGS RUN SUCCESSFULLY"
    #print(trace.format_shapes())

    interesting_params = ["auto_small_box_mean_mean",
                          "auto_small_box_var_mean",
                          #"auto_long_box_mean", "auto_long_box_var",
                          "auto_num_objects_weights"]

    def select_interesting():
        return dict((p, pyro.param(p)) for p in interesting_params)

    pyro.clear_param_store()
    optimizer = torch.optim.Adam
    def per_param_args(module_name, param_name):
        if module_name == 'inference_module':
            return {"lr": 0.01, 'betas': [0.9, 0.99]}
        else:
            return {'lr': 0.1, 'betas': [0.9, 0.99]}
    scheduler = pyro.optim.StepLR(
        {"optimizer": optimizer,
         'optim_args': per_param_args,
         'gamma': 0.25, 'step_size': 100})
    elbo = Trace_ELBO(max_plate_nesting=1, num_particles=4)
    svi = SVI(model.model, model.generation_guide, scheduler, loss=elbo)

    data = model.vectorizeEnvironments(environments["train"])
    data_valid = model.vectorizeEnvironments(environments["valid"])
    losses = []
    losses_valid = []
    snapshots = {}
    start_time = time.time()
    avg_duration = None
    num_iters = 301
    for i in range(num_iters):
        loss = svi.step(data, subsample_size=25)
        losses.append(loss)
        loss_valid = svi.evaluate_loss(data_valid, subsample_size=50)
        losses_valid.append(loss_valid)

        writer.add_scalar('loss', loss, i)
        writer.add_scalar('loss_valid', loss_valid, i)

        for p in interesting_params:
            if p not in snapshots.keys():
                snapshots[p] = []
            snapshots[p].append(pyro.param(p).cpu().detach().numpy().copy())

        elapsed = time.time() - start_time
        if avg_duration is None:
            avg_duration = elapsed
        else:
            avg_duration = avg_duration*0.9 + elapsed*0.1
        start_time = time.time()
        if (i % 10 == 0):
            print "Loss %f (%f), Per iter: %f, To go: %f" % (loss, loss_valid, elapsed, (num_iters - i)*elapsed)
        if (i % 10 == 0):
            print select_interesting()
    print "Done"