from __future__ import print_function
from collections import namedtuple
from copy import deepcopy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import yaml

import pydrake  # MUST BE BEFORE TORCH OR PYRO
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager
from tensorboardX import SummaryWriter
from torchviz import make_dot

import scene_generation.data.dataset_utils as dataset_utils
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_model import *

ScoreInfo = namedtuple('ScoreInfo', 'joint_score latents_score f total_score')

def print_param_store(grads=False):
    for param_name in pyro.get_param_store().keys():
        val = pyro.param(param_name)#.tolist()
        grad = pyro.param(param_name).grad
        #if isinstance(val, float):
        #    val = [val]
        if grads:
            print(param_name, ": ", val.data, ", unconstrained grad: ", pyro.get_param_store()._params[param_name].grad)
        else:
            print(param_name, ": ", val.data)

def rotate_yaml_env(env, r):
    rotation_origin = np.array([0.5, 0.5])
    rotmat = np.array([[np.cos(r), -np.sin(r)],
                       [np.sin(r), np.cos(r)]])
    for obj_k in range(env["n_objects"]):
        obj_yaml = env["obj_%04d" % obj_k]
        init_pose = np.array(obj_yaml["pose"])
        init_pose[2] += r
        init_pose[2] = np.mod(init_pose[2], np.pi*2.)
        init_pose[:2] = rotmat.dot(init_pose[:2] - rotation_origin) + rotation_origin
        obj_yaml["pose"] = init_pose.tolist()

def score_sample_sync(env, guide_gvs, outer_iterations=2, num_attempts=3):
    observed_tree, joint_score = guess_parse_tree_from_yaml(
        env, guide_gvs=guide_gvs, outer_iterations=outer_iterations, num_attempts=num_attempts, verbose=False)
    # Joint score is P(T, V_obs)
    # Latent score is P(T | V_obs)
    latents_score, _ = observed_tree.get_total_log_prob(include_observed=False)
    f = joint_score - latents_score
    baseline = 0.
    total_score = -(latents_score * (f.detach() - baseline) + f)
    score_info = ScoreInfo(joint_score=joint_score,
                           latents_score=latents_score,
                           f=f,
                           total_score=total_score)
    print("Obs tree with joint score %f, latents score %f, f %f, total score %f" % (joint_score, latents_score, f, total_score))
    active_param_names = set().union(
        *[node.get_param_names() for node in observed_tree.nodes],
        *[[n + "_est" for n in node.get_global_variable_names()] for node in observed_tree.nodes])
    return score_info, active_param_names

def score_sample_async(thread_id, env, guide_gvs, eval_backward, output_queue, synchro_prims, outer_iterations=2, num_attempts=3):
    post_parsing_barrier, grads_reset_event, done_event = synchro_prims
    observed_tree, joint_score = guess_parse_tree_from_yaml(
        env, guide_gvs=guide_gvs, outer_iterations=outer_iterations, num_attempts=num_attempts, verbose=False)
    # Joint score is P(T, V_obs)
    # Latent score is P(T | V_obs)
    post_parsing_barrier.wait()
    if thread_id == 0:
        # Thread 0 resets the gradients to 0
        for name in pyro.get_param_store().keys():
            pyro.get_param_store()._params[name].grad.data.zero_()
        grads_reset_event.set()
    else:
        grads_reset_event.wait()

    latents_score, _ = observed_tree.get_total_log_prob(include_observed=False)
    f = joint_score - latents_score
    baseline = 0.
    total_score = -(latents_score * (f.detach() - baseline) + f)
    print("Obs tree with joint score %f, latents score %f, f %f, total score %f" % (joint_score, latents_score, f, total_score))

    if eval_backward:
        total_score.backward(retain_graph=True)
    score_info = ScoreInfo(joint_score=joint_score.detach(),
                           latents_score=latents_score.detach(),
                           f=f.detach(),
                           total_score=total_score.detach())
    output_queue.put((total_score.detach(), score_info))
    done_event.wait()
    
def score_subset_of_dataset_sync(dataset, n, guide_gvs):
    # Computes an SVI ELBO estimate of n samples from the dataset,
    # with a Delta-distribution mean-field variational distribution over
    # the global latent variables, and an implicit sampled distribution
    # over the local latent variables.
    losses = []
    all_score_infos = []
    active_param_names = set()
    baseline = 0
    for p_k in range(n):
        # Domain randomization
        env = random.choice(dataset)
        #rotate_yaml_env(env, np.random.uniform(0, 2*np.pi))
        score_info, active_param_names_local = score_sample_sync(env, guide_gvs)
        losses.append(score_info.total_score)
        all_score_infos.append(score_info)
        active_param_names = set().union(
            active_param_names,
            active_param_names_local)
    loss = torch.stack(losses).mean()
    return loss, all_score_infos, active_param_names

def calc_score_and_backprob_async(dataset, n, guide_gvs, optimizer=None):
    # Select out minibatch
    envs = []
    for p_k in range(n):
        # Domain randomization
        env = random.choice(dataset)
        #rotate_yaml_env(env, np.random.uniform(0, 2*np.pi))
        envs.append(env)

    do_backprop = optimizer is not None
    all_params_to_optimize = set(pyro.get_param_store()._params[name] for name in pyro.get_param_store().keys())
    
    if True:   # ASYNC
        sync_manager = SyncManager()
        sync_manager.start()
        post_parsing_barrier = sync_manager.Barrier(n)
        grads_reset_event = mp.Event()
        processes = []
        losses = []
        all_score_infos = []
        output_queue = mp.SimpleQueue()
        done_event = mp.Event()
        synchro_prims = [post_parsing_barrier, grads_reset_event, done_event]
        for i, env in enumerate(envs):
            p = mp.Process(
                target=score_sample_async, args=(
                    i, env, guide_gvs, do_backprop, output_queue, synchro_prims))
            p.start()
            processes.append(p)
        for k in range(n):
            loss, score_info = output_queue.get()
            losses.append(loss)
            all_score_infos.append(score_info)
        done_event.set()
        for p in processes:
            p.join()
        loss = torch.stack(losses).mean()
        print("Loss async: ", loss)
        if do_backprop:
            # Apply averaging to gradients
            for p in all_params_to_optimize:
                p.grad.data /= float(n)
            optimizer(all_params_to_optimize)
        return loss, all_score_infos
    else:    # EQUIVALENT SINGLE-THREAD
        loss, all_score_infos, active_param_names = score_subset_of_dataset_sync(dataset, n, guide_gvs)
        #for param in all_params_to_optimize:
        ##    param.grad *= -1.0
        print("Loss sync: ", loss)
        if do_backprop:
            for p in all_params_to_optimize:
                p.grad.data.zero_()
            print("PRE EVAL: ")
            for name in pyro.get_param_store().keys():
                print(name, " grad: ", pyro.get_param_store()._params[name].grad)
            loss.backward(retain_graph=True)
            print("POST EVAL: ")
            for name in pyro.get_param_store().keys():
                print(name, " grad: ", pyro.get_param_store()._params[name].grad)
            optimizer(all_params_to_optimize)
        return loss, all_score_infos

if __name__ == "__main__":

    seed = int(time.time()) % (2**32-1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    pyro.clear_param_store()

    hyper_parse_tree = generate_hyperexpanded_parse_tree()
    guide_gvs = hyper_parse_tree.get_global_variable_store()

    train_dataset = dataset_utils.ScenesDataset("../data/table_setting/table_setting_environments_simple_train")
    test_dataset = dataset_utils.ScenesDataset("../data/table_setting/table_setting_environments_simple_test")
    print("%d training examples" % len(train_dataset))
    print("%d test examples" % len(test_dataset))

    
    #plt.figure().set_size_inches(15, 10)
    ##parse_trees = [guess_parse_tree_from_yaml(test_dataset[k], guide_gvs=hyper_parse_tree.get_global_variable_store())[0] for k in range(4)]
    ##parse_trees = guess_parse_trees_batch_async(test_dataset[:4], guide_gvs=hyper_parse_tree.get_global_variable_store())
    #print("Parsed %d trees" % len(parse_trees))
    #print(parse_trees[0].nodes)
    #for k in range(4):
    #    plt.subplot(2, 2, k+1)
    #    DrawYamlEnvironmentPlanar(test_dataset[k], base_environment_type="table_setting", ax=plt.gca())
    #    draw_parse_tree(parse_trees[k], label_name=True, label_score=True, alpha=0.7)
    #plt.show()
    #sys.exit(0)

    use_writer = True

    log_dir = "/home/gizatt/projects/scene_generation/models/runs/psg/table_setting/simple/trash_" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%m-%s")

    if use_writer:
        writer = SummaryWriter(log_dir)
        def write_np_array(writer, name, x, i):
            for yi, y in enumerate(x):
                writer.add_scalar(name + "/%d" % yi, y, i)

    param_val_history = []
    score_history = []
    score_test_history = []

    # Initialize the guide GVS as mean field
    # Note -- if any terminal nodes have global variables associated with
    # them, they won't be in the guide.
    for var_name in guide_gvs.keys():
        guide_gvs[var_name][0] = pyro.param(var_name + "_est",
                                            guide_gvs[var_name][0],
                                            constraint=guide_gvs[var_name][1].support)
    # do gradient steps
    print_param_store()
    best_loss_yet = np.infty

    # setup the optimizer
    adam_params = {"lr": 0.001, "betas": (0.8, 0.95)}
    all_params_to_optimize = set(pyro.get_param_store()._params[name] for name in pyro.get_param_store().keys())
    # Ensure everything in pyro param store has zero grads
    for p in all_params_to_optimize:
        assert(p.requires_grad == True)
        p.grad = torch.zeros(p.shape)
        p.share_memory_()
        p.grad.share_memory_()

    def per_param_callable(module_name, param_name):
        if "var" in param_name or "weights" in param_name:
            return {"lr": 0.01, "betas": (0.8, 0.95)}

        else:
            return {"lr": 0.001, "betas": (0.8, 0.95)}
    optimizer = Adam(per_param_callable)
    baseline = 0.
    baseline = 0.

    def write_score_info(i, prefix, writer, loss, all_score_infos):
        f = torch.stack([score_info.f for score_info in all_score_infos]).mean()
        latents_score = torch.stack([score_info.latents_score for score_info in all_score_infos]).mean()
        joint_score = torch.stack([score_info.joint_score for score_info in all_score_infos]).mean()
        writer.add_scalar(prefix + "loss", loss.item(), i)
        writer.add_scalar(prefix + "f", f, i)
        writer.add_scalar(prefix + "latents_score", latents_score, i)
        writer.add_scalar(prefix + "joint_score", joint_score, i)

    snapshots = {}
    total_step = 0
    for step in range(500):
        # Synchronize gvs and param store. In the case of constrained parameters,
        # the constrained value returned by pyro.param() is distinct from the
        # unconstrianed value we optimize, so we need to regenerate the constrained value.
        for var_name in guide_gvs.keys():
            guide_gvs[var_name][0] = pyro.param(var_name + "_est")

        loss, all_score_infos = calc_score_and_backprob_async(train_dataset, n=1, guide_gvs=guide_gvs, optimizer=optimizer)
        #loss = svi.step(observed_tree)
        score_history.append(loss)

        if (total_step % 10 == 0):
            # Evaluate on a few test data points
            loss_test, all_score_infos_test = calc_score_and_backprob_async(test_dataset, n=10, guide_gvs=guide_gvs)
            score_test_history.append(loss_test)
            print("Loss_test: ", loss_test)

            if loss_test < best_loss_yet:
                best_loss_yet = loss
                pyro.get_param_store().save("best_on_test_save.pyro")

            if use_writer:
                write_score_info(total_step, "test_", writer, loss_test, all_score_infos_test)

                # Also generate a few example environments
                # Generate a ground truth test environment
                plt.figure().set_size_inches(20, 20)
                for k in range(4):
                    plt.subplot(2, 2, k+1)
                    parse_tree = generate_unconditioned_parse_tree(initial_gvs=guide_gvs)
                    yaml_env = convert_tree_to_yaml_env(parse_tree)
                    try:
                        DrawYamlEnvironmentPlanar(yaml_env, base_environment_type="table_setting", ax=plt.gca())
                    except:
                        print("Unhandled exception in drawing yaml env")
                    draw_parse_tree(parse_tree, label_name=True, label_score=True, alpha=0.75)
                writer.add_figure("generated_envs", plt.gcf(), total_step, close=True)

                # Also parse some test environments
                test_envs = [random.choice(test_dataset) for k in range(4)]
                test_parses = guess_parse_trees_batch_async(test_envs, guide_gvs=guide_gvs)
                plt.figure().set_size_inches(20, 20)
                for k in range(4):
                    plt.subplot(2, 2, k+1)
                    try:
                        DrawYamlEnvironmentPlanar(test_envs[k], base_environment_type="table_setting", ax=plt.gca())
                    except:
                        print("Unhandled exception in drawing yaml env")
                    draw_parse_tree(test_parses[k], label_name=True, label_score=True, alpha=0.75)
                writer.add_figure("parsed_test_envs", plt.gcf(), total_step, close=True)

        all_param_state = {name: pyro.param(name).detach().cpu().numpy().copy() for name in pyro.get_param_store().keys()}
        if use_writer:
            write_score_info(total_step, "train_", writer, loss, all_score_infos)
            for param_name in all_param_state.keys():
                write_np_array(writer, param_name, all_param_state[param_name], total_step)
        param_val_history.append(all_param_state)
        #print("active param names: ", active_param_names)
        print("Place setting plate mean est: ", pyro.param("place_setting_plate_mean_est"))
        print("Place setting plate var est: ", pyro.param("place_setting_plate_var_est"))
        total_step += 1
    print("Final loss: ", loss)
    print_param_store()