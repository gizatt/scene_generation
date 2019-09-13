from __future__ import print_function
from collections import namedtuple
from copy import deepcopy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
import time
import yaml

import pydrake  # MUST BE BEFORE TORCH OR PYRO
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from multiprocessing.managers import SyncManager
from tensorboardX import SummaryWriter

import scene_generation.data.dataset_utils as dataset_utils
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_nodes_dish_bin import *
from scene_generation.models.probabilistic_scene_grammar_model import *
from scene_generation.models.probabilistic_scene_grammar_fitting import *

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":

    seed = int(time.time()) % (2**32-1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # CONFIGURATION STUFF
    root_node_type = DishBin
    output_dir = "../data/table_setting/icra_runs/dish_bin/test/"
    os.system("mkdir -p %s" % output_dir)


    root_node = root_node_type()
    hyper_parse_tree = generate_hyperexpanded_parse_tree(root_node)
    guide_gvs = hyper_parse_tree.get_global_variable_store()

    train_dataset = dataset_utils.ScenesDataset("../data/dish_bins/dish_bin_environments_greg.yaml")
    test_dataset = dataset_utils.ScenesDataset("../data/dish_bins/dish_bin_environments_greg.yaml")
    print("%d training examples" % len(train_dataset))
    print("%d test examples" % len(test_dataset))

    
    plt.figure().set_size_inches(15, 10)
    #parse_trees = [guess_parse_tree_from_yaml(test_dataset[k], root_node_type=root_node_type, guide_gvs=hyper_parse_tree.get_global_variable_store(), outer_iterations=2, num_attempts=5, verbose=True)[0] for k in range(4)]
    parse_trees = guess_parse_trees_batch_async(test_dataset[:4], root_node_type=root_node_type, guide_gvs=hyper_parse_tree.get_global_variable_store(), outer_iterations=4, num_attempts=5)
    print("Parsed %d trees" % len(parse_trees))
    #print("*****************\n0: ", parse_trees[0].nodes)
    #print("*****************\n1: ", parse_trees[1].nodes)
    ##print("*****************\n2: ", parse_trees[2].nodes)
    ##print("*****************\n3: ", parse_trees[3].nodes)
    for k in range(4):
        yaml_env = convert_tree_to_yaml_env(parse_trees[k])
        DrawYamlEnvironment(yaml_env, base_environment_type="dish_bin", alpha=0.5)
        draw_parse_tree_meshcat(parse_trees[k], color_by_score=True)
        print(parse_trees[k].get_total_log_prob())
        input("Press enter to continue...")
        
    #plt.show()
    sys.exit(0)

    use_writer = False

    log_dir = output_dir + "fixed_elbo_" + datetime.datetime.now().strftime(
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

    def write_score_info(i, prefix, writer, loss, all_score_infos):
        f = torch.stack([score_info.f for score_info in all_score_infos]).mean()
        latents_score = torch.stack([score_info.latents_score for score_info in all_score_infos]).mean()
        joint_score = torch.stack([score_info.joint_score for score_info in all_score_infos]).mean()
        observed_score = torch.stack([score_info.observed_score for score_info in all_score_infos]).mean()
        writer.add_scalar(prefix + "loss", loss.item(), i)
        writer.add_scalar(prefix + "f", f, i)
        writer.add_scalar(prefix + "latents_score", latents_score, i)
        writer.add_scalar(prefix + "joint_score", joint_score, i)
        writer.add_scalar(prefix + "observed_score", observed_score, i)

    snapshots = {}
    total_step = 0
    pyro.get_param_store().save(output_dir + "param_store_initial.pyro")
    f_history = []
    for step in range(500):
        # Synchronize gvs and param store. In the case of constrained parameters,
        # the constrained value returned by pyro.param() is distinct from the
        # unconstrianed value we optimize, so we need to regenerate the constrained value.
        for var_name in guide_gvs.keys():
            guide_gvs[var_name][0] = pyro.param(var_name + "_est")

        if len(f_history) > 0:
            baseline = torch.stack(f_history).mean()
        else:
            baseline=0.
        loss, all_score_infos = calc_score_and_backprob_async(train_dataset, n=10, root_node_type=root_node_type, guide_gvs=guide_gvs, optimizer=optimizer, baseline=baseline)
        #loss = svi.step(observed_tree)
        score_history.append(loss)
        f_history.append(torch.stack([score_info.f for score_info in all_score_infos]).mean())
        if len(f_history) > 30:
            f_history.pop(0)

        if (total_step % 10 == 0):
            # Evaluate on a few test data points
            loss_test, all_score_infos_test = calc_score_and_backprob_async(test_dataset, n=10, root_node_type=root_node_type, guide_gvs=guide_gvs)
            score_test_history.append(loss_test)
            print("Loss_test: ", loss_test)

            if loss_test < best_loss_yet:
                best_loss_yet = loss
                pyro.get_param_store().save(output_dir + "param_store_best_on_test.pyro")

            if use_writer:
                write_score_info(total_step, "test_", writer, loss_test, all_score_infos_test)

                # Also generate a few example environments
                # Generate a ground truth test environment
                plt.figure().set_size_inches(20, 20)
                for k in range(4):
                    plt.subplot(2, 2, k+1)
                    parse_tree = generate_unconditioned_parse_tree(root_node, initial_gvs=guide_gvs)
                    yaml_env = convert_tree_to_yaml_env(parse_tree)
                    try:
                        DrawYamlEnvironmentPlanarForTableSettingPretty(yaml_env, ax=plt.gca())
                    except:
                        print("Unhandled exception in drawing yaml env")
                    draw_parse_tree(parse_tree, label_name=True, label_score=True, alpha=0.75)
                writer.add_figure("generated_envs", plt.gcf(), total_step, close=True)

                # Also parse some test environments
                test_envs = [random.choice(test_dataset) for k in range(4)]
                test_parses = guess_parse_trees_batch_async(test_envs, root_node_type=root_node_type, guide_gvs=guide_gvs.detach())
                plt.figure().set_size_inches(20, 20)
                for k in range(4):
                    plt.subplot(2, 2, k+1)
                    try:
                        DrawYamlEnvironmentPlanarForTableSettingPretty(test_envs[k], ax=plt.gca())
                    except:
                        print("Unhandled exception in drawing yaml env")
                    draw_parse_tree(test_parses[k], label_name=True, label_score=True, alpha=0.75)
                writer.add_figure("parsed_test_envs", plt.gcf(), total_step, close=True)

        all_param_state = {name: pyro.param(name).detach().cpu().numpy().copy() for name in pyro.get_param_store().keys()}
        if use_writer:
            write_score_info(total_step, "train_", writer, loss, all_score_infos)
            writer.add_scalar("baseline", baseline, total_step)
            for param_name in all_param_state.keys():
                write_np_array(writer, param_name, all_param_state[param_name], total_step)
        param_val_history.append(all_param_state)
        #print("active param names: ", active_param_names)
        total_step += 1
    print("Final loss: ", loss)
    pyro.get_param_store().save(output_dir + "param_store_final.pyro")
    print_param_store()