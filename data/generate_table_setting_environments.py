from __future__ import print_function
from copy import deepcopy
from functools import partial
import multiprocessing
import time
import random
import sys
import yaml
import weakref

import matplotlib.pyplot as plt
import numpy as np

import pydrake
import torch
import pyro

from scene_generation.data.dataset_utils import (
    DrawYamlEnvironmentPlanar, DrawYamlEnvironment, DrawYamlEnvironmentPlanarForTableSettingPretty, ProjectEnvironmentToFeasibility,
    BuildMbpAndSgFromYamlEnvironment)
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_nodes_place_setting import *
from scene_generation.models.probabilistic_scene_grammar_model import *

if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)

    torch.set_default_tensor_type(torch.DoubleTensor)
    pyro.enable_validation(True)

    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    
    root_node_type = Table

    nominal_values = {
        'place_setting_plate_mean': torch.tensor([0.0, 0.14, 0.0]),
        'place_setting_plate_var': torch.tensor([0.02, 0.01, 3.]),
        'place_setting_cup_mean': torch.tensor([0.0, 0.14 + 0.12, 0.0]),
        'place_setting_cup_var': torch.tensor([0.03, 0.01, 3.]),
        'place_setting_left_fork_mean': torch.tensor([-0.15, 0.12, 0.]),
        'place_setting_left_fork_var': torch.tensor([0.02, 0.02, 0.02]),
        'place_setting_left_knife_mean': torch.tensor([-0.15, 0.12, 0.]),
        'place_setting_left_knife_var': torch.tensor([0.02, 0.02, 0.02]),
        'place_setting_left_spoon_mean': torch.tensor([-0.15, 0.12, 0.]),
        'place_setting_left_spoon_var': torch.tensor([0.02, 0.02, 0.02]),
        'place_setting_right_fork_mean': torch.tensor([0.15, 0.12, 0.]),
        'place_setting_right_fork_var': torch.tensor([0.02, 0.02, 0.02]),
        'place_setting_right_knife_mean': torch.tensor([0.15, 0.12, 0.]),
        'place_setting_right_knife_var': torch.tensor([0.02, 0.02, 0.02]),
        'place_setting_right_spoon_mean': torch.tensor([0.15, 0.12, 0.]),
        'place_setting_right_spoon_var': torch.tensor([0.02, 0.02, 0.02]),
    }
    hyper_parse_tree = generate_hyperexpanded_parse_tree(root_node=root_node_type())
    guide_gvs = hyper_parse_tree.get_global_variable_store()
    for var_name in guide_gvs.keys():
        guide_gvs[var_name][0] = pyro.param(var_name + "_est",
                                            nominal_values[var_name],
                                            constraint=guide_gvs[var_name][1].support)

    # Draw + plot a few generated environments and their trees
    #plt.figure().set_size_inches(20, 20)
    #N = 3
    #for k in range(N**2):
    #    start = time.time()
    #    pyro.clear_param_store()
    #    parse_tree = generate_unconditioned_parse_tree(initial_gvs=guide_gvs, root_node=Table())
    #    end = time.time()
##
    #    print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
    #    
    #    # Recover and print the parse tree
    #    plt.subplot(N, N, k+1)
    #    plt.gca().clear()
    #    plt.gca().set_xlim(0.1, 0.9)
    #    plt.gca().set_ylim(0.1, 0.9)
    #    score, score_by_node = parse_tree.get_total_log_prob()
    #    
    #    # Resample it to feasibility
    #    #parse_tree = resample_parse_tree_to_feasibility(parse_tree, base_environment_type="table_setting")
    #    yaml_env = convert_tree_to_yaml_env(parse_tree)
    #    yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type="table_setting", make_nonpenetrating=True, make_static=False)[-1]
    #    DrawYamlEnvironmentPlanarForTableSettingPretty(yaml_env, ax=plt.gca())
    #    node_class_to_color_dict = {"Table":[0., 1., 0.], "PlaceSetting":[0., 0., 1.]}
    #    draw_parse_tree(parse_tree, label_name=False, label_score=False, alpha=0.25,
    #                    node_class_to_color_dict=node_class_to_color_dict)
    #    print("Our score: %f" % score)
    #    #print("Trace score: %f" % trace.log_prob_sum())
    #    plt.gca().set_xlim(0.1, 0.9)
    #    plt.gca().set_ylim(0.1, 0.9)
    #plt.show()
    #sys.exit(0)

    # Save out the param store to "nominal"
    pyro.get_param_store().save("place_setting_outlier_param_store.pyro")

    plt.figure().set_size_inches(15, 10)
    for k in range(200):
        start = time.time()
        pyro.clear_param_store()
        parse_tree = generate_unconditioned_parse_tree(initial_gvs=guide_gvs, root_node=Table())
        end = time.time()

        print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
        
        # Recover and print the parse tree
        plt.gca().clear()
        plt.gca().set_xlim(0.1, 0.9)
        plt.gca().set_ylim(0.1, 0.9)
        score, score_by_node = parse_tree.get_total_log_prob()
        
        # Make it feasible
        #parse_tree = resample_parse_tree_to_feasibility(parse_tree, base_environment_type="table_setting")
        
        yaml_env = convert_tree_to_yaml_env(parse_tree)
        yaml_env = ProjectEnvironmentToFeasibility(yaml_env, base_environment_type="table_setting", make_nonpenetrating=True, make_static=False)[-1]
        DrawYamlEnvironmentPlanarForTableSettingPretty(yaml_env, ax=plt.gca())
        node_class_to_color_dict = {"Table":[0., 1., 0.], "PlaceSetting":[0., 0., 1.]}
        draw_parse_tree(parse_tree, label_name=False, label_score=False, alpha=0.25,
                        node_class_to_color_dict=node_class_to_color_dict)
        print("Our score: %f" % score)
        #print("Trace score: %f" % trace.log_prob_sum())
        plt.gca().set_xlim(0.1, 0.9)
        plt.gca().set_ylim(0.1, 0.9)

        plt.pause(0.1)

        with open("table_setting_environments_generated_outlier.yaml", "a") as file:
            yaml.dump({"env_%d" % int(round(time.time() * 1000)): yaml_env}, file, Dumper=noalias_dumper)