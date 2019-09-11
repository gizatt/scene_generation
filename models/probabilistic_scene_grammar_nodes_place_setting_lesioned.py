from __future__ import print_function
from copy import deepcopy
from functools import partial
import time
import random
import sys
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pydrake
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro import poutine

import scene_generation.data.dataset_utils as dataset_utils
from scene_generation.data.dataset_utils import (
    DrawYamlEnvironmentPlanar, ProjectEnvironmentToFeasibility)
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_nodes_place_setting import *
from scene_generation.models.probabilistic_scene_grammar_model import *

torch.set_default_tensor_type(torch.DoubleTensor)

class TableWithoutPlaceSettings(CovaryingSetNode, RootNode):

    class ObjectProductionRule(ProductionRule):
        def __init__(self, name, object_name, object_type, mean_prior_params, var_prior_params):
            self.object_name = object_name
            self.object_type = object_type
            self.mean_prior_params = mean_prior_params
            self.var_prior_params = var_prior_params
            self.global_variable_names = ["table_%s_mean" % self.object_name,
                                         "table_%s_var" % self.object_name]
            ProductionRule.__init__(self,
                name=name,
                product_types=[object_type])

        def _recover_rel_pose_from_abs_pose(self, parent, abs_pose):
            return chain_pose_transforms(invert_pose(parent.pose), abs_pose)

        def sample_global_variables(self, global_variable_store):
            # Handles class-general setup
            mean_prior_dist = dist.Normal(loc=self.mean_prior_params[0],
                                          scale=self.mean_prior_params[1]).to_event(1)
            var_prior_dist = dist.InverseGamma(concentration=self.var_prior_params[0],
                                               rate=self.var_prior_params[1]).to_event(1)
            mean = global_variable_store.sample_global_variable("table_%s_mean" % self.object_name,
                                              mean_prior_dist).double()
            var = global_variable_store.sample_global_variable("table_%s_var" % self.object_name,
                                             var_prior_dist).double()
            self.offset_dist = dist.Normal(loc=mean, scale=var).to_event(1)
            
        def sample_products(self, parent, obs_products=None):
            # Observation should be absolute position of the product
            if obs_products is not None:
                assert(len(obs_products) == 1 and isinstance(obs_products[0], self.object_type))
                obs_rel_pose = self._recover_rel_pose_from_abs_pose(parent, obs_products[0].pose)
                rel_pose = pyro.sample("%s_pose" % (self.name),
                                       self.offset_dist, obs=obs_rel_pose)
                return obs_products
            else:
                rel_pose = pyro.sample("%s_pose" % (self.name), self.offset_dist).detach()
                abs_pose = chain_pose_transforms(parent.pose, rel_pose)
                return [self.object_type(name="%s_%s" % (self.name, self.object_name), pose=abs_pose)]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], self.object_type):
                return torch.tensor(-np.inf)
            # Get relative offset of the PlaceSetting
            rel_pose = self._recover_rel_pose_from_abs_pose(parent, products[0].pose.detach())
            return self.offset_dist.log_prob(rel_pose.double()).double()


    class TableProductionRule(ProductionRule):
        def __init__(self, name):
            ProductionRule.__init__(self,
                name=name,
                product_types=[TableWithoutPlaceSettings])

        def sample_products(self, parent, obs_products=None):
            # Observation should be absolute position of the product
            if obs_products is not None:
                assert(len(obs_products) == 1 and isinstance(obs_products[0], TableWithoutPlaceSettings))
                return obs_products
            else:
                return [TableWithoutPlaceSettings(name="%s_%s" % (self.name, "table_lesioned"))]

        def score_products(self, parent, products):
            if len(products) != 1 or not isinstance(products[0], TableWithoutPlaceSettings):
                return torch.tensor(-np.inf)
            return torch.tensor(0.).double()

    

    def __init__(self, name="table_lesioned"):
        self.pose = torch.tensor([0.5, 0.5, 0.]).double()

        # Represent each object's relative position to the
        # the table origin with a diagonal Normal distribution.
        self.object_types_by_name = {
            "plate": Plate,
            "cup": Cup,
            "fork": Fork,
            "knife": Knife,
            "spoon": Spoon
        }
        # Key: Class name (from above)
        # Value: Nominal (Mean, Variance) used to set up prior distributions
        param_guesses_by_name = {
            "plate": ([0., 0., 0.], [0.1, 0.1, 3.]),
            "cup": ([0., 0., 0.], [0.1, 0.1, 3.]),
            "fork": ([0., 0., 0.], [0.1, 0.1, 0.1]),
            "knife": ([0., 0., 0.], [0.1, 0.1, 0.1]),
            "spoon": ([0., 0., 0.], [0.1, 0.1, 0.1]),
        }
        production_rules = []
        name_to_ind = {}
        for k, object_name in enumerate(self.object_types_by_name.keys()):
            mean_init, var_init = param_guesses_by_name[object_name]
            # Reasonably broad prior on the mean
            mean_prior_variance = (torch.ones(3)*0.05).double()
            # Use an inverse gamma prior for variance. It has MEAN (rather than mode)
            # beta / (alpha - 1) = var
            # (beta / var) + 1 = alpha
            # Picking bigger beta/var ratio leads to tighter peak around the guessed variance.
            var_prior_width_fact = 1
            assert(var_prior_width_fact > 0.)
            beta = var_prior_width_fact*torch.tensor(var_init).double()
            alpha = var_prior_width_fact*torch.ones(len(var_init)).double() + 1
            production_rules.append(
                self.ObjectProductionRule(
                    name="%s_prod_%03d" % (name, k),
                    object_name=object_name,
                    object_type=self.object_types_by_name[object_name],
                    mean_prior_params=(torch.tensor(mean_init, dtype=torch.double), mean_prior_variance),
                    var_prior_params=(alpha, beta)))
            # Build name mapping for convenience of building the hint dictionary
            name_to_ind[object_name] = k

        # And finally add one production rule for producing another Table to spawn a new object
        production_rules.append(
            self.TableProductionRule(
                name="%s_prod_table" % (name)))
        name_to_ind["table"] = k + 1

        # Initialize the production rules here. (They're parameters, so they don't have a prior.)
        production_weights_hints = {
            # tuple(): 1., # Nothing
        }

        init_weights = CovaryingSetNode.build_init_weights(
            num_production_rules=len(production_rules),
            #production_weights_hints=production_weights_hints,
            remaining_weight=1.0)
        init_weights = pyro.param("table_production_weights", init_weights, constraint=constraints.simplex)
        self.param_names = ["table_production_weights"]
        CovaryingSetNode.__init__(self, name=name, production_rules=production_rules, init_weights=init_weights)


if __name__ == "__main__":
    #seed = 52
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    pyro.enable_validation(True)

    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    
    if False:
        # Draw + plot a few generated environments and their trees
        plt.figure().set_size_inches(20, 20)
        N = 2
        for k in range(N**2):
            start = time.time()
            pyro.clear_param_store()
            root_node = TableWithoutPlaceSettings()
            trace = poutine.trace(generate_unconditioned_parse_tree).get_trace(root_node=root_node)
            parse_tree = trace.nodes["_RETURN"]["value"]
            end = time.time()

            print(bcolors.OKGREEN, "Generated data in %f seconds." % (end - start), bcolors.ENDC)
            #print("Full trace values:" )
            #for node_name in trace.nodes.keys():
            #    if node_name in ["_INPUT", "_RETURN"]:
            #        continue
            #    print(node_name, ": ", trace.nodes[node_name]["value"].detach().numpy())

            # Recover and print the parse tree
            plt.subplot(N, N, k+1)
            plt.gca().clear()
            plt.gca().set_xlim(0.1, 0.9)
            plt.gca().set_ylim(0.1, 0.9)
            score, score_by_node = parse_tree.get_total_log_prob()
            #print("Score by node: ", score_by_node)

            # Resample it to feasibility
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
            assert(abs(score - trace.log_prob_sum()) < 0.001)
        plt.show()
    
    else:
        # Load in a test dataset and try parsing some envs
        pyro.enable_validation(True)
        pyro.clear_param_store()

        root_node = TableWithoutPlaceSettings()
        # Max number of the same object that can appear is enough here.
        hyper_parse_tree = generate_hyperexpanded_parse_tree(root_node, max_iters=8)
        guide_gvs = hyper_parse_tree.get_global_variable_store()

        train_dataset = dataset_utils.ScenesDataset("/home/gizatt/projects/scene_generation/data/table_setting/table_setting_environments_generated_nominal_train")
        test_dataset = dataset_utils.ScenesDataset("/home/gizatt/projects/scene_generation/data/table_setting/table_setting_environments_generated_nominal_test")
        print("%d training examples" % len(train_dataset))
        print("%d test examples" % len(test_dataset))

        # Parse examples from the test set using the learned params, and draw the parses
        plt.figure().set_size_inches(10, 10)

        #parse_trees = guess_parse_trees_batch_async(test_dataset[:4], guide_gvs=guide_gvs.detach())
        for k in range(4):
            plt.subplot(2, 2, k+1)
            parse_tree, _ = guess_parse_tree_from_yaml(test_dataset[k], guide_gvs=guide_gvs, max_iters_for_hyper_parse_tree=8, outer_iterations=2, num_attempts=2, verbose=False, root_node_type=TableWithoutPlaceSettings)
            DrawYamlEnvironmentPlanarForTableSettingPretty(test_dataset[k], ax=plt.gca())
            draw_parse_tree(parse_tree, label_name=True, label_score=True, alpha=0.7)
        plt.tight_layout()

        # Calculate ELBO (which in the single-parse-tree case is just the probability of the observed nodes):
        #for parse_tree in parse_trees:
        #    joint_score = parse_tree.get_total_log_prob()[0]
        #    latents_score = parse_tree.get_total_log_prob(include_observed=False)[0]
        #    print("ELBO: ", joint_score - latents_score)

        plt.show()
#        for var_name in guide_gvs.keys():
#            guide_gvs[var_name][0] = pyro.param(var_name + "_est",
#                                                guide_gvs[var_name][0],
#                                                constraint=guide_gvs[var_name][1].support)
