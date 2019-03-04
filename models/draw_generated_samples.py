import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import yaml
import sys

import pydrake
import pyro

import scene_generation.data.dataset_utils as dataset_utils

#base_path = "../models/generated_planar_bin_static_scenes_stacks_tall"
base_path = "../notebooks/generated_planar_bin_static_scenes_with_context_stacks_tall"
raw_path = base_path + "_raw.yaml"
static_path = base_path + "_static.yaml"
nonpen_path = base_path + "_nonpen.yaml"

all_dataset_paths = [raw_path, nonpen_path, static_path]
all_dataset_names = ["raw", "nonpen", "static"]
n_datasets = len(all_dataset_names)
all_datasets = [dataset_utils.ScenesDataset(p) for p in all_dataset_paths]

plt.figure().set_size_inches(20, 3)
for data_i in range(100):
    try:
        env_name = None
        for i, name in enumerate(all_dataset_names):
            plt.subplot(1, n_datasets, i+1)
            plt.gca().clear()
            if env_name is None:
                env = all_datasets[i][data_i]
                env_name = all_datasets[i].yaml_environments_names[data_i]
            else:
                env = all_datasets[i][all_datasets[i].get_environment_index_by_name(env_name)]
            dataset_utils.DrawYamlEnvironmentPlanar(
                    env, "planar_bin",
                    xlim=[-1, 1], ylim=[0.0, 1],
                    ax=plt.gca())
        plt.pause(1.)
    except Exception as e:
        print e