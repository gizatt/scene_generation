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
from scene_generation.models.planar_multi_object_multi_class_2_with_context import (
    MultiObjectMultiClassModelWithContext)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path",
                        help="Dataset used for model.")
    parser.add_argument("param_path",
                        help="Parameters to load.")
    args = parser.parse_args()

    scenes_dataset = dataset_utils.ScenesDatasetVectorized(args.data_path)
    # Load model
    # Load in params *before* making model, so that the model modules
    # update properly
    pyro.get_param_store().load(args.param_path)
    model = MultiObjectMultiClassModelWithContext(scenes_dataset)
    # Call once to fill in param store
    model.model()

    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True

    while (1):
        generated_data, generated_encodings, generated_contexts = model.model()
        scene_yaml = scenes_dataset.convert_vectorized_environment_to_yaml(
            generated_data)[0]
        try:
            scene_nonpen, scene_static = dataset_utils.ProjectEnvironmentToFeasibility(
                scene_yaml, "planar_bin")

            print "Scene original"
            dataset_utils.DrawYamlEnvironment(
                scene_yaml, "planar_bin",
                zmq_url="tcp://127.0.0.1:6001")
            time.sleep(0.5)
            print "Scene nonpen"
            dataset_utils.DrawYamlEnvironment(
                scene_nonpen, "planar_bin",
                zmq_url="tcp://127.0.0.1:6001")
            time.sleep(0.5)
            print "Scene static"
            dataset_utils.DrawYamlEnvironment(
                scene_static, "planar_bin",
                zmq_url="tcp://127.0.0.1:6001")

            for k in range(scene_static["n_objects"]):
                obj_yaml = scene_static["obj_%04d" % k]
                # Check if x or z is outside of bounds
                pose = np.array(obj_yaml["pose"])
                if pose[0] > 1.2 or pose[0] < -1.2 or pose[1] > 2.0 or pose[1] < 0.0:
                    raise ValueError("Skipping scene due to bad projection.")

            env_name = "env_%d" % int(round(time.time() * 1000))
            with open("generated_planar_bin_static_scenes_with_context_stacks_tall_raw.yaml", "a") as file:
                yaml.dump({env_name:
                          scene_yaml}, file, Dumper=noalias_dumper)
            with open("generated_planar_bin_static_scenes_with_context_stacks_tall_nonpen.yaml", "a") as file:
                yaml.dump({env_name:
                          scene_nonpen}, file, Dumper=noalias_dumper)
            with open("generated_planar_bin_static_scenes_with_context_stacks_tall_static.yaml", "a") as file:
                yaml.dump({env_name:
                          scene_static}, file, Dumper=noalias_dumper)

        except Exception as e:
            print "Unhandled exception: ", e
