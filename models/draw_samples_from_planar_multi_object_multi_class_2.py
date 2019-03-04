import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

import pydrake
import pyro

import scene_generation.data.dataset_utils as dataset_utils
from scene_generation.models.planar_multi_object_multi_class_2 import (
    MultiObjectMultiClassModel)

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
    model = MultiObjectMultiClassModel(scenes_dataset)
    pyro.get_param_store().load(args.param_path)

    plt.plot(10, 10)
    while (1):
        generated_data, generated_encodings, generated_contexts = model.model()
        scene_yaml = scenes_dataset.convert_vectorized_environment_to_yaml(
            generated_data)[0]
        try:
            dataset_utils.DrawYamlEnvironmentPlanar(
                scene_yaml, "planar_bin", ax=plt.gca())
            plt.pause(1.0)
        except Exception as e:
            print "Unhandled exception: ", e
        plt.gca().clear()
