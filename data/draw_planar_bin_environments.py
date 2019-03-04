import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

import pydrake

import scene_generation.data.dataset_utils as dataset_utils

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path",
                        help="Planar bin dataset to visualize.")
    parser.add_argument("-i", "--index",
                        type=int,
                        default=-1,
                        help="Entry to draw, or -1 for random.")
    parser.add_argument("-n", "--name",
                        type=str,
                        default="",
                        help="Entry to draw, or -1 for random.")
    args = parser.parse_args()

    data_file = args.dataset_path
    dataset = dataset_utils.ScenesDataset(data_file)

    plt.plot(10, 10)
    if args.index < 0 and len(args.name) == 0:
        while (1):
            dataset_utils.DrawYamlEnvironmentPlanar(
                dataset[np.random.randint(0, len(dataset))],
                "planar_bin", ax=plt.gca())
            plt.pause(1.0)
            plt.gca().clear()
    elif args.index >= 0:
        dataset_utils.DrawYamlEnvironmentPlanar(
            dataset[args.index], "planar_bin", ax=plt.gca())
        print dataset[args.index]
        plt.show()
    else:
        print dataset.get_environment_index_by_name(args.name)
        dataset_utils.DrawYamlEnvironmentPlanar(
            dataset[dataset.get_environment_index_by_name(args.name)],
            "planar_bin", ax=plt.gca())
        print dataset[args.index]
        plt.show()
