import argparse
import numpy as np
import time

import pydrake

import scene_generation.data.dataset_utils as dataset_utils

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path",
                        help="Dataset to visualize.")
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
            index = np.random.randint(0, len(dataset))
            print("Drawing index %d" % index)
            dataset_utils.DrawYamlEnvironment(dataset[index], 'dish_bin')
            input("Press to continue...")

    elif args.index >= 0:
        print("Drawing index %d" % args.index)
        dataset_utils.DrawYamlEnvironment(
            dataset[args.index], 'dish_bin')
        print(dataset[args.index])
    else:
        index = dataset.get_environment_index_by_name(args.name)
        print("Drawing index %d" % index)
        dataset_utils.DrawYamlEnvironment(dataset[index], 'dish_bin')
        print(dataset[args.index])
