import numpy as np
import random
import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    print("WARNING: Defaulting to non-C [slow] YAML loader.")
    from yaml import Loader

f_in_base = "planar_tabletop_lines_scenes"
train_fraction = 0.8
max_environments_per_subfile = 100

with open(f_in_base + ".yaml", "r") as f:
    raw_yaml_environments = yaml.load(f, Loader=Loader)
num_envs = len(raw_yaml_environments)
print("Input: %d environments" % num_envs)

all_keys = raw_yaml_environments.keys()
inds = range(num_envs)
random.shuffle(inds)

end_train_ind = int(train_fraction * num_envs)
start_test_ind = end_train_ind + 1

print("Splitting into %d train, %d test in files of size %d" % (
    int(end_train_ind),
    int(num_envs - end_train_ind),
    max_environments_per_subfile))


# Chunk off training envs
folder_name = f_in_base + "_train"
os.mkdir(folder_name)
for start_ind in range(0, end_train_ind, max_environments_per_subfile):
    end_ind = min(start_ind + max_environments_per_subfile, end_train_ind)

    these_envs = {all_keys[ind]: raw_yaml_environments[all_keys[ind]]
                  for ind in inds[start_ind:end_ind]}

    fname = os.path.join(
        folder_name,
        f_in_base + "_train_%04d_to_%04d.yaml" % (start_ind, end_ind))
    with open(fname, "w") as f:
        print("Saving %d envs to file %s" % (len(these_envs), fname))
        yaml.dump(these_envs, f)

# Chunk off test envs
folder_name = f_in_base + "_test"
os.mkdir(folder_name)
for start_ind in range(start_test_ind, num_envs, max_environments_per_subfile):
    end_ind = min(start_ind + max_environments_per_subfile, num_envs)

    these_envs = {all_keys[ind]: raw_yaml_environments[all_keys[ind]]
                  for ind in inds[start_ind:end_ind]}

    fname = os.path.join(
        folder_name,
        f_in_base + "_test_%05d_to_%05d.yaml" % (
            start_ind-start_test_ind, end_ind-start_test_ind))
    with open(fname, "w") as f:
        print("Saving %d envs to file %s" % (len(these_envs), fname))
        yaml.dump(these_envs, f)
