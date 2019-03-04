import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    print "WARNING: Defaulting to non-C [slow] YAML loader."
    from yaml import Loader

f = open("planar_bin_static_scenes_stacks.yaml", "r")
fo = open("planar_bin_static_scenes_stacks_tall.yaml", "w")

raw_yaml_environments = yaml.load(f, Loader=Loader)
print "Starting with ", len(raw_yaml_environments), " environments."
prune_keys = []

num_envs = len(raw_yaml_environments.keys())
env_keys_in_order = sorted(raw_yaml_environments.keys())
for i, key in enumerate(env_keys_in_order):
    env = raw_yaml_environments[key]
    highest_z = 0.
    pruned_for_oob = False
    env["n_objects"] = int(env["n_objects"])
    for k in range(env["n_objects"]):
        obj_yaml = env["obj_%04d" % k]
        # Check if x or z is outside of bounds
        pose = np.array(obj_yaml["pose"])
        if pose[0] > 2.0 or pose[0] < -2.0 or pose[1] > 1.0 or pose[1] < 0.0:
            pruned_for_oob = True
            prune_keys.append(key)
            break
        highest_z = max(highest_z, pose[1])
    if not pruned_for_oob and highest_z < 0.5:
        prune_keys.append(key)


for key in prune_keys:
    del raw_yaml_environments[key]
print "Ending with ", len(raw_yaml_environments), " environments."
yaml.dump(raw_yaml_environments, fo)
