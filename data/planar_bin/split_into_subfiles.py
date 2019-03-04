import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    print "WARNING: Defaulting to non-C [slow] YAML loader."
    from yaml import Loader

f = open("planar_bin_static_scenes_stacks.yaml", "r")
fo = open("planar_bin_static_scenes_stacks_new.yaml", "w")

raw_yaml_environments = yaml.load(f, Loader=Loader)
print "Starting with ", len(raw_yaml_environments), " environments."
prune_keys = []
for key in raw_yaml_environments:
    env = raw_yaml_environments[key]
    for k in range(env["n_objects"]):
        obj_yaml = env["obj_%04d" % k]
        # Check if x or z is outside of bounds
        pose = np.array(obj_yaml["pose"])
        if pose[0] > 2.0 or pose[0] < -2.0 or pose[1] > 1.0 or pose[1] < 0.0:
            prune_keys.append(key)
            break

for key in prune_keys:
    del raw_yaml_environments[key]
print "Ending with ", len(raw_yaml_environments), " environments."
yaml.dump(raw_yaml_environments, fo)
