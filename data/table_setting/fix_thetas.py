import numpy as np
import random
import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    print("WARNING: Defaulting to non-C [slow] YAML loader.")
    from yaml import Loader

f_in_base = "table_setting_environments"

with open(f_in_base + ".yaml", "r") as f:
    raw_yaml_environments = yaml.load(f, Loader=Loader)
num_envs = len(raw_yaml_environments)
print("Input: %d environments" % num_envs)

for key in raw_yaml_environments:
    env = raw_yaml_environments[key]
    for k in range(env["n_objects"]):
        orig = env["obj_%04d" % k]["pose"][2]
        orig *= -1
        orig = (orig + np.pi*2) % (np.pi*2)
        env["obj_%04d" % k]["pose"][2] = orig

with open(f_in_base + "_fixed.yaml", "w") as f:
    yaml.dump(raw_yaml_environments, f)