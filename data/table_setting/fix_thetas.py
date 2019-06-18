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
    found = 0
    real_num_objects = 0
    for k in range(env["n_objects"] + 100):
        if ("obj_%04d" % k) not in env.keys():
            continue
        elif env["obj_%04d" % k]["class"] == "table":
            env.pop("obj_%04d" % k)
        else:
            env["obj_%04d" % real_num_objects] = env.pop("obj_%04d" % k)
            real_num_objects += 1
    env["n_objects"] = real_num_objects

with open(f_in_base + "_fixed.yaml", "w") as f:
    yaml.dump(raw_yaml_environments, f)