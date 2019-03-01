import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

f = open("planar_bin_static_scenes.yaml", "r")
fo = open("planar_bin_static_scenes_new.yaml", "w")

raw_yaml_environments = yaml.load(f, Loader=Loader)
for key in raw_yaml_environments:
    env = raw_yaml_environments[key]
    for k in range(env["n_objects"]):
        obj_yaml = env["obj_%04d" % k]
        if obj_yaml["class"] == "2d_sphere":
            obj_yaml["params"] = [obj_yaml["radius"]]
            obj_yaml["params_names"] = ["radius"]
            del obj_yaml["radius"]
        elif obj_yaml["class"] == "2d_box":
            obj_yaml["params"] = [obj_yaml["height"], obj_yaml["length"]]
            obj_yaml["params_names"] = ["height", "length"]
            del obj_yaml["length"]
            del obj_yaml["height"]
yaml.dump(raw_yaml_environments, fo)