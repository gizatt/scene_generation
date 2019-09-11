from __future__ import print_function
from collections import namedtuple
from copy import deepcopy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import yaml

import pydrake  # MUST BE BEFORE TORCH OR PYRO
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import torch.multiprocessing as mp
from multiprocessing.managers import SyncManager
from tensorboardX import SummaryWriter
from torchviz import make_dot

import scene_generation.data.dataset_utils as dataset_utils
from scene_generation.models.probabilistic_scene_grammar_nodes import *
from scene_generation.models.probabilistic_scene_grammar_model import *

ScoreInfo = namedtuple('ScoreInfo', 'joint_score latents_score f total_score')

if __name__ == "__main__":
    seed = 52
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pyro.enable_validation(True)

    start = time.time()
    pyro.clear_param_store()
    trace = poutine.trace(generate_unconditioned_parse_tree).get_trace(root_node="dish_bin")
    parse_tree = trace.nodes["_RETURN"]["value"]
    end = time.time()

    print("Generated: ", parse_tree)