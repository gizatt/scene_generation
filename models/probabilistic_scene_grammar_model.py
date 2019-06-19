import matplotlib.pyplot as plt
import numpy as np
import time

import pyro
import pyro.distributions as dist
from pyro import poutine


class ProbabilisticSceneGrammarModel():
    class Node(object):
        successors = []
        def get_successors():
            return self.successors

    class TerminalNode(object):
        pass

    class RootNode(object):
        pass

    def __init__(self):

    def model(self, data=None):
        nodes = [self._make_root_node()]
        for node in nodes:


        return pyro.sample("test", dist.Normal(0., 1.))

    def guide(self, data):
        pass


if __name__ == "__main__":
    pyro.enable_validation(True)

    model = ProbabilisticSceneGrammarModel()

    start = time.time()
    pyro.clear_param_store()
    out = model.model()
    end = time.time()

    print("Generated data in %f seconds:" % (end - start))
    print(out)
