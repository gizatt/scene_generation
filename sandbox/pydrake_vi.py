from copy import deepcopy
import numpy as np
import sys

import pydrake
import pydrake.common
import pydrake.math
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
import pydrake.symbolic as sym

''' Variational Inference from Drake symbolic expressions. '''


def model(alpha):
    x = sym.Variable('x', sym.Variable.Type.RANDOM_GAUSSIAN)
    return alpha * x


random_types = [
    sym.Variable.Type.RANDOM_UNIFORM,
    sym.Variable.Type.RANDOM_GAUSSIAN,
    sym.Variable.Type.RANDOM_EXPONENTIAL
]

if __name__ == "__main__":
    # Define model
    alpha = sym.Variable("alpha")
    y = model(alpha)
    print("Model: y = %s" % str(y))

    # Generate observations
    alpha_real = 2.7
    g = pydrake.common.RandomGenerator()
    N = 5
    # TODO(gizatt) Evaluate probably works on vector types
    y_obs = np.array([
        y.Evaluate(env={alpha: alpha_real},
                   generator=g)
        for k in range(N)])
    print("Observations: y_obs = %s" % str(y_obs))

    # Solve with mean-field VI:
    # create a symbolic variational distribution ("guide")
    # that replaces every latent variable with
    # an independent parameter to be optimized
    # (alongside the model parameters).

    # Implicit convert to Formula so I can use
    # GetFreeVariables
    all_vars = (y == 0.).GetFreeVariables()

    param_syms = []
    latent_syms = []
    for var in all_vars:
        if var.get_type() == sym.Variable.Type.CONTINUOUS:
            param_syms.append(var)
        elif var.get_type() in random_types:
            latent_syms.append(var)
        else:
            raise ValueError("Unsupported parameter type.")
    print("Params: %s, Latents: %s" % (str(param_syms), str(latent_syms)))

    def sum_neg_prior_log_likelihoods(latent_decs):
        total_cost = latent_decs[0]*0
        for i, latent_sym in enumerate(latent_syms):
            if latent_sym.get_type() == sym.Variable.Type.RANDOM_UNIFORM:
                continue
            elif latent_sym.get_type() == sym.Variable.Type.RANDOM_GAUSSIAN:
                total_cost -= -((latent_decs[i])**2)/2. + \
                    pydrake.math.log(1./pydrake.math.sqrt(2*np.pi))
            elif latent_sym.get_type() ++ sym.Variable.Type.RANDOM_EXPONENTIAL:
                total_cost -= latent_decs[i]
            else:
                raise ValueError("Invalid symbolic type.")
        return total_cost

    prog = mp.MathematicalProgram()
    param_decs = prog.NewContinuousVariables(len(param_syms), 'params')
    latent_decs_per_datum = [
        prog.NewContinuousVariables(len(latent_syms), 'latents_%d' % k)
        for k in range(N)]

    # This is wrong in some way.
    # It converges to infinite alpha, very small latent x,
    # because there is no prior on alpha that keeps it from
    # growing forever, and clustering x around 1 maximizes
    # log likelihood.

    y_with_param_dec_vars = deepcopy(y)
    for param_sym, param_dec in zip(param_syms, param_decs):
        y_with_param_dec_vars = y_with_param_dec_vars.Substitute(
            param_sym, param_dec)
    for k in range(N):
        y_local = y_with_param_dec_vars
        for i, latent_sym in enumerate(latent_syms):
            y_local = y_local.Substitute(
                latent_sym, latent_decs_per_datum[k][i])
            prog.AddCost(sum_neg_prior_log_likelihoods,
                         vars=latent_decs_per_datum[k])
            prog.SetInitialGuess(latent_decs_per_datum[k][i], np.random.randn())
        print 100*(y_local - y_obs[k])**2
        prog.AddConstraint((y_local - y_obs[k])**2 <= 0.001)
    print prog.Solve(), prog.GetSolverId().name()
    print("Sol for params: %s" % str(prog.GetSolution(param_decs)))
    for k in range(N):
        print("Sol for latents for data %d: %s" %
              (k, str(prog.GetSolution(latent_decs_per_datum[k]))))
