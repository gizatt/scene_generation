# Scene Generation

Notebooks relevant to the 9.660 final project report:
- 20181204 Planar Scene Arrangements Pyro, Single Object, Single Multivariate Normal, with Projections
- 20181213 Differentiable Feasibility Projection Sandbox.ipynb
- 20181216 Planar Scene Arrangements Pyro, Two Object, with Projections.ipynb


Mostly a project sandbox. Personal notes:
- On `FateAmenableToChange`, the env is `py27_pyro`.
- On `ProblemChild`, the env is `py27`.

## Deps

```
conda create -n py27_pyro python=2.7
conda install numpy scipy matplotlib pyyaml ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

Install pytorch 1.0.0, e.g. if you have CUDA 9.0, use:

```
conda install pytorch torchvision -c pytorch
```

or otherwise reference https://pytorch.org/.


Then pull down Pyro from https://github.com/uber/pyro (version needs
to be more recent than 3.0) and `pip install .`. I think now that
they published 3.0, it could be gotten from `conda install pyro-ppl`, but
the install from source is pretty painless since it's close to pure
Python.

## TODOS

### BROAD TODOS
- Cite Lawson Wong's work here and see what I can learn from it.
  - [Collision-Free State Estimation (ICRA 2012)](http://people.csail.mit.edu/lsw/papers/icra2012-collision-free.pdf)
  is directly relevant. The core idea is that he models object states with a multivariate Gaussian for
  the recursive state estimation, but use a NLP for projection when generating a MAP estimate.
  - [Manipulation-based Active Search for Occluded Objects (ICRA 2013)](http://people.csail.mit.edu/lsw/papers/icra2013-search.pdf)
  is also really directly relevant: *"We present a
novel generative model for representing container contents by
using object co-occurrence information and spatial constraints."* He breaks the world into "containers"
that have a latent composition (distributed over objects, with a prior informing what objects are
likely to co-occur), and a generative process for placing the objects into the container. The composition
is modeled with a logistic-multivariate-normal distribution (not a dirichlet, as a dirichlet doesn't
keep track of covariances between object types), with the logistic part used to transform it into
a valid simplex over object types. Inference over this is done with MCMC from training data. Posterior
inference of scene configurations given some observed object placements and the composition is done
with a hand-specified sampling method that probes how easy it is to insert new objects in the unobserved space.
  - [Xinkun Nie's extension](http://people.csail.mit.edu/lsw/papers/icra2016-search.pdf) of the ICRA 2013 work
  includes a description of how sampling is done for sampling scene configurations, and details
  a generative model for the scene with explicit add/remove/relabel actions for objects in the scene,
  and describes how search is done over that model with MH / MCMC.
  - His [PhD thesis](http://people.csail.mit.edu/lsw/papers/mit2016-thesis.pdf) contains
  a lot of more advanced work, but I didn't find anything directly relevant.
  (It's mostly on nonparametric (DP-based) generative models for MHT?)

- Generate a "harder" dataset -- simplest extension might be planar with these two classes, but with actual
consistent object relationships
  - Medium difficulty: small box tends to cluster around the long box.
  - High difficulty, but really interesting: Small boxes tend to form pyramids in the the global +x direction.
  - High difficulty, but really interesting: Small boxes tend to form pyramids in the the box +x direction. (These two pick apart whether we should represent pairwise relationships in object or world frame... hmm... object frame is probably more natural, but both would be nice... when would world-frame ever be important?)
- Implement pairwise object relationships.
- Establish a way of doing model comparison. One of my core claims is that having projection enables simpler models to have higher accuracy (as long as the projection reflects reality in a useful way), since the models no longer need to explicitly capture that complex manifold. Can I just straight-up compare ELBO on a held-out test set? (Literature *sort of implies* yes, with a little controversy.)


### SPECIFIC THINGS
- Review the background in [Manipulation-based Active Search for Occluded Objects (ICRA 2013)](http://people.csail.mit.edu/lsw/papers/icra2013-search.pdf)
and make sure I am aware of everything he mentions.
- Deep dive into Drake MBT random support -- try porting the generative model over to that language.
  - Can I ask for the same derivatives (w.r.t. the primitive distribution inputs)?
  - Can I at least sample feasible configurations?
- Finish resolving the improper X vs Y scaling fitting that I observed at the end of the 9.660 final project push: make a simple single-object model distributed with higher variance in X than Y (or vice versa) and make sure that works properly, and that making that fix improves the fits in the 2-object case. I think the fixes I've made (mostly due to my misunderstanding of how VI was working and what latent variables needed to exist) should have fixed this now.
- Refactor:
  - Move models into a subfolder to start simplifying new notebooks.
  - Clean out and reorganize data with proper train/valid/test split.