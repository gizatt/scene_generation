# Scene Generation

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