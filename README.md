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

Also `tqdm`, `attrs`, `scikit-image`

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

## Contents
- `data`: Many data generation utilities and their generated datasets.
  - `generate_cardboard_boxes.py`: Generates cardboard box meshes, textures, + SDFs.
  - `generate_cardboard_box_environments.py`: Loads random arrangements of boxes,
  simulates them, and renders labeled images from different perspecties into
  a specialized directory-based pile of data.

- `monte_carlo_carrot_flipping`: Old experiment folder for exploring the
success / failure landscape of a planar robot flipping a little half cylinder
carrot piece, w.r.t. carrot initial position and an open-loop flip trajectory.
I was trying to get a sense of how "weird" the dependency between env distribution
and success distribution is.

- `notebooks`: Tons of jupyter notebooks for various experiments. Mostly a sandbox.

- `sandbox`: Stuff I don't want to sort right now.

- `inverse_graphics`: Analysis-by-synthetis VI approach project subfolder.