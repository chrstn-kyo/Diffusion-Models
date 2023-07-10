# Diffusion-Models
Model to generate images from the dataset MNIST

## Setting up the environment

```bash
virtualenv venv --python=python3.9
source venv/bin/activate
pip3 install -r requirements.txt
```

The requirements.txt file is actually the list of modules on the server, please do not change it.

## Running the commands

To regenerate the python code from the notebook, run `./gen.sh`. It overwrites `notebooks/main.ipynb` and `colabcode.py`.

To generate 2048 samples in the directory `samples`, run `python generate.py`.
