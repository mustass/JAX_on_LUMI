$WITH_CONDA
PATH_TO_VENV=path
python3 -m venv $PATH_TO_VENV --system-site-packages
source $PATH_TO_VENV/bin/activate
python3 -m pip install optax==0.2.2 flax==0.8.3 jax==0.4.28
python3 -m pip install -r requirements.txt --ignore-installed