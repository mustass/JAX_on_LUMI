#!/bin/bash
#SBATCH --account=project_465001020
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=56   # 7 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00
DIRECTORY=path
PATH_TO_VENV=path

srun singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source ${PATH_TO_VENV}/bin/activate; python3 ${DIRECTORY}/scripts/dataparallelism.py"