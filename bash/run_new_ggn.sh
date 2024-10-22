#!/bin/bash
#SBATCH --account=project_465001020
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7   # 7 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

srun singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /project/project_465001020/hackathon/venv/bin/activate; python3 /project/project_465001020/hackathon/lumi_hackathon/scripts/dataparallelism_multinode_ggn_expt.py"