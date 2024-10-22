#!/usr/bin/env -S bash -e
#SBATCH --job-name=omnitrace_1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output="/project/project_465001020/hackathon/lumi_hackathon/omnitrace/log_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=15:00
#SBATCH --account=project_465001020

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems/default

SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif

VENV_PATH=/project/project_465001020/hackathon/venv/bin/activate
HOME_PATH=/pfs/lustrep3/users/stassyro/
OMNITRACE=/pfs/lustrep3/scratch/project_462000394
export VENV_PATH

singularity exec -B /tmp:/tmp $SIF bash /project/project_465001020/hackathon/lumi_hackathon/bash/profile_inner.sh


    