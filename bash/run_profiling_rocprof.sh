#!/usr/bin/env -S bash -e
#SBATCH --job-name=profile_rocprof_small_numworkers
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output="/project/project_465001020/hackathon/lumi_hackathon/rocprof/log_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=15:00
#SBATCH --account=project_465001020

ROCPROF_OUTPUT_PATH=/project/project_465001020/hackathon/lumi_hackathon/rocprof
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif
VENV_PATH=/project/project_465001020/hackathon/venv/bin/activate
SCRIPT=/project/project_465001020/hackathon/lumi_hackathon/scripts/dataparallelism.py

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems

srun singularity exec $SIF bash -c "\$WITH_CONDA; source $VENV_PATH ; rocprofv3 --output-format pftrace -d $ROCPROF_OUTPUT_PATH  --stats --hip-trace --kernel-trace --memory-copy-trace -- python3 $SCRIPT"