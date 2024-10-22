#!/bin/bash
#SBATCH --account=project_465001020
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1    # we start a single torchrun process, which will take care of spawning more
#SBATCH --cpus-per-task=7   # 7 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:15:00

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems
module load CrayEnv
module load rocm/6.2.2
module use /pfs/lustrep3/scratch/project_462000394/amd-sw/modules
module load omnitrace/1.12.0-rocm6.0.x
export ROCP_METRICS=/opt/rocm/lib/rocprofiler/metrics.xml

srun omnitrace -- singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /project/project_465001020/hackathon/venv/bin/activate; python3 /project/project_465001020/hackathon/lumi_hackathon/scripts/dataparallelism.py"


#srun singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /project/project_465001020/hackathon/venv/bin/activate; rocprofv3 --output-format pftrace -d /project/project_465001020/hackathon/lumi_hackathon/rocprof --stats --hip-trace --kernel-trace --memory-copy-trace -- python3 /project/project_465001020/hackathon/lumi_hackathon/scripts/dataparallelism.py"