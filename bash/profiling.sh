module load CrayEnv
module load buildtools/24.03

module load PrgEnv-cray/8.5.0
module load cce/17.0.1
module load craype-accel-amd-gfx90a
module load craype-x86-trento
module load cray-python

module use /pfs/lustrep3/scratch/project_462000394/amd-sw/modules

module load rocm/6.0.3 omnitrace/1.12.0-rocm6.0.x omniperf/2.1.0

export SALLOC_ACCOUNT=project_465001020
export SBATCH_ACCOUNT=project_465001020
