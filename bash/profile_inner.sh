$WITH_CONDA

source $VENV_PATH

OMNITRACE=/pfs/lustrep3/scratch/project_462000394
PROJECT_PATH=/project/project_465001020
PATH=${OMNITRACE}/amd-sw/omnitrace/1.12.0-rocm6.2.x/bin:${PATH}
LD_LIBRARY_PATH=${OMNITRACE}/amd-sw/omnitrace/1.12.0-rocm6.2.x/lib:${LD_LIBRARY_PATH}

export PYTHONPATH=$OMNITRACE/amd-sw/omnitrace/1.12.0-rocm6.2.x/lib/python/site-packages:$PYTHONPATH
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
export OMNITRACE_OUTPUT_PATH="${PROJECT_PATH}/hackathon/lumi_hackathon/omnitrace/${SLURM_JOB_NAME}"
omnitrace-python -- -u "${PROJECT_PATH}/hackathon/lumi_hackathon/scripts/dataparallelism.py"
   
