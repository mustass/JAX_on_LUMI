# LUMI Hackathon: Laplax: scalable uncertainty quantification
Use this collaborative document to plan your steps, discuss questions and take notes.

[Overview page of markdown syntax and HedgeDoc  features like embedding pictures, pdf, links](https://md.sigma2.no/features#Table-of-Contents)

## Goals for Hackathon
1. Perform Jacobian-vector and vector-Jacobian products on AMD GPUs @LUMI
2. Scale Jacobian-vector and vector-Jacobian products to large models and big datasets. Large models and big datasets each increase the complexety of the problem as the Jacobian matrix grows with both. Scaling is supposed to be across multiple GPUs and nodes. 

## Steps and whose working on what
1. Mininmal example 
2. 

## Notes
- Our project is project_465001020



### commands used once to create a virtual env

To get inside:
```
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems/default
singularity shell /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif
```
Once inside:
```
$WITH_CONDA
```
Navigate to the repo you want and run a `setup.sh` script.
For some reason care needs to be given to how to install jax. Just installing requirements does not seem to avoid updating the jax version. 
a manual call of 

```
pip install optax==0.2.2 flax==0.8.3 jax==0.4.28
```
is needed before 

```
pip install -r requirements.txt
```

### commands to run a job on a GPU node

A nice environment variable to set (dont ask just do)
```
export PYTHONNOUSERSITE=1
```

Some modules to get one started. Second binds the folders so that they are visible to all compute nodes
```
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems
```


Options and project boilerplate
```
--account=project_465001020 --partition=small-g --nodes=1 --gpus=1 --time=05:00
```

Allocate a node: 

```
salloc --account=project_465001020 --partition=small-g --nodes=1 --gpus=2 --time=10:00
```

Send stuff into it:

```
srun singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /project/project_465001020/hackathon/venv/bin/activate; python3 /project/project_465001020/hackathon/lumi_hackathon/scripts/train_cifar100.py"
```

===============

A dump of commands to run

```
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems

srun --account=project_465001020 --partition=small-g --nodes=1 --gpus=1 --time=05:00 singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /scratch/project_465001020/hackathon/venv/bin/activate; python3 src/training/train_fc.py"
```
