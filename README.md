# How to run code on LUMI AMD GPUs (JAX)
This repo is a collection of notes from the LUMI Hackathon


## LUMI G node

A node has 4 AMD MI250X GPUs with 128GB memory. However, the MI250x GPU modules have two Graphics Compute Dies, each with 64GB memory. So, that's in total 8 devices with 64GB memory each. The latter is how we think about it when interacting with lumi. 8 GPUs with 64GB memory each. JAX also sees 8 devices. 

![LUMI-G node](/pictures/lumig-node-overview.svg)

https://docs.lumi-supercomputer.eu/hardware/lumig/

## Setup:
The login node, upon login, has no software that we can use to train our amazing models. Futhermore, it has no real *environment/container* which we can use to run our software. On LUMI, **singularity** containers are used and these can be found in the folder: `/appl/local/containers/sif-images/`. We care only about the JAX one located at `/appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif`. Obviuosly, there will be updates so the paths can change. But in general, we need a singularity container with the correct ROCM drivers and a JAX installation built. So far it seems that JAX is built from source by people working for LUMI. So, whatever we do, we NEVER update or change the `jax` and `jaxlib` installations. 

In the following we will setup a virtual environment for our python code. This needs to be done within the container we are going to use. 

### To get inside:
```
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems/default
singularity shell /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif
```

First line will load: `cotainr`, `singularity-bindings`, `singularity-userfilesystems`, `cotainr_installation`, `singularity-CPEbits` modules. I am not sure we need the `cotainr` ones, but the other three setup the `singularity` container. 

The second line binds our specific paths with singularity. So we can see our home and project folders. 

### Once inside:

We need to activate the Conda environment inside. Otherwise the installed JAX will not show:
```
$WITH_CONDA
```

With this command `jax` and `jaxlib` will be available. To use further pytrhon packages with our code, we have to create a python virtual environment and install the things inside. However, it is very important that we do NOT change the version of `jax` and prohibit any library dependent on `jax` to update it according to its dependency declarations. For this reason, once the virtual environment is created with the `--system-site-packages` flag:

```
python3 -m venv /path/ --system-site-packages
```

we have to run:

```
pip install optax==0.2.2 flax==0.8.3 jax==0.4.28 #these should also be hardcoded like this in a requirements.txt file
```

which will make sure to install `flax` and `optax` that are compatible with the `jax` version 0.4.28. For some weird reason we still need to pass `jax==0.4.28` to **really** make sure we don't update `jax`. Then we can go on installing stuff that is independent of `jax` by 

```
pip install -r requirements.txt
```

or

```
pip install .
```

with or without the editable flag `-e`. Assuming you're working from inside a root of a python package. 

### Nothing makes sense and `jax` is of a wrong version just after `$WITH_CONDA`
Given that I don't know how to use this correctly, but if one installs some packages on the login node whithout thinking, those can sometimes end up in the `~/.local/` path that singularity will always bind and thus the packages an be from there. I chose violence and deleted the whole directory at the LUMI hackathon. Quite sure it's not the best solution but it worked and didn't give any complications so far. 

### A sample setup script is available in `bash/setup.sh`

## Run on a GPU node

To run on GPU with LUMI, we need to either use the `small-g` or `standard-g` partition, where the former one is for debugging and where the ressources allocated can be placed in suboptimal layout. For instance, one does not get a full node. The second always allocates a node (8 gpus) even if we only use 2. 

One can submit jobs using `sbatch` or run jobs using `srun`. Both will have to run some code inside the singularity container. So this code needs to activate things we need as the first thing. 

A nice environment variable to set (dont ask just do)
```
export PYTHONNOUSERSITE=1
```


### `srun`

`srun` is used to launch some code in a GPU node from the login node. 

#### One can either do the whole thing in a long command: 

```
srun --account=--account=project_your_project_number0 --partition=small-g --nodes=1 --gpus=1 --time=05:00 singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /path/to/venv; python3 path/to/script.py"
```

Where: 

```
--account=--account=project_your_project_number --partition=small-g --nodes=1 --gpus=1 --time=05:00
```
just defines where to run the code, which account has the ressources and other settings.

```
 "\$WITH_CONDA; source /path/to/venv; python3 path/to/script.py"
```
is the command that is going to be executed inside the singularity container. And 

```
singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c
```
executes the command. 

#### or allocate ressources and send scripts to the allocation:


Allocate:
```
salloc --account=project_your_project_number --partition=small-g --nodes=1 --gpus=8 --time=10:00 
```

Send:

```
srun singularity exec /appl/local/containers/sif-images/lumi-jax-rocm-6.2.0-python-3.12-jax-0.4.28.sif bash -c "\$WITH_CONDA; source /path/to/venv; python3 path/to/script.py"
```

This way we will not wait each time for allocation of ressources. But the ressources will be billed for the whole time allocation is valid for. 


### `sbatch`

This method is pretty much the same as on other HPC infrastructure. We need a script `job.sh` and we submit it by

```
sbatch job.sh
```

The script itself has a header with parameters:

```
#!/usr/bin/env -S bash -e
#SBATCH --job-name=
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7 
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --output="where_to_store/log_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=15:00
#SBATCH --account=project_your_project_number
```

They are all mostly self-explanatory. `#SBATCH --mem=60G` reflects that we have 64GB memory per GPU, this flag says we want it all. A rule of thumb is to use 7 CPUS per task (7 CPUS per GPU is more correct, but the distinction between tasks and GPUs is not that clear yet.) `#SBATCH --cpus-per-task=7`.

If we run on one node with 8 GPUS, then:
```
#SBATCH --cpus-per-task=56 
#SBATCH --gpus-per-node=8
```

will set the correct ressources. 

If we run on several nodes, we need to actually have 8 tasks per node, ie. 8 processes per node. One per GPU. `#SBATCH --tasks-per-node=8`. Otherwise, we will not see the correct devices with JAX.

A full 2-node job will look as follows:

```
#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7 
#SBATCH --mem-per-gpu=60G
```

### Sample job scripts are available in `bash/run.sh` and `bash/run_multinode.sh`

## Some useful additional commands

### Connect to your job that is running

Given a `jobid`, we can open a `bash` inside it by:
```
srun --interactive --jobid=yourjobID --pty bash
```
and inside we could watch `rocm-smi` every second:
```
watch -n 1 rocm-smi
```
or do something else completely. 

### See job status:
```
squeue --me
```

### Cancel a job:

```
scancel jobid
```

# Multi GPU and Multinode optimizations (ignore this for now)

## Bind correct CPUS for correct GPU compute tiles
This is needed so that the communication overhead is minimized, for some reason when we launch a job, the GPUs are not necessarily served by CPUs closest to them. When going through this, take a look at the picture on top and see which GPUs are connected to which CPUs. It will all make sense, hopefully. 

We can see which GPUs connected to wich GPUs by `rocm-smi --showtopo`. The output will be as follows:
```
======================================= Numa Nodes =======================================
GPU[0]		: (Topology) Numa Node: 3
GPU[0]		: (Topology) Numa Affinity: 3
GPU[1]		: (Topology) Numa Node: 3
GPU[1]		: (Topology) Numa Affinity: 3
GPU[2]		: (Topology) Numa Node: 1
GPU[2]		: (Topology) Numa Affinity: 1
GPU[3]		: (Topology) Numa Node: 1
GPU[3]		: (Topology) Numa Affinity: 1
GPU[4]		: (Topology) Numa Node: 0
GPU[4]		: (Topology) Numa Affinity: 0
GPU[5]		: (Topology) Numa Node: 0
GPU[5]		: (Topology) Numa Affinity: 0
GPU[6]		: (Topology) Numa Node: 2
GPU[6]		: (Topology) Numa Affinity: 2
GPU[7]		: (Topology) Numa Node: 2
GPU[7]		: (Topology) Numa Affinity: 2
```
 This shows, that GPU0 is part of NUMA Node 3, GPU2 is part of NUMA Node 1 etc.
When we list the cpus: `lscpu` we see:
```
NUMA:                    
  NUMA node(s):          4
  NUMA node0 CPU(s):     0-15,64-79
  NUMA node1 CPU(s):     16-31,80-95
  NUMA node2 CPU(s):     32-47,96-111
  NUMA node3 CPU(s):     48-63,112-127
```
This shows, that NUMA node0 is attached to CPU 0-15, NUMA node 1 cpus 16-31 etc. We can run a new task with specific CPUs and print the actual CPUs used to check: `taskset -c 48-63 bash -c 'taskset -p $$'` to see:
```
pid 40238's current affinity mask: fefe000000000000
```
The number fefe000000000000 is a bitmap. Every `0` counts 4. There are 12 zeros, which gives 48 which match the selection `-c 48-63`.


Using this, we can set which CPUs are used by which GPU in a slurm process e.g. by executing

```
srun --account=project_your_project_number --partition=standard-g --nodes=1 --gpus=8 --time=05:00 \
--cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
bash -c 'echo "$SLURM_PROCID -- GPUS $ROCR_VISIBLE_DEVICES -- $(taskset -p $$)"' \
| sort -n -k1

```
Each bitmap number corresponds to a rank/process. That is, the first process (rank0) uses the CPUs corresponding to bitmap number 0xfe000000000000 etc.

Not sure this works properly though. I get the following output:

```
0 -- GPUS 0,1,2,3,4,5,6,7 -- pid 127528's current affinity mask: fe000000000000
```

Also not sure how this will work with multinode. Anyway, 

```
 --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
```
