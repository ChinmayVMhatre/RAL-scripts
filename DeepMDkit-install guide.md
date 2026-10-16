- Begin a gpu session using the following command: 
	Pitt : `srun -M gpu -p preempt --gres=gpu:1  --pty bash -i`
- Check whether the session is running or not by giving command: 
	`nvidia-smi`: if it gives out an out put you are logged in to the GPU node

- Make sure to have a cuda driver and a C++ compiler preinstalled before initiating the installation. (Look for sinfo or respective command on HPC cluster to check the GPU information)
### Pre-requisites = GCC and CUDA (12.0.0 preferred)

- Create DeepMD-kit installation by command: 
`conda create -n deepmd`
The environment will not contain any packages. Then, install following packages:

`> conda activate deepmd
`> conda install cudnn cudatoolkit 

Then `module load cuda/11.8 and module load gcc` ,crc-quota

`CONDA_OVERRIDE=11.8 conda install tensorflow=2.17.0=cuda* -c conda-forge`
 
`CONDA_OVERRIDE=11.8 conda install deepmd-kit=3.0.0=cuda* lammps horovod -c conda-forge`

This will by-default download cpu compatible packages, you need to use another command to enable GPU compatibility: specifically for tensorflow, pytorch, deepmd-kit and lammps.

Check for GPU enable of these packages by `conda search PACKAGE_NAME -c conda-forge`
- Since you are logged into to GPU node and module CUDA loaded, make sure to have a GPU enabled version of pytorch backend (To know more about it: [1. Backend — DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/stable/backend.html)), using command:
`CONDA_OVERRIDE_CUDA=12.0.0  conda install pytorch::pytorch-cuda -c conda-forge`

## Testing installation:

- After finishing up the installation, check if the installation is completed with: `dp -h` command. It should show the help option for dp executable
- Check the version of DeepMD-kit is installed by: `dp --version`
- Check LAMMPS executable by `lmp -h`

## Create a IPykernel for Jupyter Notebook:

- pip install ipykernel
- python -m ipykernel install --user --name=deepmdGPU

