#Author: Chinmay Mhatre

'''
This code calculates Service Units consumed during active learning training of homogenous phase systems using pyGSM, including neural network training, molecular dynamics runs and DFT relabling. 
The details about the archetecture can be found on https://crc.pitt.edu/overview-crc-services/computing-hardware
'''

import numpy as np
import os
import sys
import time
import re
import datetime
import math

path = os.getcwd()
ALgen = 47

folders = [item for item in os.listdir(path+'/gen-1/train') if os.path.isdir(os.path.join(path,'gen-1','train', item))]
folders_md = [item for item in os.listdir(path+'/gen-1/md/md.02.5meth') if os.path.isdir(os.path.join(path,'gen-1','md','md.02.5meth',item)) and item.isdigit()]

def get_hr(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) + int(m)/60 + int(s)/3600

#----------------------------------------------------------
#Training done on GPU

runtime=[]
hr_train=[]

for gen in range(30,ALgen+1):
    for i in folders:
        for dirpath,dirname,filenames in os.walk(os.path.join(f'YOUR/DIR/PATH/gen-{gen}/train',i)):
            for filename in filenames:
                if filename.startswith("slurm"):
                    with open(dirpath+'/'+filename,'r') as f:
                        for line in f.readlines():
                            if 'RunTime' in line:
                                runtime.append(line.split()[1])
for i in runtime:
    hr_train.append(get_hr(i))

#----------------------------------------------------------
#MD done on GPU

runtime_md=[]
hr_md=[]

# md_dense=['md.01.2meth','md.02.5meth','md.03.10meth']
md_dense=['md.high-dense']

for j in md_dense:
    for gen in range(30,ALgen+1):
        for dirpath,dirname,filenames in os.walk(os.path.join(f'YOUR/DIR/PATH/gen-{gen}/molec-dyn',j)):
            slurm_files = [filename for filename in filenames if filename.startswith("slurm")]
            #print(slurm_files)
            if slurm_files:
                latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                latest_file_path = os.path.join(dirpath, latest_file)
                #print(latest_file_path)  # Print the path of the latest file

                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime_md.append(line.split()[1])

for i in runtime_md:
    hr_md.append(get_hr(i))

#----------------------------------------------------------
#SP done on mpi

runtime_vasp=[]
hr_vasp=[]

for j in md_dense:
    for gen in range(30,ALgen+1):
        for i in range(5):
            for dirpath,dirname,filenames in os.walk(os.path.join(f'YOUR/DIR/PATH/gen-{gen}/molec-dyn',j,str(i))):
                slurm_files = [filename for filename in filenames if filename.startswith("report")]
            #print(slurm_files)
                if slurm_files:
                    latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                    latest_file_path = os.path.join(dirpath, latest_file)
                    #print(latest_file_path)  # Print the path of the latest file

                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime_vasp.append(line.split()[1])

for i in runtime_vasp:
    hr_vasp.append(get_hr(i))


#----------------------------------------------------------
#Information about billing

#training,running MD on GPU
billing_gpu = 8
cpu_per_node=12
node_gpu=1

#training,SP with VASP
billing_mpi = 1
m_cpu_per_node= 48
node_mpi=2


hr_train=np.array(hr_train)
hr_md=np.array(hr_md)
hr_vasp=np.array(hr_vasp)

SU_train = math.ceil(sum(node_gpu * cpu_per_node * (hr_train) * billing_gpu))
SU_explore = math.ceil(sum(node_gpu * cpu_per_node * (hr_md) * billing_gpu))
SU_relabel = math.ceil(sum(node_mpi * m_cpu_per_node * (hr_vasp) * billing_mpi))

sum_SU_gpu = SU_train + SU_explore
sum_SU_mpi = SU_relabel

all_su = sum_SU_gpu +sum_SU_mpi
print(f'Last AL generation : {ALgen}')
print(f'Total SU used for methylenimine : {all_su}')
print(f'Train SU = {SU_train} : explore-MD SU = {SU_explore} : relabel SU = {SU_relabel}')
