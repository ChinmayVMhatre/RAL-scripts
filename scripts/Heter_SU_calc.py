'''
This code calculates Service Units consumed during active learning training of heterogenous phase systems using ASE-NEB, including neural network training, molecular dynamics runs and DFT relabling. 
The details about the archetecture can be found on https://crc.pitt.edu/overview-crc-services/computing-hardware
'''

import numpy as np
import os
import sys
import time
import re
import datetime
import math

path = 'YOUR/DIRECTORY'

RALgen=4
folders = [item for item in os.listdir(path+'/gen-1/train') if os.path.isdir(os.path.join(path,'gen-0','train', item))]

def get_hr(time_str):
    h, m, s = time_str.split(':')
    return int(h) + int(m)/60 + int(s)/3600

runtime=[]
hr_train=[]

#training on GPU for RAL
#------------------------------------------------------------------

for gen in range(1,RALgen):
    for i in folders:
        for dirpath,dirname,filenames in os.walk(os.path.join(f'{path}/gen-{gen}/train',i)):
            #print(dirpath)
            slurm_files = [filename for filename in filenames if filename.startswith("slurm")]
            #print(slurm_files)
            if slurm_files:
                latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                latest_file_path = os.path.join(dirpath, latest_file)
                #print(latest_file_path)  # Print the path of the latest file    
                
                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime.append(line.split()[1])
for i in runtime:
    hr_train.append(get_hr(i))

print("#------------------------------------------------------------------")
print("SU for train - Done")
print('#------------------------------------------------------------------\n')

#------------------------------------------------------------------
#NEB running on SMP for RAL

runtime_gsm=[]
hr_gsm=[]

for gen in range(1,RALgen):
    folders_gsm = [item for item in os.listdir(path+f'/gen-{gen}/neb') if os.path.isdir(os.path.join(f'{path}',f'gen-{gen}','neb',item))]
    #print(folders_gsm)
    for i in folders_gsm:
        for dirpath,dirname,filenames in os.walk(os.path.join(f'{path}/gen-{gen}/neb',i)):
            #print(dirpath)
            slurm_files = [filename for filename in filenames if filename.startswith("slurm")]
            #print(slurm_files)
            if slurm_files:
                latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                latest_file_path = os.path.join(dirpath, latest_file)
                #print(latest_file_path)  # Print the path of the latest file    
                
                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime_gsm.append(line.split()[1])
for i in runtime_gsm:
    hr_gsm.append(get_hr(i))

print("#------------------------------------------------------------------")
print("SU for NEB - Done")
print('#------------------------------------------------------------------\n')

#------------------------------------------------------------------
#Relabeling  on SMP for RAL

runtime_relabel = []
hr_relabel = []

# Correctly accumulate all rxn folders in a single list for processing
all_rxn_folders = []

for gen in range(1,RALgen):
    relabel_path = os.path.join(path, f'gen-{gen}', 'relabel.neb')
    gsm_path = os.path.join(path, f'gen-{gen}', 'relabel.neb')
    folders_gsm = [item for item in os.listdir(gsm_path) if os.path.isdir(os.path.join(gsm_path,item))]
    #print(folders_gsm)
    for j in folders_gsm:
        folders_rxn= [item for item in os.listdir(os.path.join(path,f'gen-{gen}','relabel.neb',j)) if os.path.isdir(os.path.join(path,f'gen-{gen}','relabel.neb',j,item)) and item.startswith('0')]
        #print(folders_rxn)
        #print(f"Generation {gen} GSM folders: {folders_rxn}")
        for i in folders_rxn:
            for dirpath, dirnames, filenames in os.walk(os.path.join(path,f'gen-{gen}','relabel.neb',j,i)):
                #print(dirnames)
                slurm_files = [f for f in filenames if f.startswith("report")]
                
                if slurm_files:
            # Find the latest slurm file based on the modification time
                    latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                    latest_file_path = os.path.join(dirpath, latest_file)
                    #print("Processing file:", latest_file_path)

            # Read from the latest slurm file
                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime_relabel.append(line.split()[1])
                            #print("Found RunTime:",line.split()[1])
    else:
        continue

# Assuming get_hr is a function you have defined to convert runtime into hours
for runtime in runtime_relabel:
    hr = get_hr(runtime)  # Define this function or modify as needed
    hr_relabel.append(hr)

print("#------------------------------------------------------------------")
print("SU for relabel NEB - Done")
print('#------------------------------------------------------------------\n')

#------------------------------------------------------------------

#------------------------------------------------------------------
#Relabeling  on SMP for RAL-MD

runtime_relabel_md = []
hr_relabel_md = []

# Correctly accumulate all rxn folders in a single list for processing
all_rxn_folders_md = []

for gen in range(1,RALgen):
    relabelmd_path = os.path.join(path, f'gen-{gen}', 'relabel.md')
    md_path = os.path.join(path, f'gen-{gen}', 'relabel.md')
    folders_md = [item for item in os.listdir(md_path) if os.path.isdir(os.path.join(md_path,item))]
    #print(folders_md)
    for j in folders_md:
        #print(j)
        folders_rxn= [item for item in os.listdir(os.path.join(path,f'gen-{gen}','relabel.md',j)) if os.path.isdir(os.path.join(path,f'gen-{gen}','relabel.md',j,item)) and item.startswith('image')]
        #print(folders_rxn)
        #print(f"Generation {gen} GSM folders: {folders_rxn}")
        for i in folders_rxn:
            for dirpath, dirnames, filenames in os.walk(os.path.join(path,f'gen-{gen}','relabel.md',j,i)):
                #print(dirpath)
                slurm_files = [f for f in filenames if f.startswith("report")]

                if slurm_files:
            # Find the latest slurm file based on the modification time
                    latest_file = max(slurm_files, key=lambda x: os.path.getmtime(os.path.join(dirpath, x)))
                    latest_file_path = os.path.join(dirpath, latest_file)
                    #print("Processing file:", latest_file_path)

            # Read from the latest slurm file
                with open(latest_file_path,'r') as f:
                    for line in f.readlines():
                        if 'RunTime' in line:
                            runtime_relabel_md.append(line.split()[1])
                            #print("Found RunTime:",line.split()[1])
    else:
        continue

# Assuming get_hr is a function you have defined to convert runtime into hours
for runtime in runtime_relabel_md:
    hr = get_hr(runtime)  # Define this function or modify as needed
    hr_relabel_md.append(hr)

print("#------------------------------------------------------------------")
print("SU for relabel NEB-MD - Done")
print('#------------------------------------------------------------------\n')

#------------------------------------------------------------------

#print(hr_relabel_md)

#Information on billing
#training on GPU
billing_gpu = 8
cpu_per_node=12
node_gpu=1

#training,SE-GSM and Relabeling ON SMP
billing_smp = 1
m_cpu_per_node= 24
node_smp=1

hr_train=np.array(hr_train)
hr_gsm=np.array(hr_gsm)
hr_relabel=np.array(hr_relabel)
hr_relabel_md = np.array(hr_relabel_md)

SU_train =  math.ceil(sum(node_gpu * cpu_per_node * (hr_train) * billing_gpu))
SU_explore = math.ceil(sum(node_smp * m_cpu_per_node * (hr_gsm) * billing_smp/2))
SU_relabel = math.ceil(sum(node_smp * m_cpu_per_node * (hr_relabel) * billing_smp))
SU_relabel_md = math.ceil(sum(node_smp * m_cpu_per_node * (hr_relabel_md) * billing_smp))

sum_SU_gpu = SU_train
sum_SU_smp = SU_explore + SU_relabel + SU_relabel_md

all_su = sum_SU_gpu +sum_SU_smp 

print(f"Total SU need for TiC-RAL training:{all_su}")
print(f'Train SU = {SU_train} : explore-MD SU = {SU_explore} : relabel SU = {SU_relabel} : relabel MD = {SU_relabel_md}')
