import numpy as np
from ase import Atoms
from ase.md.andersen import Andersen
from ase.io import Trajectory
from ase.md import MDLogger
from ase.units import fs, kB
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from datetime import datetime

import dpdata
from ase.io import read,write
from deepmd.calculator import DP
from deepmd.infer import DeepPot
from deepmd.infer import calc_model_devi

import matplotlib.pyplot as plt
import time
import random
from numpy.random import uniform
import os

# Set the initial DeepMD calculator
atoms = read('POSCAR')

atoms.calc = DP(model="YOUR_DP_PATH_1")

model_list=["YOUR_DP_PATH_1",
           "YOUR_DP_PATH_2",
           "YOUR_DP_PATH_3",
           "YOUR_DP_PATH_4"]

dp_paths= []
for i in range(4):
    dp_paths += [DeepPot(f'"YOUR_DP_PATH_{i}"')]

# Define the time step (in femtoseconds) and other simulation dependent parameters
timestep = 1 #* fs
T_amp=500
T_inc=100
T_min = random.randint(2000,2000+T_inc)
T_max = random.randint(3500,3500+T_inc)
tper=uniform(2,10)
period = tper*1000
n_steps = 100000

# Store the initial cell dimensions
initial_cell = np.array(atoms.get_cell())
dte = datetime.today().strftime("%Y-%m-%d")
amplitude = uniform(0.2,0.4)  # amplitude of the sinusoidal scaling (5% of the initial length)
and_proba=0.8

# Define the sinusoidal temperature function
def temperature(t, T_min, T_max, T_amp, period):
    omega = 2 * np.pi / period
    return T_min + ((T_max - T_min)/2)*(t/(timestep*n_steps))+(T_amp*np.sin(omega * t)**2)

# Set the initial temperature and friction coefficient
initial_temperature = temperature(0, T_min, T_max, T_amp, period)
MaxwellBoltzmannDistribution(atoms, temperature_K=initial_temperature*1.0)

# Set up the logger
traj_file = 'md' + dte + '.traj'
logfile = 'md' + dte + '.log'

dyn = Andersen(atoms, timestep*fs, initial_temperature, and_proba)

trajectory = Trajectory(traj_file, 'w', atoms)
logger = MDLogger(dyn, atoms, logfile, header=True, stress=False, peratom=False, mode='a')

def write_trajectory():
    atoms.wrap()
    trajectory.write()

dyn.attach(logger, interval=10)
dyn.attach(write_trajectory, interval=10)

def predict_energy_force_deviation(atoms, atype, models):
    energies = []
    force_collections = []
    #num_atoms = len(atoms)
    # Initialize a 3D array to hold forces from all models; shape: (num_models, num_atoms, 3)
    #force_collections = np.zeros((len(models), num_atoms, 3))

    coord = atoms.get_positions().reshape([1, -1])
    cell = atoms.get_cell().reshape([1, -1])

    for model in models:
        # Assuming the eval function returns energy, forces and something else as _
        e, f, _ = model.eval(coord, cell, atype)
        print(e)
        energies.append(e)
        force_collections.append(f)
        # Store forces in the pre-allocated array
        #force_collections[i, :, :] = f.reshape(num_atoms, 3)

    energy_std_dev = np.std(np.array(energies))
    # Calculate standard deviation across the model dimension (axis=0)
    forces_std_dev = np.linalg.norm(np.std(force_collections, axis=0),axis=-1)
    force_std = np.std(forces_std_dev,axis=-1)
    #print(force_std)

    return energy_std_dev,force_std

#forces_std_dev

# Initialize the LabeledSystem with your VASP POSCAR or CONTCAR file
labeled_system = dpdata.System('POSCAR', fmt='vasp/poscar')

# Get atomic types
atype = labeled_system['atom_types']

model_deviations = []
deviation_threshold = 0.8  #eV/A
energy_threshold = 0.5 #eV/N^0.5

print('-----------------------------------------------------------------------------')
print("Initiating MD.......")
print('-----------------------------------------------------------------------------')
# Execution time!!
start_time = time.time()

for step in range(n_steps):
    omega = 2 * np.pi / period
    scale_factor = 1 + (amplitude/2)*(step/(timestep*n_steps))+(amplitude*np.sin(omega * step)**2) # Convert time to picoseconds
    #print(scale_factor)
    new_cell = initial_cell * scale_factor
    dyn.atoms.set_cell(new_cell, scale_atoms=True)
    #print(dyn)
    current_temperature = temperature(step,T_min,T_max,T_amp,period)
    print(current_temperature)
    dyn.set_temperature(current_temperature)
    dyn.run(1)
    
    coord = atoms.get_positions().reshape([1, -1])
    cell = atoms.get_cell().reshape([1, -1])
    
    model_devi_f = calc_model_devi(coord, cell, atype, dp_paths,nopbc=False)
    #std_e = predict_energy_force_deviation(atoms, atype, dp_paths)
    std_e, std_dev= predict_energy_force_deviation(atoms, atype, dp_paths)
     
    #print(model_devi_f[0])
    #std_dev=(model_devi_f[0][4]- model_devi_f[0][5])/4
    print(std_dev,(std_e/np.sqrt(atoms.get_global_number_of_atoms())))
    
    if std_dev > deviation_threshold or std_e/np.sqrt(atoms.get_global_number_of_atoms()) > energy_threshold:
        print("Terminating MD simulation due to high model deviation.")
        break

write('final-1.poscar',atoms,format='vasp')

print(f"Simulation finished.\nLog saved in '{logfile}'.")
print(f'Simulation time: {time.time() - start_time} seconds')
