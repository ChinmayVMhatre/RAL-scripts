'''
Create DeepMD datastructure from VASP OUTCAR 
'''
import numpy as np
import os
import sys

from ase import Atoms
from ase.io import read,write

import dpdata

for i in range(5):
    outcar_path = f'{i}/OUTCAR'

    # Try to read potential energy
    try:
        En = read(outcar_path).get_potential_energy()
        print(En)
    except Exception as e:
        print(f"Error reading energy from {outcar_path}: {e}, skipping...")
        continue

    # Skip if energy is positive
    if En > 0:
        continue

    # Try to create LabeledSystem and save to deepmd npy format
    try:
        g_outcar = dpdata.LabeledSystem(outcar_path, fmt='vasp/outcar')

        # Check if the system has non-empty energy and force arrays
        if len(g_outcar['energies']) == 0 or len(g_outcar['forces']) == 0:
            print(f"No valid data in {outcar_path}, skipping...")
            continue

        # Save to deepmd npy format
        g_outcar.to_deepmd_npy(f'deepmd/raw-file-npy-{i}')

    except ValueError as ve:
        print(f"ValueError for {outcar_path}: {ve}, skipping...")
    except Exception as e:
        print(f"Unexpected error for {outcar_path}: {e}, skipping...")
