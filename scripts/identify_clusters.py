'''
The script processes LAMMPS DUMP trajectories to identify CH* bond fragments during length of simulations using networkx packages.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from ase import Atoms
from ase.geometry import get_distances

import random
import networkx as nx

num_atoms=356
num_meth=20

bond_threshold=1.3
symbols = ['C', 'H', 'H', 'H', 'H']*num_meth

input_file='dump_Ti2C.lammpstrj' #'dump.neigh'
arr = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]*10001)

ro = [col+(num_atoms+9)*(i) for i, col in enumerate(arr)]
ro = np.array(ro).flatten()
#print(ro)
atom_dict={1:"Ti",2:"C",3:"H"}
snps = list(range(0,10000,1)) + [10000]

#res = random.sample(range(500,10000),10)

def process_chunk(chunk):
    first_row = chunk.iloc[0] 
    other_rows = chunk.iloc[1:]  
    return chunk 


# Define bond length cutoffs
CH_bond_cutoff_max = 1.3  # C-H bond max Å
# Dictionary to store connection summaries for each frame
connection_summary = {}

for i, df in enumerate(pd.read_csv(input_file, skiprows = ro, chunksize=num_atoms, names = ['id','type','x','y','z','fx','fy','fz'], usecols = ['id','type','x','y','z'], delim_whitespace=True, iterator = True)):
    if i%100 == 0:
        print(f"ts:{snps[i]*100} fs")
    df.set_index('id', inplace=True)
    ads=df[df.index <101]
    grouped = ads.groupby(ads.index // 5,group_keys=True).apply(process_chunk)
    molecule = Atoms(symbols=symbols, positions=grouped.values[:,1:4])
    
    # Create a graph using NetworkX
    G = nx.Graph()

    # Add atoms as nodes
    for j, atom in enumerate(symbols):
        G.add_node(j, element=atom)
                
    for k, atom_k in enumerate(molecule):
        for l, atom_l in enumerate(molecule):
            if k >= l:
                continue  # Avoid duplicate checks and self-bonding
            dist = molecule.get_distance(k, l)
            
            # Check for C-H bonds (distance ≤ 1.2 Å)
            if (atom_k.symbol == 'C' and atom_l.symbol == 'H' or atom_k.symbol == 'H' and atom_l.symbol == 'C') and dist <= CH_bond_cutoff_max:
                G.add_edge(k, l, distance=dist)

            
    # Analyze C atom connectivity in the graph
    c_connected_4 = []
    c_connected_3 = []
    c_connected_2 = []
    c_connected_1 = []
    c_connected_0 = []

    for m, atom_type in enumerate(symbols):
        if atom_type == 'C':
            neighbors = list(G.neighbors(m))  # Get neighbors of this carbon atom
            num_connections = len(neighbors)
            
            # Classify based on number of connections
            if num_connections == 4:
                c_connected_4.append(m)
            elif num_connections == 3:
                c_connected_3.append(m)
            elif num_connections == 2:
                c_connected_2.append(m)
            elif num_connections == 1:
                c_connected_1.append(m)
            else:
                c_connected_0.append(m)
   
    connection_summary[i] = {
        'C connected to 4 atoms': c_connected_4,
        'C connected to 3 atoms': c_connected_3,
        'C connected to 2 atoms': c_connected_2,
        'C connected to 1 atom': c_connected_1,
        'C connected to 0 atoms': c_connected_0,
    }

# Convert bond_summary dictionary to a pandas DataFrame
df_bond_summary = pd.DataFrame(connection_summary).T

# Set proper column names
df_bond_summary.columns = [
    'CH4',
    'CH3*',
    'CH2*',
    'CH*',
    'C*',
]
# Function to replace lists with their lengths
df_bond_summary = df_bond_summary.applymap(lambda x: len(x) if isinstance(x, list) else x)
df_bond_summary.to_csv("bond_data_600K_Ti2C.csv",index=False) 
