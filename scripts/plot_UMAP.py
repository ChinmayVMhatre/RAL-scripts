'''
Plotting dimensionality reduction of the DeepMD training data into a 2D UMAP plot. The script generates barplots for 10 generations, can be modified
as per requirements.
'''

from ase.io import read, write
import ase
from ase.neb import NEB
from ase.optimize import MDMin
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS,FIRE, MDMin

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import json

import dpdata
from deepmd.calculator import DP
from deepmd.infer import DeepPot
from deepmd.infer import calc_model_devi

import umap
from dscribe.descriptors import SOAP

from sklearn.preprocessing import StandardScaler

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import ConvexHull

eV2kcal=23.060541945329334

total_generation = 10 # change total number of generations as needed
main_path= 'YOUR/DIR/PATH/FOR/DATA' # enter the path here - directory should have paths to all generations of DPs

# Define the SOAP descriptor
def get_soap_descriptor(atoms_list, species, rcut=5.0, nmax=8, lmax=6):
    # SOAP descriptor object
    soap = SOAP(
        species=species,      # List of atomic species
        periodic=False,       # Change to True if working with periodic systems
        r_cut=rcut,            # Radial cutoff (in angstroms)
        n_max=nmax,            # Number of radial basis functions
        l_max=lmax,            # Maximum degree of spherical harmonics
        sigma=0.5,            # Width of Gaussian smearing
        average = "inner",
    )
    
    # Generate SOAP features for all atoms objects
    soap_features = [soap.create(atoms) for atoms in atoms_list]
    return np.array(soap_features)

def create_atoms_from_npy(data_i, swap=False):
    # Load data from npy files
    coord = np.load(data_i + '/set.000/coord.npy')
    coord = coord.reshape(coord.shape[0], int(coord.shape[1] / 3), 3)
    box = np.load(data_i + '/set.000/box.npy')
    box = box.reshape(box.shape[0], 3, 3)
    atype = np.loadtxt(data_i + '/type.raw')
    atype_map = list(np.loadtxt(data_i + '/type_map.raw', dtype=str))
        
    atm_list = []

    # Iterate over each configuration
    for i in range(len(coord)):
        sym = [atype_map[int(x)] for x in atype]  # Map atomic indices to symbols

        # Count number of each atom type
        totals = {atype_symbol: sym.count(atype_symbol) for atype_symbol in atype_map}

        # Build the formula string dynamically
        formula = ''.join([f'{symbol}{totals[symbol]}' for symbol in atype_map if totals[symbol] > 0])

        # Create the Atoms object
        atm = Atoms(formula)

        # Set positions
        if not swap:
            atm.set_positions(coord[i])
        else:  # Swapping positions
            swapped_positions = []
            for symbol in atype_map:
                indices = [j for j, s in enumerate(sym) if s == symbol]
                swapped_positions.append(coord[i][indices])
            swapped_positions = np.vstack(swapped_positions[::-1])  # Reverse the order of atom types
            atm.set_positions(swapped_positions)

        # Set cell and periodic boundary conditions
        atm.set_cell(box[i])
        atm.set_pbc([True, True, True])

        # Append the Atoms object to the list
        atm_list.append(atm)

    return atm_list

def get_train_data_path(path):
    with open(path, 'r+') as f:
        data = json.load(f)
        train_d = data["training"]["training_data"]["systems"]
    return train_d

def compute_convex_hull_volume(embedding):
    hull = ConvexHull(embedding)
    return hull.volume, hull

def compute_emd(embedding1, embedding2):
    emd_x = wasserstein_distance(embedding1[:, 0], embedding2[:, 0])  # EMD for dimension 1
    emd_y = wasserstein_distance(embedding1[:, 1], embedding2[:, 1])  # EMD for dimension 2
    return (emd_x + emd_y) / 2  # Average EMD across both dimensions

def plot_convex_hulls_together(embeddings, generation_labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    volumes = []
    emd_values = []
    cmap = cm.get_cmap('hsv',len(generation_labels))
    colors = [cmap(i) for i in range(len(generation_labels))]#['red', 'purple', 'cyan', 'green', 'blue']  # Colors for different generations

    for i, (embedding, label) in enumerate(zip(embeddings, generation_labels)):
        volume, hull = compute_convex_hull_volume(embedding)
        jitter_strength = 0.02
        jittered_embedding = embedding + np.random.normal(scale=jitter_strength, size=embedding.shape)
        
        ax1.scatter(jittered_embedding[:, 0], jittered_embedding[:, 1], color=colors[i], alpha=0.5, s=10, label=f'Generation {label}')
        for simplex in hull.simplices:
            ax1.plot(embedding[simplex, 0], embedding[simplex, 1], color=colors[i], lw=1.5)
        
        volumes.append(volume)

    #print(volumes)
    ax1.set_xlabel('UMAP Dimension 1',size=20)
    ax1.set_ylabel('UMAP Dimension 2',size=20)
    ax1.tick_params(direction='in', right=True, top=True)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.legend(prop = {'size':12}, frameon=False)

    ax2.bar(generation_labels, volumes, color=colors, alpha=0.7,ec="black", lw=1.75)
    
    ax2.tick_params(direction='in', right=True, top=True)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_xlabel('Generation', size=20)
    ax2.set_ylabel('Volume', size=20)
    ax2.set_xticks(generation_labels)

    for i in range(1, len(embeddings)):
        emd_value = compute_emd(embeddings[i-1], embeddings[i])
        emd_values.append(emd_value)
    emd_values = [0] + emd_values
    
    ax3.bar(generation_labels, emd_values, color=colors, alpha=0.7,ec="black", lw=1.75)
    
    ax3.set_xlabel('Generation', fontsize=20)
    ax3.set_ylabel('EMD', fontsize=20)
    ax3.set_xticks(generation_labels)

    ax3.tick_params(direction='in', right=True, top=True)
    ax3.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.show()

    return volumes

train_data_list=[]
for i in range(15):
    train_i = f'{main_path}/gen-{i}/train/dp0/input.json'
    if i > 0:
        temp=get_train_data_path(train_i)
        append=list(set(temp)-set(train_data_list[i-1]))
    else:
        append=get_train_data_path(train_i)
    train_data_list+=[append]
    
# Load your ASE Atoms objects (update paths accordingly)

if 'UMAP_embeddings' not in os.listdir():
    os.mkdir('UMAP_embeddings')

umap_path = 'UMAP_embeddings'

for i, gen_train in enumerate(train_data_list):
    print(f'gen-{i}--------------')
    for j, dj in enumerate(gen_train):
        if f'gen-{i}-embd-{str(j).zfill(2)}.npy' not in os.listdir('UMAP_embeddings'):
            print(f'data-{j}')
            try:
                atoms_list = create_atoms_from_npy(dj) # Replace with your actual file paths
                species = set([atom.symbol for atoms in atoms_list for atom in atoms])     # Get unique atomic species

                # Get SOAP descriptors
                soap_features = get_soap_descriptor(atoms_list, species=species)

                # Standardize the features
                scaler = StandardScaler()
                soap_features_scaled = scaler.fit_transform(soap_features)

                # Apply UMAP to reduce dimensions
                umap_model = umap.UMAP(n_neighbors=15, min_dist=0.0, metric='euclidean')
                try:
                    embedding = umap_model.fit_transform(soap_features_scaled)
                    np.save(f'UMAP_embeddings/gen-{i}-embd-{str(j).zfill(2)}.npy',embedding)
                except:
                    print(f'Issue with embedding')
            except:
                print('Error with loading data')

# Load embeddings

embeddings = []
for i in range(total_generation):
    gen_i = sorted([x for x in os.listdir(f'UMAP_embeddings/') if f'gen-{i}' in x])
    
    for j, em_path in enumerate(gen_i):
        if j == 0:
            embedding=np.load(f'UMAP_embeddings/{em_path}')
        else:
            embedding = np.vstack((embedding, np.load(f'UMAP_embeddings/{em_path}')))
    embeddings += [embedding]
    volumes = plot_convex_hulls_together(embeddings, np.arange(0,15))
    print("Convex Hull Volumes:", volumes)
