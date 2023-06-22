import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
import torch
import torch.nn.functional as F

import pdb

def graph_to_nx(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx,
                   chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G

def removeSubgraph(Graph, center, percent=0.2): 
    """
    Remove a connected subgraph from the original molecule graph. 
    Args:
        1. Original graph (networkx graph)
        2. Index of the starting atom from which the removal begins (int)
        3. Percentage of the number of atoms to be removed from original graph
    Outputs:
        1. Resulting graph after subgraph removal (networkx graph)
        2. Indices of the removed atoms (list)
    """
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
        # To avoid endless loop when meet a disconnected graph
        # Like O=[C]c1ccc2c(c1)COC2.[CH2]Br
        if len(temp)==0 and len(removed) < num: break
            # print('Meet disconnected molecules!')          
    return G, removed

class ContrastiveDataset(Dataset):
    def __init__(self, pyg_data, num_atom_type, num_edge_type):
        '''
        Input: PyG type graph
        Output: Two masked graphs with different inital mask center
        '''
        super(Dataset, self).__init__()
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.pyg_data = pyg_data

    def __getitem__(self, index):
        mol = self.pyg_data[index]
        
        N = mol.x.shape[0]        

        # Sample 2 different centers to start for i and j
        if N < 2: 
            start_i, start_j = [0], [0]
        
        else: start_i, start_j = random.sample(list(range(N)), 2)

        # Construct the original molecular graph from edges (bonds)
        molGraph = graph_to_nx(mol)

        percent_i, percent_j = 0.25, 0.25
        # percent_i, percent_j = 0.2, 0.2
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)

        # ======== Add the reconstruction label =========
        node_attr_label_i = F.one_hot(mol.x[removed_i][:, 0], num_classes=self.num_atom_type).float()
        node_attr_label_j = F.one_hot(mol.x[removed_j][:, 0], num_classes=self.num_atom_type).float()
        
        # Mask the atoms in the removed list
        x_i = mol.x.clone().detach()
        for atom_idx in removed_i:
            # Change atom type to 118, and chirality to 0
            x_i[atom_idx,:] = torch.tensor([self.num_atom_type, 0])
        x_j = mol.x.clone().detach()
        for atom_idx in removed_j:
            # Change atom type to 118, and chirality to 0
            x_j[atom_idx,:] = torch.tensor([self.num_atom_type, 0])
        
        # Get the bonds that should be removed   
        keep_edge_i, keep_edge_j = [], []
        for bond_idx, (u, v) in enumerate(mol.edge_index.cpu().numpy().T):
             
            append_i, append_j = True, True
            for atom_idx in removed_i:
                if atom_idx in set((u, v)): append_i = False
            if append_i: keep_edge_i.append(bond_idx)

            for atom_idx in removed_j:
                if atom_idx in set((u, v)): append_j = False
            if append_j: keep_edge_j.append(bond_idx)
        
        edge_index_i = mol.edge_index[:, keep_edge_i]
        edge_index_j = mol.edge_index[:, keep_edge_j]
        edge_attr_i = mol.edge_attr[keep_edge_i,:]
        edge_attr_j = mol.edge_attr[keep_edge_j,:] 
        
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j) 
        mae_i = Data(x=x_i, edge_index=mol.edge_index, edge_attr=mol.edge_attr, node_label=node_attr_label_i)
        mae_j = Data(x=x_j, edge_index=mol.edge_index, edge_attr=mol.edge_attr, node_label=node_attr_label_j)
        
        return data_i, data_j, mae_i, mae_j


    def __len__(self):
        return len(self.pyg_data)