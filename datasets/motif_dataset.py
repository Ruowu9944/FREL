import os
import os.path as osp
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS, Recap
from rdkit import Chem
from rdkit import Chem
import json
from os.path import join
import random

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from functools import reduce


allowable_features = {
    'possible_atomic_num_list':       list(range(0, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e) if l.index(e)==0 else l.index(e)-1
    except:
        return len(l) - 1

def smiles_to_graph(s):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    mol = AllChem.MolFromSmiles(s)
    atom_features_list = []
    for atom in mol.GetAtoms():
        # if atom.GetAtomicNum()==0:pdb.set_trace()
        atom_feature = [safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum())] + \
                       [safe_index(allowable_features['possible_chirality_list'], atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [safe_index(allowable_features['possible_bonds'], bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_substructure(mol=None, smile=None, decomp='brics'):
    assert mol is not None or smile is not None,\
        'need at least one info of mol'
    assert decomp in ['brics', 'recap'], 'Invalid decomposition method'
    if mol is None:
        mol = Chem.MolFromSmiles(smile)

    if decomp == 'brics':
        substructures = BRICS.BRICSDecompose(mol)
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())
    return substructures


def substructure_batch(smile):
    sub_structures = [get_substructure(smile=smile)]
    return graph_from_substructure(sub_structures)


def graph_from_substructure(subs):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_graph = [smiles_to_graph(x) for x in sub_struct_list]
    return sub_graph

class MotifDataset(InMemoryDataset):
    def __init__(self, root, n_mol=50000, n_conf=1, n_upper=1000, transform=None, 
                 seed=42, pre_transform=None, pre_filter=None, empty=False):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter
        super(MotifDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data_smiles_list = []

        dir_name = '../datasets/GEOM/rdkit_folder'
        drugs_file = '{}/summary_drugs.json'.format(dir_name)
        with open(drugs_file, 'r') as f:
            drugs_summary = json.load(f)
        smiles_list = list(drugs_summary.items())
        print('# of SMILES: {}'.format(len(drugs_summary)))

        random.seed(self.seed)
        random.shuffle(drugs_summary)

        print('Converting SMILES strings into substructures...')
        data_list = []
        # train index is the first 3378606        
        for smile, _ in tqdm(smiles_list):
            motif_idx = 0   # ======= warning ======
            motifs = substructure_batch(smile)
            
            edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
            for idx, graph in enumerate(motifs):
                edge_idxes.append(graph.edge_index + lstnode)
                edge_feats.append(graph.edge_attr)
                node_feats.append(graph.x)
                lstnode += graph.x.shape[0]
                batch.append(torch.full((graph.x.shape[0], ), motif_idx, dtype=torch.long))
                motif_idx += 1

            result = {
                'edge_index': torch.cat(edge_idxes, axis=-1),
                'edge_attr': torch.cat(edge_feats, axis=0),
                'x': torch.cat(node_feats, axis=0),
                'motif_batch': torch.cat(batch, axis=0)
            }
            data = Data(**result)
            data_list.append(data)

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        # Save the SMILES string to csv format
        print('len of smiles: {}\tlen of set(smiles): {}'.format(
            len(smiles_list), len(set(smiles_list))))
        data_smiles_series = pd.Series(smiles_list)
        saver_path = osp.join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

import pdb
if __name__ == '__main__':
    from torch_geometric.nn import global_add_pool
    dataset = MotifDataset()
    
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    