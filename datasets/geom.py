import os
import json
import pdb
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from os.path import join
from itertools import repeat
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data, InMemoryDataset

allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
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


def mol_to_graph_data_obj_simple_3D(mol, type_view):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '../datasets/GEOM/rdkit_folder'
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        #==== Path should match ====
        if sub_dic.get('pickle_path', '') == '':
            continue

        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class GEOMDataset(InMemoryDataset):

    def __init__(self, root, n_mol=50000, n_conf=1, n_upper=1000, transform=None, 
                 seed=42, pre_transform=None, pre_filter=None, empty=False):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(GEOMDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]
        return data

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
        drugs_summary = list(drugs_summary.items())
        print('# of SMILES: {}'.format(len(drugs_summary)))

        random.seed(self.seed)
        random.shuffle(drugs_summary)
        mol_idx, idx, notfound = 0, 0, 0
        for smiles, sub_dic in tqdm(drugs_summary):
            # === Path should match ===
            if sub_dic.get('pickle_path', '') == '':
                # === No pickle path available ===
                notfound += 1
                continue

            # === The conformers are pre-produced ===
            mol_path = join(dir_name, sub_dic['pickle_path'])
            with open(mol_path, 'rb') as f:
                mol_dic = pickle.load(f)
                conformer_list = mol_dic['conformers']

                # === count should match ===
                conf_n = len(conformer_list)
                if conf_n < self.n_conf or conf_n > self.n_upper:
                    notfound += 1
                    continue

                # === SMILES should match ===
                #  export prefix=https://github.com/learningmatter-mit/geom
                #  Ref: ${prefix}/issues/4#issuecomment-853486681
                #  Ref: ${prefix}/blob/master/tutorials/02_loading_rdkit_mols.ipynb
                conf_list = [
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(
                            Chem.MolToSmiles(rd_mol['rd_mol'])))
                    for rd_mol in conformer_list[:self.n_conf]]

                conf_list_raw = [
                    Chem.MolToSmiles(rd_mol['rd_mol'])
                    for rd_mol in conformer_list[:self.n_conf]]
                # check that they're all the same
                same_confs = len(list(set(conf_list))) == 1
                same_confs_raw = len(list(set(conf_list_raw))) == 1
                if not same_confs:
                    if same_confs_raw is True:
                        print("Interesting")
                    notfound += 1
                    continue
                
                for conformer_dict in conformer_list[:self.n_conf]:
                    rdkit_mol = conformer_dict['rd_mol']
                    data = mol_to_graph_data_obj_simple_3D(rdkit_mol, self.type_view)
                    data.id = torch.tensor([idx])
                    data.mol_id = torch.tensor([mol_idx])
                    data_smiles_list.append(smiles)
                    data_list.append(data)
                    idx += 1



            # select the first n_mol molecules
            if mol_idx + 1 >= self.n_mol:
                break
            if same_confs:
                mol_idx += 1
        

        print('mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}'.format(
            mol_idx, len(data_smiles_list), len(set(data_smiles_list))))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False) 

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers/fingerprint have been processed" % idx)
        return


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, default=50000, help='number of unique smiles/molecules')
    parser.add_argument('--n_conf', type=int, default=5, help='number of conformers of each molecule')
    parser.add_argument('--n_upper', type=int, default=1000, help='upper bound for number of conformers')
    args = parser.parse_args()

    if args.sum:
        sum_list = summarise()
        with open('../datasets/summarise.json', 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol, n_conf, n_upper = args.n_mol, args.n_conf, args.n_upper
        root = '../datasets/GEOM/processed/GEOM_nmol%d_nconf%d/' % (n_mol, n_conf)

        GEOMDataset(root=root, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)


