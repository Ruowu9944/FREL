from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
import nni
import os
from nni.utils import merge_parameter
from functools import partial
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from datasets.motif_dataset import MotifDataset

from config import args
# from datasets.pcqm4mv2 import PygPCQM4Mv2Dataset
from datasets.geom import GEOMDataset
from datasets.dataloader import DataLoaderMaskingPred
from models.model2D import GNNDecoder, GNN, Multi_Motif_Fusion
from utils import NTXentLoss, dual_CL


def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]\n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            path = args.output_model_dir 
            if not os.path.exists(path):  os.makedirs(path)
            torch.save(model.state_dict(), path + 'Mask_{}_T{}.pth'.format(args.mask_rate, args.T))

    return

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def recons_loss(batch, model, dec_atom):
    criterion = partial(sce_loss, alpha=1)
    node_repr = model(batch)
    mask_idx = batch.masked_atom_indices
    
    node_label = batch.node_attr_label
    pred_node = dec_atom(node_repr, batch.edge_index, batch.edge_attr, mask_idx)
    loss = 0
    loss += criterion(node_label, pred_node[mask_idx])

    return loss

def contrast_loss(batch_motif, batch, model, att_model,epoch):
    '''
    node_repr: [N_atoms, h_dim]
    motif_repr: [N_motifs_max, h_dim]
    aggr_repr: [N_mol, h_dim]
    molecule_repr: [N_mol, h_dim]
    ''' 
    motif_repr = global_mean_pool(model(batch_motif), batch_motif.motif_batch)

    # ==== Get the motif of the same molecule ====
    batch_idx = batch_motif.motif_batch[:-1]-batch_motif.motif_batch[1:]
    batch_idx = F.pad(batch_idx, pad=(1,0), mode='constant', value=-1)
    new_idx = batch_motif.batch[batch_idx==-1]
    out, mask = to_dense_batch(motif_repr, new_idx)     # out: [n_mol, n_motif_max, hid_dim]
    motif_repr, coeff = att_model(out, mask)

    molecule_repr = global_mean_pool(model(batch), batch=batch.batch)

    loss, _ = dual_CL(motif_repr, molecule_repr, args)

    return loss

def train(args, model_list, device, loaders, optimizer_list, contrast=False):   
    start_time = time.time()
    model, dec_pred_atoms, att_model = model_list
    optimizer_model, optimizer_dec_pred_atoms = optimizer_list
    model.train()
    dec_pred_atoms.train()
    att_model.train()

    loss_accum = 0
    loss_mae_accum = 0
    loss_con_accum = 0
    [loader_motif, loader] = loaders 
    l = tqdm(zip(loader_motif, loader), total=len(loader))
        
    for step, (batch_motif, batch_mol) in enumerate(l): 
        if contrast: 
            batch_motif = batch_motif.to(device)    
        batch_mol = batch_mol.to(device)

        # ===== Calculate reconstruction Loss =====
        if contrast: 
            loss = recons_loss(batch_motif, model, dec_pred_atoms)
            loss_mae_accum += float(loss.cpu().item())
            con_loss = args.lamda * contrast_loss(batch_motif, batch_mol, model, att_model, epoch)
            loss_con_accum += float(con_loss.cpu().item())
            loss += con_loss
        else:
            loss = recons_loss(batch_mol, model, dec_pred_atoms)
            loss_mae_accum += float(loss.cpu().item())
        

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()
        loss_accum += float(loss.cpu().item())

    global optimal_loss
    loss_accum /= len(loader)
    loss_mae_accum /= len(loader)
    loss_con_accum /= len(loader)

    temp_loss = loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print('Total Loss: {:.5f}\tMAE Loss: {:.5f}\t\tContrastive Loss: {:.5f}\tTime: {:.5f}'.format(
        loss_accum, loss_mae_accum, loss_con_accum, time.time() - start_time))
    
    nni.report_intermediate_result(loss_accum)

    return 

if __name__ == '__main__':
    print('===start===\n')
    params = nni.get_next_parameter()
    args = merge_parameter(args, params)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)
    
    data_root =  args.input_data_dir
    dataset = GEOMDataset(data_root)
    dataset_motif = MotifDataset(data_root)

    torch.manual_seed(args.seed)
    dataset = dataset.shuffle()
    torch.manual_seed(args.seed)
    dataset_motif = dataset_motif.shuffle()
    
    torch.manual_seed(args.seed)
    loader_motif = DataLoaderMaskingPred(dataset_motif, batch_size=args.batch_size, shuffle=True, mask_rate=args.mask_rate,
                                   num_workers = args.num_workers)
    torch.manual_seed(args.seed)                               
    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, mask_rate=0,
                                    num_workers = args.num_workers)
    NUM_NODE_ATTR = 119
    
    # === 2D Encoder Building ===  
    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim).to(device)
    att_model = Multi_Motif_Fusion(args.emb_dim).to(device)
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.net2d).to(device)

    
    model_list = [model, atom_pred_decoder, att_model]
    model_param_group = [] 
    
    model_param_group.append({'params': model.parameters()})
    model_param_group.append({'params': att_model.parameters()})
    optimizer_model = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms]
    optimal_loss = 1e10
    

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, model_list, device, [loader_motif,loader], optimizer_list, contrast=True)
    
