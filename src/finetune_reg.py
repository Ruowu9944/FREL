from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nni
from nni.utils import merge_parameter
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import StepLR
from splitters import scaffold_split, random_split, random_scaffold_split

from config import args
from models.model2D import MLP, GNNComplete
from datasets.regression_dataset import MoleculeDatasetComplete

meann, mad = 0, 1
def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def compute_mean_mad(values):
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

def train_general(model, device, loader, optimizer):
# def train_general(device, loader, optimizer):
    model.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        # h = global_mean_pool(model(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        pred = output_layer(h)

        y = batch.y.view(pred.shape)   
        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, device, loader, judge=None):
# def eval_general(device, loader):
    model.eval()
    output_layer.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            h = global_mean_pool(model(batch), batch.batch)
            # h = global_mean_pool(model(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            pred = output_layer(h)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    # if judge is True:
    #     np.save("true_{}_pre.npy".format(args.dataset),y_true)
    #     np.save("pred_{}_pre.npy".format(args.dataset),y_pred)
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred



if __name__ == '__main__':
    params = nni.get_next_parameter()
    args = merge_parameter(args, params)
    seed_all(args.runseed)

    # device = torch.device('cuda:' + str(args.device)) \
    #    if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = 1
    dataset_folder = '../datasets/molecule_net/'
    dataset = MoleculeDatasetComplete(dataset_folder + args.dataset, dataset=args.dataset)
    print('dataset_folder:', dataset_folder)
    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    # print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.mu.shape[0]/num_tasks)))
    # print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.mu.shape[0]/num_tasks)))

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), _ = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif args.split == 'random':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = random_split(
            dataset, null_value=0, frac_train=0.840779, frac_valid=0.076434,
            frac_test=0.082787, seed=args.seed, smiles_list=smiles_list)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # set up model
    model_param_group = []

    model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)
    output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                            out_channels=num_tasks, num_layers=1, dropout=0).to(device)
    model_param_group.append({'params': output_layer.parameters(),'lr': args.lr})

    # ==== Load the pretrained model ====
    model.load_state_dict(torch.load(args.output_model_dir + 'GraphMAE_regression.pth', map_location=device))
    # model.load_state_dict(torch.load(args.output_model_dir + args.model_file, map_location='cuda:0'))

    print('=== Model loaded ===')

    model_param_group.append({'params': model.parameters(), 'lr': args.lr})               
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    reg_criterion = torch.nn.MSELoss()


    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    train_func = train_general
    eval_func = eval_general
    print('=== Parameter set ===')
    # from plot import t_sne
    # smiles_plot = []
    # cnt = 0
    # for step, batch in enumerate(train_loader):
    #     batch.to(device)
    #     h = global_mean_pool(model(batch), batch.batch).detach().cpu().numpy()
    #     y = batch.y.detach().cpu().numpy()
    #     for i in range(len(batch.y)):
    #         if batch.y[i] > 4.2 and cnt == 0: 
    #             smiles_plot.append(train_smiles[i])
    #             cnt += 1
    #         elif batch.y[i] < -1.2 and cnt == 1:
    #             smiles_plot.append(train_smiles[i])
    #             cnt += 1
    # print(smiles_plot)

        # t_sne(h, y)

    for epoch in range(1, args.epochs + 1):
        
        # loss_acc = train_func(device, train_loader, optimizer)
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval_func(model, device, train_loader)
            # train_result, train_target, train_pred = eval_func(device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred = eval_func(model, device, val_loader)
        test_result, test_target, test_pred = eval_func(model, device, test_loader, None)
        if epoch == args.epochs:
            test_result, test_target, test_pred = eval_func(model, device, test_loader, True)
        # val_result, val_target, val_pred = eval_func(device, val_loader)
        # test_result, test_target, test_pred = eval_func(device, test_loader)

        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()
        nni.report_intermediate_result(test_result['RMSE'])

        if val_result['RMSE'] < best_val_rmse:
            best_val_rmse = val_result['RMSE']
            best_val_idx = epoch - 1

    h_list = []
    y_list = []
    cnt = 0
    for step, batch in enumerate(train_loader):
        batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        y = batch.y
        # for i in range(len(batch.y)):
        #     if batch.y[i] > 0 and cnt == 0 and batch[i].x.shape[0] > 20: 
        #         smiles_plot.append(train_smiles[i])
        #         cnt += 1
        #     elif batch.y[i] < 0 and cnt == 1:
        #         smiles_plot.append(train_smiles[i])
        #         cnt += 1
        # y = batch.y > 0
        # n = batch.y < 0
        h_list.append(h.detach().cpu())
        y_list.append(y.detach().cpu())

    h_list = torch.cat(h_list, dim=0).numpy()
    y_list = torch.cat(y_list, dim=0).numpy()
    torch.save(h_list, './h_{}_pre.pt'.format(args.dataset))
    torch.save(y_list, './y_{}_pre.pt'.format(args.dataset))
    # from plot import t_sne
    # t_sne(h_list , y_list)
    
    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
    nni.report_final_result(test_result_list[best_val_idx]['RMSE'])
