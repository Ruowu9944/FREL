import argparse

parser = argparse.ArgumentParser()

# Seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=2)
parser.add_argument('--device', type=int, default=0)

# Dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='../datasets/GEOM/processed/')
parser.add_argument('--output_model_dir', type=str, default='./model_saved/')
parser.add_argument('--model_file', type=str, default='regression_T_0.1_contrast_0.pth')
parser.add_argument('--dataset', type=str, default='esol')
parser.add_argument('--num_workers', type=int, default=8)

# Training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--warm_up_steps', type=int, default=10)
parser.add_argument('--mask_rate', type=float, default=0.3)
parser.add_argument('--mask_edge', action="store_true")     #default=false

# GNN for Molecules
parser.add_argument('--net2d', type=str, default='GIN')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5) 
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')

# Modelality
parser.add_argument('--modality', type=str, default='graph',
                    choices=['smiles', 'graph'])

# GraphGPS Config
parser.add_argument("--model", type=str, default="gt", choices=["gt", "bigcn"])
parser.add_argument("--gt_layer_type", type=str, default="GIN+Transformer", help="")
parser.add_argument("--gt_layers", type=int, default=4, help="num of gt_layers")
parser.add_argument("--gt_n_heads", type=int, default=8)
parser.add_argument("--posenc_EquivStableLapPE_enable", action="store_true", help="")       # default=False
parser.add_argument("--gt_dropout", type=float, default=0.2, help="")
parser.add_argument("--gt_attn_dropout", type=float, default=0.2, help="")
parser.add_argument("--gt_layer_norm", action="store_true")                       # default=false
parser.add_argument("--gt_batch_norm", action="store_false")                       # default=true
parser.add_argument("--gt_bigbird", default=None)

# Loss Hyperparameter 
parser.add_argument('--loss_fn', type=str, default='sce')
parser.add_argument('--alpha', type=float, default=1, help='Coefficient of MAE loss')
parser.add_argument('--beta', type=float, default=1, help='Coefficient of contrastive loss')
parser.add_argument('--lamda', type=float, default=0.4)

# Contrastive CL
parser.add_argument('--T', type=float, default=0.07)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=True) 

# Learning rate for different networks
parser.add_argument('--lr_decay_step_size', type=int, default=15)
parser.add_argument('--lr_decay_factor', type=int, default=0.5)


# If print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

args = parser.parse_args()
print('arguments\t', args)

