# Enhancing Drug Property Prediction with Dual-Channel Transfer Learning based on Molecular Fragment

Authors: Yue Wu, Xinran Ni, Zhihao Wang, Weike Feng

This repository provides the source code for the paper **Enhancing Drug Property Prediction with Dual-Channel Transfer Learning based on Molecular Fragment.**

## Environments

```markdown
numpy             1.21.2
scikit-learn      1.0.2
pandas            1.3.4
python            3.7.11
torch             1.10.2+cu113
torch-geometric   2.0.3
transformers      4.17.0
rdkit             2020.09.1.0
ase               3.22.1
descriptastorus   2.3.0.5
ogb               1.3.3
```

## Dataset Preparation

### Datasets

For dataset download, please follow the instruction from [GraphMVP](https://github.com/chao1224/GraphMVP/tree/main/datasets) .

```bash
cd datasets
python molecule_preparation.py
```

## Experiments

To pre-train classification model, run the following code:

```bash
cd src
python pretrain_cls.py --dropout_ratio=0
```

To pre-train regression model, run the following code:

```bash
python pretrain_reg.py --dropout_ratio=0
```

To fine-tune classification model, run the following code:

```bash
python finetune_cls.py --dropout_ratio=0.5 --dataset=bace
```

To pre-train regression model, run the following code:

```bash
python finetune_reg.py --dropout_ratio=0.5 --dataset=esol
```