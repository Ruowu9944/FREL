import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

def class_stats(data, num_class):
    # pdb.set_trace()
    y = data.y.view((data.id.shape[0], num_class))
    is_valid = y ** 2 > 0
    y = (y + 1) / 2
    y = torch.where(is_valid, y, torch.zeros(y.shape).to(y.dtype))
    classes = torch.zeros(num_class)
    for i in range(num_class):
        classes[i] += y[:, i].sum()
    
    return classes


import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)




def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, args.T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc

# def do_cosine_CL(X, Y, args):
#     if args.normalize:
#         X = F.normalize(X, dim=-1)
#         Y = F.normalize(Y, dim=-1)
    
    
#     criterion = nn.CrossEntropyLoss()
#     B = X.size()[0]
#     distance = torch.nn.CosineSimilarity(dim=-1)
#     logits = distance(X, Y)
#     logits = torch.div(logits, args.T)
#     labels = torch.arange(B).long().to(logits.device)  # B*1

#     CL_loss = criterion(logits, labels)
#     pred = logits.argmax(dim=1, keepdim=False)
#     CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

#     return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2

def do_CL_info(c, X, Y, args):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = int(X.size()[0] / c)
    batch = torch.arange(B)
    idx = batch.repeat_interleave(c).long().to(X.device)

    logits = torch.mm(X, Y.transpose(1, 0))  # Bc*Bc
    logits = torch.div(logits, args.T)

    multi_logits = torch.zeros(B, B).to(X.device)
    tmp = torch.zeros(B*c, B).to(X.device)
    tmp = scatter_add(logits, idx, out=tmp, dim=1)
    multi_logits = scatter_add(tmp, idx, out=multi_logits, dim=0)

    labels = torch.arange(B).long().to(multi_logits.device)  # B*1

    CL_loss = criterion(multi_logits, labels)
    pred = multi_logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / (B)
    
    return CL_loss, CL_acc

def dual_CL_info(c, X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL_info(c, X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL_info(c, Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2

