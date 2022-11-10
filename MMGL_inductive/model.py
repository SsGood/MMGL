import os
import random
import sys

import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import roc_auc_score
import matplotlib.cm
import networkx as nx 
from sklearn.metrics import confusion_matrix
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
import scipy.sparse as sp

from network import *
from utils import *


class disease_dataset(Dataset):
    def __init__(self, feat, label, ind):
        super(Dataset, self).__init__()
        
        self.feat  = feat[ind]
        self.label = label[ind]
        
    def __getitem__(self, index):
        return self.feat[index], self.label[index]
    
    def __len__(self):
        return np.shape(self.feat)[0]


class EvalHelper:
    def __init__(self, input_data_dims, feat, label, hyperpm, train_index, test_index):
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
        #feat = torch.from_numpy(feat).float().to(dev)
        #label = torch.from_numpy(label).long().to(dev)
        self.dev = dev
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.feat = feat
        self.targ = label
        self.best_acc = 0
        self.best_acc_2 = 0
        self.MF_sav = tempfile.TemporaryFile()
        self.GCMP_sav = tempfile.TemporaryFile()
        num = train_index.shape[0]
        self.trn_idx = train_index
        self.val_idx = np.array(test_index)
        self.tst_idx = np.array(test_index)
        
        self.trn_dataset = disease_dataset(feat, label, self.trn_idx)
        self.val_dataset = disease_dataset(feat, label, self.val_idx)
        self.tst_dataset = disease_dataset(feat, label, self.tst_idx)
        
        self.trn_loader = DataLoader(self.trn_dataset, batch_size = 256, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = 256, shuffle=False)
        self.tst_loader = DataLoader(self.tst_dataset, batch_size = 256, shuffle=False)
        
        trn_label = label[self.trn_idx]
        
        counter = Counter(trn_label)
        print(counter)
        weight = len(trn_label)/np.array(list(counter.values()))/self.n_class
        
        self.out_dim = self.d_v * self.n_head + self.modal_num**2
        self.weight = torch.from_numpy(weight).float().to(dev)
        if self.MF_mode == 'sum':
            self.ModalFusion = VLTransformer_Gate(input_data_dims, hyperpm).to(dev)
        else:
            self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(dev)
        self.GraphConstruct = GraphLearn(self.out_dim, th = self.th, mode = self.GC_mode).to(dev)
        
        if self.MP_mode == 'GCN':
            self.MessagePassing = GCN(self.out_dim, self.out_dim // 2, self.n_class, self.dropout).to(dev)
        elif self.MP_mode == 'GAT':
            self.MessagePassing = GAT(self.out_dim, self.out_dim // 2, self.n_class, self.dropout, self.alpha, nheads = 2).to(dev)
        
        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_GC = optim.Adam(self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_MP = optim.Adam(self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        
        self.ModalFusion.apply(my_weight_init)
        
    def forward(self, dataloader, dev):
        loss, pred, targ = 0, [], []
        num_batches, size = len(dataloader), len(dataloader.dataset)
        
        for i, (feat, label) in enumerate(dataloader):
            feat, label = feat.float().to(dev), label.long().to(dev)
            prob, hidden, attn = self.ModalFusion(feat)
            cls_loss = F.nll_loss(prob, label)
            cls_loss.backward()
            
            pred.extend(prob.argmax(1).cpu().numpy())
            targ.extend(label.cpu().numpy())
            loss += cls_loss.item()
        
        correct = (np.array(pred) == np.array(targ)).sum()
        auc = roc_auc_score(one_hot(targ, self.n_class).numpy(), one_hot(pred, self.n_class).numpy())
        return loss/num_batches, correct/size, auc
    
    def forward_MF(self, dev, test=False):
        loss, pred, targ = 0, [], []
        dataloader = self.trn_loader
        num_batches, size = len(dataloader), len(dataloader.dataset)
        hidden_matrix = torch.empty((0)).to(dev)
        for i, (feat, label) in enumerate(dataloader):
            feat, label = feat.float().to(dev), label.long().to(dev)
            prob, hidden, attn = self.ModalFusion(feat)
            cls_loss = F.nll_loss(prob, label)
            #cls_loss.backward()
            
            pred.extend(prob.argmax(1).cpu().numpy())
            targ.extend(label.cpu().numpy())
            loss += cls_loss#.item()
            
            hidden_matrix = torch.cat([hidden_matrix,hidden],0)
            
        trn_acc = (np.array(pred) == np.array(targ)).sum()/size
        trn_auc = roc_auc_score(one_hot(targ, self.n_class).numpy(), one_hot(pred, self.n_class).numpy())
        trn_loss = loss.item()/num_batches
        
        adj = self.GraphConstruct(hidden_matrix)
        graph_loss = GraphConstructLoss(hidden_matrix, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
        adj = {'adj':adj, 'label':np.array(targ)}
        loss += graph_loss
        loss.backward()

    
        val_acc, val_auc, val_loss = None, None, None
        
        if test != False:
            loss, pred, tst_targ = 0, [], []
            if test == 'val':
                val_loader = self.val_loader
            else:
                val_loader = self.tst_loader
            num_batches, size = len(val_loader), len(val_loader.dataset)
            
            for i, (feat, label) in enumerate(val_loader):
                feat, label = feat.float().to(dev), label.long().to(dev)
                prob, hidden, attn = self.ModalFusion(feat)
                cls_loss = F.nll_loss(prob, label)
                
                pred.extend(prob.argmax(1).cpu().numpy())
                tst_targ.extend(label.cpu().numpy())
                loss += cls_loss.item()
                
                hidden_matrix = torch.cat([hidden_matrix,hidden],0)
            val_acc = (np.array(pred) == np.array(tst_targ)).sum()/size
            val_auc = roc_auc_score(one_hot(tst_targ, self.n_class).numpy(), one_hot(pred, self.n_class).numpy())
            val_loss = loss/num_batches
            
            targ.extend(tst_targ)
            adj = self.GraphConstruct(hidden_matrix.detach())
            adj = {'adj':adj, 'label':np.array(targ)}
            
        return (trn_acc, trn_auc, trn_loss, graph_loss.item()), (val_acc, val_auc, val_loss), hidden_matrix, adj


    def forward_graph(self, hidden_matrix, adj_dict, dev, test=False):
        loss, pred, targ = 0, [], []
        adj, label = adj_dict['adj'], adj_dict['label']
        np.save('adj.npy', adj.cpu().detach().numpy())
        normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
        sp_adj = sp.coo_matrix(normalized_adj.cpu().detach().numpy())
        G = dgl.from_scipy(sp_adj).to(dev)
        G.ndata['feat'] = hidden_matrix
        G.ndata['label'] = torch.tensor(label).to(dev)
        G.edata['w'] = torch.tensor(sp_adj.data).to(dev)
        
        if test != False:
            idx = list(range(G.num_nodes()))[-len(self.tst_idx):]
        else:
            idx = list(range(G.num_nodes()))
        sampler = MultiLayerNeighborSampler([5,10])
        node_loader = NodeDataLoader(G,
                                    torch.tensor(idx).to(dev),
                                    sampler,
                                    batch_size=1000,
                                    shuffle=False,
                                    drop_last=False)
        num_batches, size = len(node_loader), len(node_loader.dataset)
        for input_nodes, output_nodes, blocks in node_loader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_feat = blocks[0].srcdata['feat']
            label = blocks[-1].dstdata['label']
            prob = self.MessagePassing(blocks, input_feat)
            cls_loss = F.nll_loss(prob, label)
            #print('---------------------------')
            #print(cls_loss)
            cls_loss.backward()
            pred.extend(prob.argmax(1).cpu().numpy())
            targ.extend(label.cpu().numpy())
            loss += cls_loss.item()
            
        
        acc = (np.array(pred) == np.array(targ)).sum()/len(idx)
        auc = roc_auc_score(one_hot(targ, self.n_class).numpy(), one_hot(pred, self.n_class).numpy())
        loss = loss/num_batches
        
        return acc, auc, loss
        
        
            
        
    def run_epoch(self, mode, end = ''):
        dev = self.dev
        if mode == 'pre-train':
            self.ModalFusion.train()
            self.GraphConstruct.eval()
            self.MessagePassing.eval()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            trn_loss, trn_acc, trn_auc = self.forward(self.trn_loader, dev)
            self.optimizer_MF.step()
            
            print('trn-loss-MF: %.4f' % trn_loss, end=' ')
            #print('trn-acc-MF:  %.4f' % trn_acc, end=' ')
        
        if mode == 'simple-2':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            
            self.optimizer_MF.zero_grad()
            (trn_acc, trn_auc, trn_loss, GC_loss), _, hidden_matrix, _ = self.forward_MF(dev)
            self.optimizer_MF.step()
            #print('trn-loss-MF: %.4f ' % trn_loss, end=' ')
            
            self.MessagePassing.train()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            _, _, hidden_matrix, adj = self.forward_MF(dev)
            acc, auc, loss = self.forward_graph(hidden_matrix.detach(), adj, dev)
            self.optimizer_MF.step()
            self.optimizer_GC.step()
            self.optimizer_MP.step()
            print('trn-loss: %.4f trn-loss-GC: %.4f' % (loss, GC_loss), end=' ')
            #print('acc: ', acc, 'auc: ', auc)
            

    
    def print_trn_acc(self, mode = 'pre-train'):
        print('trn-', end='')
        trn_acc, trn_auc = self._print_acc(self.trn_loader, mode, tst = False, end=' val-')
        val_acc, val_auc = self._print_acc(self.val_loader, mode, tst = 'val')
        #print('pred:',pred_val[:10], 'targ:',targ_val[:10])
        return trn_acc, val_acc

    def print_tst_acc(self, mode = 'pre-train'):
        print('tst-', end='')
        tst_acc, tst_auc = self._print_acc(self.tst_loader, mode, tst = True)
        #conf_mat = confusion_matrix(targ_tst.detach().cpu().numpy(), pred_tst.detach().cpu().numpy())
        return tst_acc, tst_auc
    

    def _print_acc(self, eval_idx, mode, tst = False, end='\n'):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        if mode == 'pre-train':
            loss, acc, auc = self.forward(eval_idx, self.dev)
        elif mode == 'simple-2':
            _, _, hidden_matrix, adj = self.forward_MF(self.dev, test = tst)
            acc, auc, loss = self.forward_graph(hidden_matrix.detach(), adj, self.dev, test = tst)
            
            
        print('auc: %.4f  acc: %.4f' % (auc, acc), end=end)
        return acc, auc
        
        
        