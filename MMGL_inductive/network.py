import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from layers import *
import dgl
from dgl.nn import GraphConv


class VLTransformer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            #x = x.transpose(1, 0)#nn.multi_head_attn
            #x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            #x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
           
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
    
    
class VLTransformer_Gate(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_Gate, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        
        self.FGLayer = FusionGate(self.modal_num)    
        self.Outputlayer = OutputLayer(self.d_v * self.n_head, self.d_v * self.n_head, self.n_class)
        
    def forward(self, x):
        bs = x.size(0)
        x, attn = self.InputLayer(x)
        for i in range(self.n_layer):
            x, attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            x = self.FeedForward[i](x)
        x, norm = self.FGLayer(x)    
        x = x.sum(-2)/norm
        output, hidden = self.Outputlayer(x)
        return output, hidden

    
class GraphLearn(nn.Module):
    def __init__(self, input_dim, th, mode = 'Sigmoid-like'):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th
        
    def forward(self, x, test = False):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)
        
        if self.mode == "Sigmoid-like":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = diff.pow(2).sum(dim=2).pow(1/2)
            diff = (diff + self.threshold) * self.t
            output = 1 - torch.sigmoid(diff)
            
        elif self.mode == "adaptive-learning":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim = 1)
        
        elif self.mode == 'weighted-cosine':
            th = self.th
            x = self.p(x)
            x_norm = F.normalize(x,dim=-1)
            #x_norm_repeat = x_norm.repeat_interleave(num, dim = 0).view(num, num, feat_dim).detach()
            #cos_sim = torch.mul(x_norm.unsqueeze(0), x_norm_repeat)
            #score = cos_sim.sum(dim = -1)
            score = torch.matmul(x_norm, x_norm.T)
            mask = (score > th).detach().float()
            markoff_value = 0
            output = score * mask + markoff_value * (1 - mask)
        
        if test == True:
            mask = torch.ones(output.shape)
            mask[-bs:][-bs:] = torch.zeros((bs, bs))
            output = output * mask
        return output
    
    
    
    
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = dropout

    def forward(self, blocks, in_feat):
        h = F.relu(self.conv1(blocks[0], in_feat))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.conv2(blocks[1], h))
        return F.log_softmax(h, dim=1)    