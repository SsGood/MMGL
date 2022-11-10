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
import matplotlib.cm
import networkx as nx 

from network import *


def real2col(x):
    assert 0.0 <= x <= 1.0
    r, g, b, a = matplotlib.cm.gist_ncar(x)
    return '%d,%d,%d' % (r * 255, g * 255, b * 255)


def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]


def GraphConstructLoss(feat, adj, theta_smooth, theta_degree, theta_sparsity):
    # Graph regularization
    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    L = torch.diagflat(torch.sum(adj, -1)) - adj
    vec_one = torch.ones(adj.size(-1)).to(dev)
    
    
    smoothess_penalty = torch.trace(torch.mm(feat.T, torch.mm(L, feat))) / int(np.prod(adj.shape))
    degree_penalty = torch.mm(vec_one.unsqueeze(0), torch.log(torch.mm(adj, vec_one.unsqueeze(-1)) + 1e-5)).squeeze() / adj.shape[-1]
    sparsity_penalty = torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))
    
    return theta_smooth * smoothess_penalty - theta_degree * degree_penalty + theta_sparsity * sparsity_penalty


def ClsLoss(output, labels, idx, weight):
    
    return F.nll_loss(output[idx], labels[idx], weight)

def ClsLoss_noweight(output, labels, idx):
    
    return F.nll_loss(output[idx], labels[idx])


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj


def RBF_kernel(x_data):
    distv = distance.pdist(x_data, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    return sparse_graph


def KNN_graph(x_data, topk=100, markoff_value=0):
    attention = RBF_kernel(x_data)
    topk = min(topk, attention.size(-1))
    knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
    weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def my_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def visualize_as_gdf(g, savfile, label, color, pos_gml=None):
    assert isinstance(g, nx.Graph)
    n = g.number_of_nodes()
    if not savfile.endswith('.gdf'):
        savfile += '.gdf'
    assert len(label) == n
    color = np.asarray(color, dtype=np.float32).copy()
    color = (color - color.min()) / (color.max() - color.min() + 1e-6)
    assert color.shape == (n,)
    if isinstance(pos_gml, str) and os.path.isfile(pos_gml):
        layout_g = nx.read_gml(pos_gml)
        layout_g = dict(layout_g.nodes)
        pos = np.zeros((n, 2), dtype=np.float64)
        for t in range(n):
            pos[t] = (layout_g[str(t)]['graphics']['x'],
                      layout_g[str(t)]['graphics']['y'])
        scale = 1
    else:
        pos = nx.random_layout(g)
        scale = 1000
    with open(savfile, 'w') as fout:
        fout.write('nodedef>name VARCHAR,label VARCHAR,'
                   'x DOUBLE,y DOUBLE,color VARCHAR\n')
        for t in range(n):
            fout.write("%d,%s,%f,%f,'%s'\n" %
                       (t, label[t], pos[t][0] * scale, pos[t][1] * scale,
                        real2col(color[t])))
        fout.write('edgedef>node1 VARCHAR,node2 VARCHAR\n')
        for (u, v) in g.edges():
            fout.write('%d,%d\n' % (u, v))
            
