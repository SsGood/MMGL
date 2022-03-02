import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import matplotlib
import pandas as pd
from collections import Counter

import numpy as np
import scipy.io as scio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='TADPOLE')

args = parser.parse_args()

    
data_name = args.datname
attn_map_list = np.load('./attn/attn_map_{}.npy'.format(data_name), allow_pickle=True)
label = np.load('./graph/{}_weighted-cosine_graph.npz'.format(data_name))['label']
#em = np.load('./graph/{}_weighted-cosine_graph.npz'.format(data_name))['fused']
labels   = label
idx_del  = np.where(labels==-1)[0]
idx      = np.arange(labels.shape[0])
idx      = np.delete(idx,idx_del)

index_0 = []
index_1 = []
index_2 = []
for i in range(len(labels)):
    if labels[i] == 0:
        index_0.append(i)
    elif labels[i] == 1:
        index_1.append(i)
    else:
        index_2.append(i)


layer = 2 if data_name == 'TADPOLE' else 1
fig = plt.figure(figsize=(20,7))
for i in range(layer+1):
    plt.subplot(1,layer+1, i+1)
    x = np.load('./result/{}_{}.npy'.format(data_name, i))
    palette = np.array(sns.color_palette("hls", 5))
    if data_name == 'ABIDE':
        plt.scatter(x[index_0,0], x[index_0,1], lw=0, s=40, color = palette[0], label = 'ASD')
        plt.scatter(x[index_1,0], x[index_1,1], lw=0, s=40, color = palette[2], label = 'NC')
    #plt.scatter(x[index_2,0], x[index_2,1], lw=0, s=40, color = palette[4], label = 'AD')
    elif data_name == 'TADPOLE':
        plt.scatter(x[index_0,0], x[index_0,1], lw=0, s=40, color = palette[0], label = 'NC')
        plt.scatter(x[index_1,0], x[index_1,1], lw=0, s=40, color = palette[2], label = 'MCI')
        plt.scatter(x[index_2,0], x[index_2,1], lw=0, s=40, color = palette[4], label = 'AD')
    plt.title('Attention_Vis_{}'.format(i), y=-0.15, fontsize=18)