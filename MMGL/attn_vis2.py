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
em = np.load('./graph/{}_weighted-cosine_graph.npz'.format(data_name))['fused']

layer = 2 if data_name =="TADPOLE" else 1
labels = label
attn  = attn_map_list[layer]
attn_ = attn.reshape([attn.shape[0],-1])
_attn = attn.sum(axis=1)
print('attn shape ', attn_map_list.shape, ' layer ', layer, 'attn_ shape ', attn_.shape)

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
if data_name == 'ABIDE':
    attn_0 = np.mean(attn_[index_0],axis = 0)
    attn_1 = np.mean(attn_[index_1],axis = 0)

    fig = plt.figure(figsize=(20,5))
    palette = np.array(sns.color_palette("hls", 5))
    index = np.array(list(range(16)))


    fig,ax = plt.subplots(1, 2, figsize=(10,3))
    for i in range(1):
        #plt.subplot(2, 5, i+1)
        ax[0].set_ylim(0, 0.6)
        ax[0].bar(index[:4], attn_0[:4], color = palette[0], align='center', label='1st Modality')
        ax[0].bar(index[4:8], attn_0[4:8], color = palette[1], align='center', label='2nd Modality')
        ax[0].bar(index[8:12], attn_0[8:12], color = palette[2], align='center', label='3rd Modality')
        ax[0].bar(index[12:], attn_0[12:], color = palette[3], align='center', label='4th Modality')
        #ax[0].set_xlabel('({})'.format(chr(97+i)), {'fontsize': 'large'})


        #plt.subplot(2, 5, i+6)
        ax[1].set_ylim(0, 0.6)
        ax[1].bar(index[:4], attn_1[:4], color = palette[0], align='center', label='1st Modality')
        ax[1].bar(index[4:8], attn_1[4:8], color = palette[1], align='center', label='2nd Modality')
        ax[1].bar(index[8:12], attn_1[8:12], color = palette[2], align='center', label='3rd Modality')
        ax[1].bar(index[12:], attn_1[12:], color = palette[3], align='center', label='4th Modality')
        #ax[1].set_xlabel('({})'.format(chr(97+i+5)), {'fontsize': 'large'})

    ax[0].set_xlabel('(a) Autism', {'fontsize': 'x-large'})
    ax[1].set_xlabel('(b) Normal', {'fontsize': 'x-large'})

    ax[0].set_ylabel('Attention score', {'fontsize': 'large'})
    ax[1].set_ylabel('Attention score', {'fontsize': 'large'})

    plt.tight_layout()
    ax[0].legend(bbox_to_anchor=(0.6, 1.35), loc=2, borderaxespad=0, ncol=2, frameon=False, fontsize = 'x-large')
#fig.savefig('attn_avg/{}_{}_layer.png'.format(data_name,layer), dpi=600, format='png', bbox_inches = 'tight')

elif data_name == 'TADPOLE':
    attn_0 = np.mean(attn_[index_0],axis = 0)
    attn_1 = np.mean(attn_[index_1],axis = 0)
    attn_2 = np.mean(attn_[index_2],axis = 0)

    #fig = plt.figure(figsize=(100,100))
    palette = np.array(sns.color_palette("hls", 6))
    index = np.array(list(range(36)))


    fig,ax = plt.subplots(1, 3, figsize=(20,3))
    for i in range(1):
        #plt.subplot(2, 5, i+1)
        ax[1].set_ylim(0, 0.35)
        ax[1].bar(index[:6], attn_0[:6], color = palette[0], align='center', label='1st Modality')
        ax[1].bar(index[6:12], attn_0[6:12], color = palette[1], align='center', label='2nd Modality')
        ax[1].bar(index[12:18], attn_0[12:18], color = palette[2], align='center', label='3rd Modality')
        ax[1].bar(index[18:24], attn_0[18:24], color = palette[3], align='center', label='4th Modality')
        ax[1].bar(index[24:30], attn_0[24:30], color = palette[4], align='center', label='5th Modality')
        ax[1].bar(index[30:36], attn_0[30:36], color = palette[5], align='center', label='6th Modality')
        #ax[0].set_xlabel('({})'.format(chr(97+i)), {'fontsize': 'large'})


        #plt.subplot(2, 5, i+6)
        ax[0].set_ylim(0, 0.35)
        ax[0].bar(index[:6], attn_1[:6], color = palette[0], align='center', label='1st Modality')
        ax[0].bar(index[6:12], attn_1[6:12], color = palette[1], align='center', label='2nd Modality')
        ax[0].bar(index[12:18], attn_1[12:18], color = palette[2], align='center', label='3rd Modality')
        ax[0].bar(index[18:24], attn_1[18:24], color = palette[3], align='center', label='4th Modality')
        ax[0].bar(index[24:30], attn_1[24:30], color = palette[4], align='center', label='5th Modality')
        ax[0].bar(index[30:36], attn_1[30:36], color = palette[5], align='center', label='6th Modality')
        #ax[1].set_xlabel('({})'.format(chr(97+i+5)), {'fontsize': 'large'})

        ax[2].set_ylim(0, 0.35)
        ax[2].bar(index[:6], attn_2[:6], color = palette[0], align='center', label='1st Modality')
        ax[2].bar(index[6:12], attn_2[6:12], color = palette[1], align='center', label='2nd Modality')
        ax[2].bar(index[12:18], attn_2[12:18], color = palette[2], align='center', label='3rd Modality')
        ax[2].bar(index[18:24], attn_2[18:24], color = palette[3], align='center', label='4th Modality')
        ax[2].bar(index[24:30], attn_2[24:30], color = palette[4], align='center', label='5th Modality')
        ax[2].bar(index[30:36], attn_2[30:36], color = palette[5], align='center', label='6th Modality')

    ax[0].set_xlabel('(a) NC', {'fontsize': 'x-large'})
    ax[1].set_xlabel('(b) sMCI', {'fontsize': 'x-large'})
    ax[2].set_xlabel('(b) AD', {'fontsize': 'x-large'})

    ax[0].set_ylabel('Attention score', {'fontsize': 'large'})
    ax[1].set_ylabel('Attention score', {'fontsize': 'large'})
    ax[2].set_ylabel('Attention score', {'fontsize': 'large'})

    plt.tight_layout()
    ax[0].legend(bbox_to_anchor=(1.1, 1.35), loc=2, borderaxespad=0, ncol=3, frameon=False, fontsize = 'x-large')
#fig.savefig('attn_avg/{}_{}_layer_no_self.png'.format(data_name,layer), dpi=600, format='png', bbox_inches = 'tight')