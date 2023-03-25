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
from sklearn.model_selection import KFold,StratifiedKFold
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import pandas as pd
from network import *
from utils import *
from model import *
import dgl


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    dgl.random.seed(seed)
    
    
def sen(con_mat,n):#n为分类数
    
    sen = []
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def spe(con_mat,n):
    
    spe = []
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe
    
    
def train_and_eval(datadir, datname, hyperpm):
    set_rng_seed(hyperpm.seed)
    path = datadir + datname + '/'
    modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
    data = pd.read_csv(path + 'processed_standard_data.csv').values
    print('data shape: ', data.shape)
    if datname == 'TADPOLE':
        hyperpm.nclass = 3
        hyperpm.nmodal = 6
    elif datname == 'ABIDE':
        hyperpm.nclass = 2
        hyperpm.nmodal = 4
    #np.random.shuffle(data)
    
    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    input_data_dims = []
    for i in modal_feat_dict.keys():
        input_data_dims.append(len(modal_feat_dict[i]))
    print('Modal dims ', input_data_dims)
    input_data = data[:,:-1]
    label = data[:,-1]-1
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    set_rng_seed(hyperpm.seed)
    val_acc, tst_acc, tst_auc = [], [], []
    shared_acc_list, shared_auc_list = [], []
    sp_acc_list, sp_auc_list = [], []
    sens = []
    clk = 0
    for train_index, test_index in skf.split(input_data, label):
        clk += 1
        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index)
        tm = time.time()
        best_val_acc, wait_cnt = 0.0, 0
        model_sav = tempfile.TemporaryFile()
        for t in range(hyperpm.nepoch):
            print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
            agent.run_epoch(mode = hyperpm.mode, end=' ')
            _, cur_val_acc = agent.print_trn_acc(hyperpm.mode)
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [agent.ModalFusion.state_dict(),
                             agent.GraphConstruct.state_dict(),
                             agent.MessagePassing.state_dict()]
                torch.save(dict_list, model_sav)
            else:
                wait_cnt += 1
                if wait_cnt > hyperpm.early:
                    break
        print("time: %.4f sec." % (time.time() - tm))
        model_sav.seek(0)
        dict_list = torch.load(model_sav)
        agent.ModalFusion.load_state_dict(dict_list[0])
        agent.GraphConstruct.load_state_dict(dict_list[1])
        agent.MessagePassing.load_state_dict(dict_list[2])

        val_acc.append(best_val_acc)
        cur_tst_acc, cur_tst_auc = agent.print_tst_acc(hyperpm.mode)
        
        tst_acc.append(cur_tst_acc)
        tst_auc.append(cur_tst_auc)
        if np.array(tst_acc).mean() < 0.6 and clk == 5:
            break
    return np.array(val_acc).mean(), np.array(tst_acc).mean(), np.array(tst_acc).std(), np.array(tst_auc).mean(), np.array(tst_auc).std()


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='TADPOLE')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=150,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.65,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of conv layers.')
    parser.add_argument('--n_hidden', type=int, default=16,
                        help='Number of attention head.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='Number of alternate iteration.')
    parser.add_argument('--nmodal', type=int, default=6,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--th', type=float, default=0.9,
                        help='threshold of weighted cosine')
    parser.add_argument('--GC_mode', type=str, default='adaptive-learning',
                        help='graph constrcution mode')
    parser.add_argument('--MP_mode', type=str, default='GCN',
                        help='Massage Passing mode')
    parser.add_argument('--MF_mode', type=str, default=' ',
                        help='Massage Passing mode')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='alpha for GAT')
    parser.add_argument('--theta_smooth', type=float, default='1',
                        help='graph_loss_smooth')
    parser.add_argument('--theta_degree', type=float, default='0.5',
                        help='graph_loss_degree')
    parser.add_argument('--theta_sparsity', type=float, default='0.0',
                        help='graph_loss_sparsity')
    parser.add_argument('--nclass', type=int, default=3,
                        help='class number')
    parser.add_argument('--mode', type=str, default='pre-train',
                        help='training mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed setting')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        print('GC_mode:', args.GC_mode, 'MF_mode:', args.MF_mode)
        val_acc, tst_acc, tst_acc_std, tst_auc, tst_auc_std = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst_acc=%.2f%% tst_auc=%.2f%%' % (val_acc * 100, tst_acc * 100, tst_auc * 100))
        print('tst_acc_std=%.4f tst_auc_std=%.4f' % (tst_acc_std, tst_auc_std))
    return val_acc, tst_acc


if __name__ == '__main__':
    print(str(main()))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
