# Multi-modal Graph learning for Disease Prediction (MMGL)

This is a PyTorch transductive version of the MMGL model as proposed in our paper, the inductive version is be coming soon.

## Introduction
We hope MMGL as a flexible baseline could help you to explore more powerful variants and perform scenario-specific multi-modal adaptive graph learning for more biomedical tasks.

## Requirements
* PyTorch >= 1.1.0
* python 3.6
* networkx
* scikit-learn
* scipy
* munkres

## Run from
preset version:
```bash
./XX.sh
```
or modifying the network parameters and run
```bash
python run.py --hidden3 xxx --hidden2 xxx --learning_rate xxx ...
```

## Data

If you want to use your own data, you have to provide 
* a csv.file which contains multi-modal features, and
* a multi-modal feature dict.
