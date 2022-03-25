## Dataset preparation
#### Step 1: Data prprocessing
Running the code of data preprocessing in ./data/{dataset}/xxx.ipynb to preprocess the raw data to standard data as the input of MMGL.

#### Step 2: Training and test
Running ./{dataset}-simple-2-concat-weighted-cosine.sh
Notice: the sh file is used to reproduce the result reported in our paper, you could also run this script to train and test your own dataset:
```
python main.py
```
Besides, you can modify 'network.py' to establish a variant of MMGL and 'model.py' to try and expolre a better training strategy for other tasks.

#### Setp 3: Visualize the attention matrix (Optional)
you could modify the 'network.py' and 'model.py' to save the attention matrix as './attn/attn_map_{dataset}.npy'.
Then, run 
```
python attn_vis.py
python attn_vis2.py
```
to visualize the attention matrix.
