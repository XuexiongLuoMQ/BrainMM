import random
import torch
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(42)

#data_dir = "data/HIV"
# data_path = "data/HIV/HIV.mat"
batch_size = 16
hidden_size = 128
attn_hidden_size = 32
num_classes = 2
node_size = 82 # PPMI 84 # BP 82 HIV 90
# node_size = 82
train_rate = 0.8
test_rate = 0.1
lr = 0.005  #BP 0.005 HIV 0.01  #PPMI 0.0001
n_epoch = 100
n_fold = 10
device = "cuda:0"
checkpoint_dir = "checkpoint"
dropout = 0.1