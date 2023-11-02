import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Data, Batch
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import random

import config
from feat import compute_x
from functools import reduce
import os

class Item(object):
    def __init__(self, x1, x2, adj, adj_norm) -> None:
        self.x1 = x1
        self.x2 = x2
        self.adj = adj
        self.adj_norm = adj_norm
    
    def unsqueeze(self,dim):
        self.x1 = self.x1.unsqueeze(dim)
        self.x2 = self.x2.unsqueeze(dim)
        self.adj = self.adj.unsqueeze(dim)
        self.adj_norm = self.adj_norm.unsqueeze(dim)


class Data(object):
    def __init__(self, x1, x2, adj, adj_norm) -> None:
        self.x1 = x1
        self.x2 = x2
        self.adj = adj
        self.adj_norm = adj_norm

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        return Item(self.x1[index], self.x2[index], self.adj[index], self.adj_norm[index])
    
    def to(self, device):
        self.x1 = self.x1.to(device)
        self.x2 = self.x2.to(device)
        self.adj = self.adj.to(device)
        self.adj_norm = self.adj_norm.to(device)

        return self


def load_data(data_path):
    # def _check_symmetry(adj):
    #     adjT = np.transpose(adj,(0,2,1))
    #     result = np.sum((adj-adjT)<1e-12) == np.prod(adj.shape)
    #     # if not result:
    #     #     for i,j,k in zip(*np.where(adj - adjT>1e-8)):
    #     #         print("(", i,j,k, ")", adj[i,j,k], adjT[i,j,k])
    #     return result
    
    data = sio.loadmat(data_path)
    data_name = os.path.basename(data_path).split('.')[0]

    thres = {'HIV':[0.6,0.01],'BP':[0.2,0.01],'PPMI':[0.01,0.01,0.01]}

    if data_name == 'PPMI':
        m1 = []
        m2 = []
        m3 = []
        label=[]
        t=0
        x = data['X']
        labels = data['label']
        n = x.shape[0]
        for i in range(n):
            m = x[i][0]
            if labels [i] == 0:
                m1.append(m[:,:,0].copy())
                m2.append(m[:,:,1].copy())
                m3.append(m[:,:,2].copy())
                label.append(labels[i])
            elif labels [i] == 1 and t!=149:
                m1.append(m[:,:,0].copy())
                m2.append(m[:,:,1].copy())
                m3.append(m[:,:,2].copy())
                label.append(labels[i])
                t=t+1

        m1 = np.stack(m1,2)
        m2 = np.stack(m2,2)
        m3 = np.stack(m3,2)
        labels=np.stack(label,1)
        graphs = [m1, m2, m3]
        print(m1.shape, m2.shape, m3.shape)
    else:
        fmri = data['fmri']
        dti = data['dti']
        labels = data['label']
        graphs = [dti, fmri]
        print(fmri.shape, dti.shape, labels.shape)
        
    labelenc = preprocessing.LabelEncoder()
    labels = labelenc.fit_transform(labels.flatten())
    print("label 0:", np.sum(labels==0), "label 1:", np.sum(labels==1))
    # print(labels.shape, labels)
    labels = torch.LongTensor(labels)
    data_ls = []
    for i in range(len(graphs)):
        adj = process_adj(graphs[i], thres[data_name][i])
        adj_norm = normalize_adj(adj)
        x1, x2 = process_feat(adj)
        data_ls.append(Data(x1,x2,adj,adj_norm))
    return *data_ls, labels

def process_adj(A, threshold):
    A = np.transpose(A,(2,0,1))  # (N,N,n) -> (n,N,N)
    adj = (A>threshold) #* A
    mask = np.ones_like(adj)-np.expand_dims(np.eye(adj.shape[1]),axis=0)
    adj = adj*mask # remove diagonal
    adj = torch.from_numpy(adj).float()
    return adj

def normalize_adj(adjs):
    def _normalize(adj):
        adj = adj + torch.eye(adj.shape[0])
        rowsum = adj.sum(1)
        d_inv_sqrt = rowsum**(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        # import pdb; pdb.set_trace()
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    n = adjs.shape[0]
    adjs_norm = []
    for i in range(n):
        adjs_norm.append(_normalize(adjs[i]))
    # import pdb; pdb.set_trace()
    adjs_norm = torch.stack(adjs_norm)
    return adjs_norm

def process_feat(adj):
    # adj = torch.from_numpy(adj).float()
    x1 = compute_x(adj, "adj")
    #x1 = torch.from_numpy(features).float()
    x2 = compute_x(adj, "eigen")
    # convert to pyG
    # N = adj.shape[0]
    # feat_batch1 = []
    # feat_batch2 = []
    # for i in range(N):
    #     edge_index = torch.nonzero(adj[i]).t().contiguous()
    #     graph1 = Data(x=x1[i],edge_index=edge_index,y=labels[i])
    #     graph2 = Data(x=x2[i],edge_index=edge_index,y=labels[i])
    #     feat_batch1.append(graph1)
    #     feat_batch2.append(graph2)
    # feat1 = Batch.from_data_list(feat_batch1)
    # feat2 = Batch.from_data_list(feat_batch2)
    return x1, x2


# data
class  FdDataset(Dataset):
    def __init__(self, data, mode, train_idx=None, val_idx=None, test_idx=None, seed=42):
        """
        Args:
            data: tuple of (fx1, fx2, dx1,dx2, ...)
            mode: train or test
        """
        super(FdDataset, self).__init__()
        self.data = data
        self.mode = mode
        if (mode=='train' and train_idx is None) or (mode=='valid' and val_idx is None) or \
            (mode=='test' and test_idx is None):
            #np.random.seed(seed)
            #labels = data[-1]
            #pos_idx = np.where(labels==1)[0]
            #neg_idx = np.where(labels!=1)[0]
            #train_idx_pos, val_idx_pos, test_idx_pos = self._spit_idx(pos_idx)
            #train_idx_neg, val_idx_neg, test_idx_neg = self._spit_idx(neg_idx)
            #train_idx = np.concatenate([train_idx_pos, train_idx_neg])
            #val_idx = np.concatenate([val_idx_pos, val_idx_neg])
            #test_idx = np.concatenate([test_idx_pos, test_idx_neg])
            #np.random.shuffle(train_idx)
            #np.random.shuffle(val_idx)
            #np.random.shuffle(test_idx)
            n = len(data[-1])
            inds = list(range(n))
            random.seed(seed)
            random.shuffle(inds)
            n_train = int(n*config.train_rate)
            n_test = int(n*config.test_rate)
            n_val = n - n_train - n_test
            train_idx,val_idx,test_idx=inds[:n_train],inds[n_train:n_train+n_val],inds[n_train+n_val:]

        self.idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
    
    def _spit_idx(self, inds, shuffle=True):
        n = len(inds)
        if shuffle:
            np.random.shuffle(inds)
        n_train = int(n*config.train_rate)
        n_test = int(n*config.test_rate)
        n_val = n - n_train - n_test
        return inds[:n_train],inds[n_train:n_train+n_val],inds[n_train+n_val:]
        
    def __len__(self):
        return len(self.idx[self.mode])

    def __getitem__(self, index):
        index = self.idx[self.mode][index]
        return [x[index] for x in self.data]


def merge_item2(ita, itb):
    if isinstance(ita, Item):
        ita.unsqueeze(0)
    if isinstance(itb, Item):
        itb.unsqueeze(0)
    x1 = torch.cat([ita.x1, itb.x1])
    x2 = torch.cat([ita.x2, itb.x2])
    adj = torch.cat([ita.adj, itb.adj])
    adj_norm = torch.cat([ita.adj_norm, itb.adj_norm])

    return Data(x1,x2,adj,adj_norm)

def collate_fn(data):
    data = list(zip(*data))
    labels = data[-1]
    items = data[:-1]
    output = []
    for k in range(len(items)):
        output.append(reduce(merge_item2, items[k]))
    labels = torch.stack(labels)
    return *output, labels

if __name__ == '__main__':
    # import numpy as np
    plot = True
    if plot:
        data = sio.loadmat("data/PPMI/PPMI.mat")
        # import pdb; pdb.set_trace()
        # fmri = data['fmri']
        # dti = data['dti']
        m1 = []
        m2 = []
        m3 = []
        x = data['X']
        n = x.shape[0]
        for i in range(n):
            m = x[i][0]
            m1.append(m[:,:,0].copy())
            m2.append(m[:,:,1].copy())
            m3.append(m[:,:,2].copy())
        m1 = np.stack(m1,2)
        m2 = np.stack(m2,2)
        m3 = np.stack(m3,2)
        print(m1.shape, m2.shape, m3.shape)
        # N,_,n = dti.shape
        dti = m3
        fmri = m3
        # print('min', fmri.min(), 'max', fmri.max(), 'mean', fmri.mean())
        print('min', dti.min(), 'max', dti.max(), 'mean', dti.mean())
        # adjs = []
        # for thres in np.arange(0.1,1,0.1):
        #     adjs.append(process_adj(dti[:,:,0],thres))
        adj = process_adj(fmri,0.01)
        row = 7
        col = 10
        # fadj_all = np.zeros((row*N,col*N))
        fig = plt.figure(figsize=(15,10))
        fig.tight_layout()
        for i in range(row):
            for j in range(col):
                ax = fig.add_subplot(row,col,i*col+j+1)
                # fadj_all[i*N:i*N+N,j*N:j*N+N] = fmri[:,:,i*col+j]
                ax.imshow(adj[i*col+j])
                ax.axis("off")
        # plt.imshow(fadj_all)
        # for i in range(3):
        #     for j in range(3):
        #         ax = fig.add_subplot(3,3,i*3+j+1)
        #         ax.imshow(adjs[i*3+j])
        #         ax.axis("off")
        plt.savefig("PPMI-3_0.01.png")
        plt.show()
    else:
        # fx1, fx2, dx1, dx2,fadj_norm,dadj_norm,fadj,dadj,_ = load_data("data/HIV")
        # print(fx1.shape, fx2.shape, dx1.shape, dx2.shape, fadj_norm.shape, dadj_norm.shape)
        # import pdb; pdb.set_trace()
        from torch.utils.data import DataLoader
        data = load_data("data/HIV/HIV.mat")
        dataset = FdDataset(data,'train',train_idx=range(64))
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False,collate_fn=collate_fn)
        batch = next(iter(data_loader))
        import pdb; pdb.set_trace()
        print(len(batch), batch[0], batch[1])



    
