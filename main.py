import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import scatter
# from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import random
import os
import config
from process_data import load_data, FdDataset, collate_fn
from models import attentionLayer, enc_dec, shareLayer
from sklearn.metrics import accuracy_score, roc_curve,auc,f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(nargs=1, dest='data_path', default='data/HIV/HIV.mat')
args = arg_parser.parse_args()

data = load_data(args.data_path[0])

class Model(nn.Module):
    def __init__(self, nhid, attn_nhid, data, n_layers=2, dropout=0):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.encs = nn.ModuleList()
        for i in range(n_layers):
            self.encs.append(enc_dec.EncDec(data[i].x1.shape[-1], data[i].x2.shape[-1], nhid, dropout))
        self.snn = shareLayer.ShareNN(config.node_size, config.hidden_size, dropout)

        # attention
        nhid = config.hidden_size
        self.attention = attentionLayer.Attention(nhid, attn_nhid)
        
        self.fc = nn.Sequential(
            nn.Linear(nhid, config.num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, data):
        """
        Args:
            data: tuple of (fx1, fx2, dx1, dx2, fadj, dadj, ...)
        """
        # print(data[4].shape, data[5].shape)
        hids = []
        g_hids = []
        losses = []
        adj_norms = []
        # import pdb; pdb.set_trace()
        for i in range(self.n_layers):
            hid, g_hid, loss = self.encs[i](data[i].x1, data[i].x2, data[i].adj_norm)
            hids.append(hid)
            g_hids.append(g_hid)
            losses.append(loss)
            adj_norms.append(data[i].adj_norm)
        z = self.snn(*zip(hids, adj_norms))
        z_ = torch.mean(z, dim=1)
        g_hids.append(z_)
        hid = torch.stack(g_hids, dim=1)  #hid(attention)= modaltiy 1 modaltiy 2  (modaltiy 3) common
        hid, _ = self.attention(hid)
        hid = F.relu(hid)  #graph-level embedding
        output = self.fc(hid)
        loss = sum(losses)

        return output, loss



def loss_func(output, target):
    
    loss = F.nll_loss(output, target[-1])
    return loss

def auc_func(labels, scores, pos_label=1):
    posNum = np.sum(labels==pos_label)
    negNum = len(labels) - posNum
    if posNum == 0 or negNum == 0:
        return 0
    else:
        label_score = list(zip(labels, scores))
        label_score = sorted(label_score, key=lambda x: x[1])
        negSum = 0
        posGTNeg = 0
        for p in label_score:
            if p[0] == pos_label:
                posGTNeg += negSum
            else:
                negSum += 1
        return posGTNeg/(posNum * negNum)

def train(epoch):
    model.train()
    for i, data in enumerate(train_loader):
        data = [tensor.to(device) for tensor in data]
        output, loss1 = model(data)
        loss = loss_func(output, data)
        loss = loss + loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Train [Epoch {epoch} step {i}] [Loss {loss.item():.4f}]")
    return loss.item()


def test(epoch,test_loader,name="Test"):
    model.eval()
    total = 0
    loss = 0
    true_labels = []
    pred_labels = []
    pos_scores = []
    for data in test_loader:
        data = [tensor.to(device) for tensor in data]
        output, loss1 = model(data)
        loss_ = loss_func(output, data)
        loss_ = loss_ + loss1
        preds = output
        labels = preds.argmax(1)
        scores = torch.exp(preds)[:,1]  # positive score
        pos_scores.extend(scores.data.cpu().numpy())
        pred_labels.extend(labels.data.cpu().numpy())
        true_labels.extend(data[-1].data.cpu().numpy())
        # acc += (torch.argmax(output[-2],dim=-1)==data[-1]).sum()
        total += len(data[-1])
        loss += loss_*len(data[-1])
    loss = loss/total
    acc = accuracy_score(true_labels,pred_labels)
    macro_F1 = f1_score(true_labels,pred_labels, average='macro', labels=[0, 1])
    test_fpr, test_tpr, _ = roc_curve(true_labels, pos_scores,pos_label=1)
    auc_ = auc(test_fpr, test_tpr)
    # auc_ = auc_func(true_labels, pos_scores)
    print(f"{name} [Epoch {epoch}] [Accuracy {acc:.4f} AUC {auc_:.4f} F1-Mac {macro_F1:.4f} Loss {loss:.4f}]")
    return float(acc),auc_,float(macro_F1),float(loss)


device = torch.device(config.device)

# model = Model(config.hidden_size, config.attn_hidden_size)
# model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)

if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)

plot=True
# kf=StratifiedKFold(n_splits=config.n_fold, random_state=42, shuffle=True)
# 
accs = []
aucs = []
logf = open("acc.log",'w')
k=0
# for k, (train_idx, test_idx) in enumerate(kf.split(data[0], data[-1])):
# np.random.shuffle(train_idx)
# val_idx = train_idx[len(train_idx)-len(test_idx):]
# train_idx = train_idx[:len(train_idx)-len(test_idx)]

train_dst = FdDataset(data, 'train', train_idx=None)
val_dst = FdDataset(data, 'valid', val_idx=None)
test_dst = FdDataset(data, 'test', test_idx=None)
print("Fold", k, 'n_train:', len(train_dst), 'n_valid:', len(val_dst), 'n_test:', len(test_dst))

train_loader = DataLoader(train_dst, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dst, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_fn)   
test_loader = DataLoader(test_dst, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_fn) 

model = Model(config.hidden_size, config.attn_hidden_size, data, n_layers=len(data)-1, dropout=config.dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)

loss_ls = []
val_loss_ls = []
val_acc_ls = []
acc_ls = []
# best_val_acc = 0
# best_val_auc = 0
best_val_loss = np.inf
for i in range(config.n_epoch):
    loss = train(i)
    val_acc,val_auc,val_f1_mac,val_loss = test(i,val_loader,'Val')
    test_acc,_,_,_ = test(i, test_loader)
    # if val_auc > best_val_auc or (val_auc==best_val_auc and val_acc>=best_val_acc):
    if val_loss < best_val_loss:
        best_val_loss = val_loss 
        # best_val_acc = val_acc
        # best_val_auc = val_auc
        torch.save({'epoch': i, 'fold': k, 'model': model.state_dict()}, f"{config.checkpoint_dir}/model_best_val_Loss_fold{k}.pt")
    loss_ls.append(loss)
    val_loss_ls.append(val_loss)
    acc_ls.append(test_acc)
    val_acc_ls.append(val_acc)

torch.save(model.state_dict(), f"{config.checkpoint_dir}/model_fold{k}.pt")
ckpt = torch.load(f"{config.checkpoint_dir}/model_best_val_Loss_fold{k}.pt")
print('-'*30)
print('fold:', ckpt['fold'], 'epoch:', ckpt['epoch'])
model.load_state_dict(ckpt['model'])
acc_, auc_, f1_mac,_ = test(config.n_epoch,test_loader)
logf.write(f"fold: {k}, acc: {acc_:.4f},F1_mac: {f1_mac:.4f}, auc: {auc_:.4f}\n")
# accs.append(acc_)
# aucs.append(auc_)

if plot:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,3))
    plt.subplot(121)
    plt.plot(loss_ls,label='train loss')
    plt.plot(val_loss_ls,label='val loss')
    # plt.title('train loss')
    plt.subplot(122)
    plt.plot(acc_ls,label="test acc")
    plt.plot(val_acc_ls,label='val acc')
    # plt.title('test acc')
    plt.legend()
    plt.savefig('train_f%d.png'%(k))
    plt.show()
   
# print("-"*30)
# print(f"ACC: {np.mean(accs):.4f}±{np.std(accs):.4f}", f"AUC: {np.mean(aucs):.4f}±{np.std(aucs):.4f}")
# logf.write(f"ACC: {np.mean(accs):.4f}±{np.std(accs):.4f} AUC: {np.mean(aucs):.4f}±{np.std(aucs):.4f}\n")
logf.close()
