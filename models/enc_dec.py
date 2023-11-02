import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import GCNConv
from .gcn import GraphConvolution
from models.attentionLayer import Attention
import math


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))

        gh = torch.cat((torch.sum(x,dim=1),torch.sum(x2,dim=1)), dim=-1)

        return x2, gh
    

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x
    

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.sigmod=nn.Sigmoid()

    def forward(self, x, adj):
        # import pdb; pdb.set_trace()
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(x,torch.transpose(x,1,2))
        #x=self.sigmod(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim) -> None:
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp_dim, out_dim*2),
            nn.ReLU(),
            nn.Linear(out_dim*2, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep
    
def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_loss_(l_enc, g_enc):

    num_graphs = g_enc.shape[0]  # (b,d)
    b, n, _ = l_enc.shape  # (b,n,d)
    
    l_enc = l_enc.reshape(b*n,-1)
    num_nodes = b*n
    graph_id = torch.cat([torch.zeros(n,dtype=torch.long)+i for i in range(b)])

    device = g_enc.device

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)

    for nodeidx, graphidx in enumerate(graph_id):

        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


class EncDec(nn.Module):
    def __init__(self, nfeat1, nfeat2, nhid, dropout):
        super(EncDec,self).__init__()
        self.encoder1 = Encoder(nfeat1, nhid, dropout)
        self.encoder2 = Encoder(nfeat2, nhid, dropout)
        
        self.local_mlp = MLP(nhid, nhid)
        self.global_mlp = MLP(2*nhid, nhid)
        # self.attr_decoder1 = Attribute_Decoder(nfeat1, nhid, dropout)
        # self.attr_decoder2 = Attribute_Decoder(nfeat2, nhid, dropout)
        # self.structure_decoder=Structure_Decoder(nhid, dropout)
        # self.attention = Attention(nhid,32)
        # self.mlp = nn.Linear(nhid*2, nhid)
    def forward(self, x1, x2, adj):
        hid1, hg1 = self.encoder1(x1, adj)
        hid2, hg2 = self.encoder2(x2, adj)

        # g_hid1 = torch.mean(hid1, dim=1)
        # g_hid2 = torch.mean(hid2, dim=1)

        hid1 = self.local_mlp(hid1)
        hid2 = self.local_mlp(hid2)

        g_hid1 = self.global_mlp(hg1)
        g_hid2 = self.global_mlp(hg2)

        loss1 = local_global_loss_(hid1, g_hid2)
        loss2 = local_global_loss_(hid2, g_hid1)
        
        hid = hid1 + hid2
        g_hid = g_hid1 + g_hid2

        # hid = torch.cat([hid1,hid2],dim=2)
        #print(hid.shape,'2222222')
        # hid=self.mlp(hid)
        # x1_hat = self.attr_decoder1(hid, adj)
        # x2_hat = self.attr_decoder2(hid, adj)
        # s_hat=self.structure_decoder(hid,adj)
        # hid_f = torch.stack([hid1, hid, hid2], dim=1)
        # hid_f,_=self.attention(hid_f)

        #hid = torch.add(hid1,hid2)
        #print(hid_f.shape,'111111111')
        loss = loss1 + loss2


        return hid, g_hid, loss