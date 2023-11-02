import torch
import torch.nn as nn

from .enc_dec import Structure_Decoder, MLP
from functools import reduce


class ShareNN(nn.Module):
    def __init__(self, N, nhid, dropout, alpha=15, gamma=-1):
        super(ShareNN, self).__init__()
        # self.S = nn.Parameter(torch.randn(N,N))
        # self.s_mask = nn.Parameter(torch.ones(N,N)-torch.eye(N),requires_grad=False)
        # self.lambs = nn.Parameter(torch.FloatTensor([1.,1.]))

        self.struct_decoder = Structure_Decoder(nhid, dropout)
        self.mlp = MLP(N, nhid)
        
        self.alpha = alpha
        self.gamma = gamma
        

    def f(self, A, p=2):
        """
        Args:
            A: normed adjacency matrix
            order: p-order proximity
            return f(A)
        """
        a = A
        a_ls = []
        for i in range(p-1):
            a = torch.matmul(a, A)
            a_ls.append(a)
        output = A
        for a in a_ls:
            output = output + a
        return output
        
    
    def forward(self, *z_adj):
        def _rd(a,b):
            return a+b
        
        def _update_S(_x_bar, _lambs, alpha):
            I = torch.eye(z1.shape[1]).to(z1.device)
            xxt = reduce(_rd, map(lambda x, y: y*torch.matmul(x,x.transpose(1,2)), _x_bar, _lambs))
            Fasum = reduce(_rd, map(lambda x,y: x*y, _lambs, a_ls))
            Isum = reduce(_rd, _lambs)
            tmp = torch.inverse(Isum * alpha * I + xxt)
            S = torch.matmul(tmp, alpha * Fasum + xxt)
            return S
        
        def _update_lamb(S, alpha, gamma):
             for j in range(len(lambs)):
                    lambs[j] = torch.pow(-1./gamma*(torch.sum((x_bar[j].transpose(1,2)-x_bar[j].transpose(1,2).matmul(S[j]))**2)+\
                                                  alpha*torch.sum((S[j]-a_ls[j])**2)), 1./(gamma-1))
        z1 = z_adj[0][0]
        x_bar = []
        a_ls = []
        lambs = [1.] * len(z_adj)

        for i in range(len(z_adj)):
            x_bar.append(z_adj[i][0])
            a_ls.append(self.f(z_adj[i][1]))

        for i in range(20):
            S = _update_S(x_bar, lambs, self.alpha)
            _update_lamb(S, self.alpha, self.gamma)
        
        #zs1 = S @ z1
        #zs2 = S @ z2
        #struct_recon1 = self.struct_decoder(zs1, adj1)
        #struct_recon2 = self.struct_decoder(zs2, adj2)
        #S=0.5*(torch.abs(S)+torch.abs(S.transpose(1,2)))
        s_feat = self.mlp(S)
        # s_loss = torch.square(s).sum() + torch.square(z1-zs1).sum() + torch.square(z2-zs2).sum()
        
        return s_feat