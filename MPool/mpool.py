from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import SAGEConv,GCNConv as GCN,DenseGraphConv, GraphConv, DenseGCNConv,DenseGINConv, DenseSAGEConv 
from torch_geometric.utils import to_dense_batch, to_dense_adj
from MPool.m_pool_mod import dense_m_pool
from utils import fetch_assign_matrix
from ogb.graphproppred.mol_encoder import AtomEncoder
from utils import GCNConv
from torch_geometric.utils import to_networkx
from motifcluster import clustering as mccl
from motifcluster import motifadjacency as mcmo
import scipy
from scipy import sparse

from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix

import torch
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from MPool.layers import MPoolS
import os.path as osp



def motifMatrix(edge_index, motif_type):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        device = torch.device('cpu')
        if edge_index.numel() > 0:
             A=to_scipy_sparse_matrix(edge_index.cpu())
             A1=mcmo.build_motif_adjacency_matrix(A,motif_name=motif_type).todense()
             A2=mcmo.build_motif_adjacency_matrix(A,motif_name='M4').todense()
             A=A+A2        
             A = sparse.csr_matrix(A1)
             edge_index1=from_scipy_sparse_matrix(A)
             return edge_index1[0],edge_index1[1] 
        else:
             return edge_index,None



class MPool(torch.nn.Module):
    def __init__(self, num_features, num_classes, max_num_nodes, hidden, pooling_type,
                 num_layers, encode_edge=False):
        super(MPool, self).__init__()
        self.encode_edge = encode_edge

        self.atom_encoder = AtomEncoder(emb_dim=hidden)

        self.pooling_type = pooling_type
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.num_layers = num_layers

        self.conv1 = GraphConv(num_features, hidden)
        self.Spool1 = MPoolS(hidden, ratio=0.5)
        self.Sconv2 = GraphConv(hidden, hidden)
        self.Spool2 = MPoolS(hidden, ratio=0.5)
        self.Sconv3 = GraphConv(hidden, hidden)
        self.Spool3 = MPoolS(hidden, ratio=0.5)


        for i in range(num_layers):
            if i == 0:
                if encode_edge:
                    self.convs.append(GraphConv(hidden, aggr='add'))
                else:
                    self.convs.append(GraphConv(num_features, hidden, aggr='add'))
            else:
                self.convs.append(DenseSAGEConv(hidden, hidden))

        self.rms = []
        num_nodes = max_num_nodes
        for i in range(num_layers - 1):
            num_nodes = ceil(0.5 * num_nodes)
            if pooling_type == 'mlp':
                self.pools.append(Linear(hidden, num_nodes))
            else:
                self.rms.append(fetch_assign_matrix('uniform', ceil(2 * num_nodes), num_nodes))

        self.lin1_1 = torch.nn.Linear(hidden*2, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden*2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = F.relu(self.convs[0](x, edge_index, data.edge_attr))
        else:
            x = F.relu(self.convs[0](x, edge_index))
        #edge_index1,weight1= motifMatrix(edge_index,"M13")
        Sx, edge_index, _, Sbatch, perm= self.Spool1(x, edge_index,edge_index, None, batch)
        Sx1 = torch.cat([gmp(Sx, Sbatch), gap(Sx, Sbatch)], dim=1)

        x, mask = to_dense_batch(x, batch)
        
        adj = to_dense_adj(edge_index, batch)

        if self.pooling_type != 'mlp':
            s = self.rms[0][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
        else:
            s = self.pools[0](x)
        
        #x=x+s[perm.T]*Sx
        x, adj, mc, o = dense_m_pool(x, adj, s, mask)

        for i in range(1, self.num_layers - 1):
            x = F.relu(self.convs[i](x, adj))
            if self.pooling_type != 'mlp':
                s = self.rms[i][:x.size(1), :].unsqueeze(dim=0).expand(x.size(0), -1, -1).to(x.device)
            else:
                s = self.pools[i](x)
            x, adj, mc_aux, o_aux = dense_m_pool(x, adj, s)
            mc += mc_aux
            o += o_aux

        x = self.convs[self.num_layers-1](x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        
        Sx = F.relu(self.Sconv2(Sx, edge_index))
        Sx, edge_index, _, Sbatch, _ = self.Spool2(Sx, edge_index,edge_index,None, Sbatch)
        Sx2 = torch.cat([gmp(Sx, Sbatch), gap(Sx, Sbatch)], dim=1)
        
        Sx = F.relu(self.Sconv3(Sx, edge_index))
        Sx, edge_index, _, Sbatch, _ = self.Spool3(Sx, edge_index,edge_index, None, Sbatch)
        Sx3 = torch.cat([gmp(Sx, Sbatch), gap(Sx, Sbatch)], dim=1)
        
        Sx = Sx1 +Sx2+ Sx3

        Sx = F.relu(self.lin1_1(Sx))
        
        x=torch.cat((Sx,x),dim=1)
        x = self.lin2(x)
        return x, mc, o
