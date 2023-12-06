
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader,Data
from torch_geometric import utils
import torch.nn.functional as F
from torch_geometric.utils import degree, to_networkx,to_scipy_sparse_matrix
import argparse
import os
from torch.utils.data import random_split
from itertools import combinations





class customData(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super(customData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['./graph_train.dataset']

    def download(self):
        pass
    
    def process(self):
        word=['primary_school','high_school','music_makam','text_bbc']
        gnumber=[242,327,829,2225]
        rng=0
        label=0
        glabel=dict()
        graphs=dict()
        for w in word:
              print(label)      
              nverts=list()
              with open(w+'/'+w+'_nverts.txt') as f:
                  for line1 in f:
                     nverts.append(int(line1))
              f.close()

              simplices=list()
              with open(w+'/'+w+'_simplices.txt') as f:
                  for line2 in f:
                     simplices.append(int(line2))
              f.close()
              gindiator=list()
              with open(w+'/'+w+'_gindicator.txt') as f:
                  for line in f:
                      gindiator.append(int(line))
              f.close()
            
              k=0
              ind=0
              
              for j in range(rng+1,rng+gnumber[label]+1):
                  graphs[j]=-1
                  glabel[j]=label
              print(len(graphs))
              for i in nverts:
                  graph_ind=gindiator[ind]+rng
                  hyp_edge=list()
                  for j in range(k,k+i):
                       hyp_edge.append(simplices[j])
   
                  if i>1:
                       comb = combinations(hyp_edge, 2)
                       for f in list(comb):
                          if graphs[graph_ind]==-1:
                              graphs[graph_ind]={f}
                          else:
                              graphs[graph_ind]=graphs[graph_ind].union({f})
                  elif i==1:
                       if graphs[graph_ind]==-1:
                              graphs[graph_ind]={(hyp_edge[0],hyp_edge[0])}
                       else:
                              
                              graphs[graph_ind]=graphs[graph_ind].union({(hyp_edge[0],hyp_edge[0])})
                     
                  ind+=1
                  k=k+i
              rng=rng+gnumber[label]
              label+=1


        data_list = []
        print(len(graphs))
        for key in sorted(graphs.keys()):
            
            index = []
            
            nodes1 = [x[0] for x in graphs[key]]
            for i in nodes1:
              if i not in index:
                index.append(i)          
            nodes2 = [x[1] for x in graphs[key]]
            for i in nodes2:
              if i not in index:
                index.append(i)
            index_dic=dict()
            for i in range(0,len(index)):
                index_dic[index[i]]=i
            source_nodes=[index_dic[x[0]] for x in graphs[key]]
            target_nodes=[index_dic[x[1]] for x in graphs[key]]
            edge_index = torch.tensor([source_nodes,target_nodes], dtype=torch.long)
            
            x = degree(edge_index[0],len(index), dtype=torch.float).unsqueeze(1).view(-1, 1)
            label = glabel[key]
            #print(x.size()[0])
            #y = torch.LongTensor(label)

            data = Data(x=x, edge_index=edge_index, y=label)
            data.num_node=len(index) #x.size()[0]
            data.num_edge = data.edge_index.size(1)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        


#dataset = customData(root='/data/home/dlab/Ifte/SAGPool-master/SAGPool/text_bbc')

'''
for i in range(15): 
     
     print(str(i)+" "+str(dataset[i]['x'].size(0)))
     print(dataset[i]['edge_index'])

'''















  