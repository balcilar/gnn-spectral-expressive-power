from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.data.data import Data
import scipy.io as sio
import numpy as np

class TwoDGrid(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["2Dgrid.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        F=F.astype(np.float32)
        Y=a['Y']
        Y=Y.astype(np.float32)
        M=a['mask']
        M=M.astype(np.float32)


        data_list = []
        E=np.where(A>0)
        edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        x=torch.tensor(F)
        y=torch.tensor(Y)   
        m=torch.tensor(M)     
        data_list.append(Data(edge_index=edge_index, x=x, y=y,m=m))
        
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BandClassDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(BandClassDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["bandclass.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        A=a['A']
        F=a['F']
        Y=a['Y']
        F=np.expand_dims(F,2)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i,:,:]) 
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])