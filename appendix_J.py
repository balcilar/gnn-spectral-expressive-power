
from torch_geometric.data import DataLoader,InMemoryDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch.nn import Sequential, Linear, ReLU
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,
                                global_mean_pool,GATConv,ChebConv,GCNConv)
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from utils import BandClassDataset


# read dataset
dataset = BandClassDataset(root='dataset/bandclass', pre_transform=None)

# split dataset
train_loader = DataLoader(dataset[0:3000], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[3000:4000], batch_size=100, shuffle=False)
test_loader = DataLoader(dataset[4000:5000], batch_size=100, shuffle=False)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(dataset.num_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64)

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)

        
        self.fc1 = torch.nn.Linear(64, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):

        x=data.x        
            
        edge_index=data.edge_index

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x) 

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)          

        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x) 


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 32*2, cached=False)
        self.conv2 = GCNConv(32*2, 64*2, cached=False)
        self.conv3 = GCNConv(64*2, 64*2, cached=False)       
        
        self.fc1 = torch.nn.Linear(64*2, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv1(x, edge_index))  
        x = F.dropout(x, p=0.1, training=self.training)      
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv3(x, edge_index)) 

        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.conv1 = torch.nn.Linear(dataset.num_features, 32)
        self.conv2 = torch.nn.Linear(32, 64)
        self.conv3 = torch.nn.Linear(64, 64)       
        
        self.fc1 = torch.nn.Linear(64, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x)) 
        x = F.dropout(x, p=0.3, training=self.training)       
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv3(x)) 
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=5
        self.conv1 = ChebConv(dataset.num_features, 32,S)
        self.conv2 = ChebConv(32, 64, S)
        self.conv3 = ChebConv(64, 64, S)
        
        self.fc1 = torch.nn.Linear(64, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index)) 
        x = F.dropout(x, p=0.2, training=self.training)       
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x, edge_index)) 

        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x) 



class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 8, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(64, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)

        self.fc1 = torch.nn.Linear(128, 10)
        self.fc2 = torch.nn.Linear(10, 1) 

    def forward(self, data):
        x=data.x       
                            
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv3(x, data.edge_index)) 

        x = global_mean_pool(x, data.batch)        
        x = self.fc1(x)        
        return self.fc2(x) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChebNet().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_grd= (data.y) 
        pre=model(data)
        pred=F.sigmoid(pre)[:,0]        
        lss=F.binary_cross_entropy(pred, y_grd,reduction='sum')        
        lss.backward()
        optimizer.step()        
        correct += torch.round(pred).eq(y_grd).sum().item()

        L+=lss.item()
    return correct/3000,L/3000

def test():
    model.eval()
    correct = 0
    L=0
    for data in test_loader:
        data = data.to(device)
        pre=model(data)
        pred=F.sigmoid(pre)[:,0]
        y_grd= (data.y)
        correct += torch.round(pred).eq(y_grd).sum().item()        
        lss=F.binary_cross_entropy(pred, y_grd,reduction='sum')
        L+=lss.item()

    s1= correct / 1000
    correct = 0
    Lv=0
    for data in val_loader:
        data = data.to(device)
        pre=model(data)
        pred=F.sigmoid(pre)[:,0]
        y_grd= (data.y)
        correct += torch.round(pred).eq(y_grd).sum().item()
        lss=F.binary_cross_entropy(pred, y_grd,reduction='sum')
        Lv+=lss.item()

    s2= correct / 1000
    return s1,L/1000, s2, Lv/1000

bval=1000
btest=0
for epoch in range(1, 101):
    tracc,trloss=train(epoch)
    test_acc,test_loss,val_acc,val_loss = test()
    if bval>val_loss:
        bval=val_loss
        btest=test_acc    
    print('Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Valloss: {:.4f}, Val acc: {:.4f},Testloss: {:.4f}, Test acc: {:.4f},best test acc: {:.4f}'.format(epoch,trloss,tracc,val_loss,val_acc,test_loss,test_acc,btest))

