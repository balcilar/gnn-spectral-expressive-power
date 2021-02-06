from torch_geometric.data import DataLoader
import torch
import scipy.io as sio
from torch_geometric.data.data import Data
import numpy as np

import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,global_add_pool,
                                global_mean_pool,GATConv,ChebConv,GCNConv)
from torch_geometric.datasets import MNISTSuperpixels

class DegreeMaxEigTransform(object):   

    def __init__(self,adddegree=True,maxdeg=40,addposition=False):
        self.adddegree=adddegree
        self.maxdeg=maxdeg
        self.addposition=addposition

    def __call__(self, data):

        n=data.x.shape[0] 
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1         
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(1/self.maxdeg*A.sum(0)).unsqueeze(-1)],1)
        if self.addposition:
            data.x=torch.cat([data.x,data.pos],1)


        d = A.sum(axis=0) 
        # normalized Laplacian matrix.
        dis=1/np.sqrt(d)
        dis[np.isinf(dis)]=0
        dis[np.isnan(dis)]=0
        D=np.diag(dis)
        nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
        V,U = np.linalg.eigh(nL)               
        vmax=np.abs(V).max()
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax=vmax.astype(np.float32)        
        return data
   
#select if node degree and location of superpixel region would be used by model or not.
#after any chnageing please remove MNIST/processed folder in order to preprocess changes again.
transform=DegreeMaxEigTransform(adddegree=True,addposition=False)


train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True, pre_transform=transform)
test_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=False, pre_transform=transform)
train_loader = DataLoader(train_dataset[0:55000], batch_size=64, shuffle=True)
val_loader   = DataLoader(train_dataset[55000:60000], batch_size=1000, shuffle=False)
test_loader  = DataLoader(test_dataset[0:10000], batch_size=1000, shuffle=False)
trsize=55000
tsize=10000
vsize=5000


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        nn=64
        self.conv1 = GCNConv(ninp, nn, cached=False)
        self.conv2 = GCNConv(nn, nn, cached=False)
        self.conv3 = GCNConv(nn, nn, cached=False)        
        
        self.fc1 = torch.nn.Linear(nn, 32)
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x, edge_index))        
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index)) 
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))        
        return F.log_softmax(self.fc2(x), dim=1)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        self.conv1 = GATConv(ninp, 8, heads=8, dropout=0.0)        
        self.conv2 = GATConv(8 * 8, 16, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(8 * 16, 16, heads=8, concat=True, dropout=0.0)  

        self.fc1 = torch.nn.Linear(128, 32)      
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):
        x=data.x       

        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = self.conv3(x, data.edge_index) 
        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        S=5
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes
        self.conv1 = ChebConv(ninp, 64,S)
        self.conv2 = ChebConv(64, 128, S)
        self.conv3 = ChebConv(128, 128, S)
        
        self.fc1 = torch.nn.Linear(128, 32)
        self.fc2 = torch.nn.Linear(32, nout) #int(d.num_classes))

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        x = F.dropout(x, p=0.1, training=self.training)
        #x = F.relu(self.conv1(x, edge_index)) 
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = F.dropout(x, p=0.1, training=self.training)       
        #x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = F.dropout(x, p=0.1, training=self.training)
        #x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1) 


class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()
        ninp=train_dataset.num_features
        nout=train_dataset.num_classes

        nn1 = Sequential(Linear(ninp, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(128) 

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)       
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, nout)

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)  
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x) 
        

        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        ninp=train_dataset.num_features
        nout=train_dataset.num_classes

        self.conv1 = torch.nn.Linear(ninp, 64)   
        self.conv2 = torch.nn.Linear(64, 64) 
        self.conv3 = torch.nn.Linear(64, 64) 

        self.fc1 = torch.nn.Linear(64, 32)     
        self.fc2 = torch.nn.Linear(32, nout) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  

        x = global_mean_pool(x, data.batch)
        x = F.elu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChebNet().to(device)   #  GcnNet  GatNet  ChebNet GinNet MlpNet

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    
    L=0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        lss=F.nll_loss(pred, data.y,reduction='sum')
        L+=lss
        lss.backward()
        optimizer.step()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s1= correct / trsize
    return L.cpu().detach().numpy()/trsize,s1

def test():
    model.eval()
    correct = 0
    Lt=0
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        
        lss=F.nll_loss(pred, data.y,reduction='sum')
        Lt+=lss.cpu().detach().numpy()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s1= correct / tsize
    Lt=Lt/tsize
    correct = 0
    Lv=0
    for data in val_loader:
        data = data.to(device)
        pred = model(data)
        
        lss=F.nll_loss(pred, data.y,reduction='sum')
        Lv+=lss.cpu().detach().numpy()
        pred = pred.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    s2= correct / vsize
    Lv=Lv/vsize
    return s1,Lt,s2,Lv

bval=0
btest=0
for epoch in range(1, 3001):
    trloss,tr_acc=train(epoch)
    test_acc,tloss,val_acc,vloss = test()
    if bval<val_acc:
        bval=val_acc
        btest=test_acc
    print('Epoch: {:02d}, train: {:.4f},{:.4f}, Val: {:.4f},{:.4f}, Test: {:.4f}, {:.4f} besttest:{:.4f} '.format(epoch,trloss,tr_acc,vloss,val_acc,tloss, test_acc,btest))
