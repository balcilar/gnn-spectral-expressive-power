import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,GINConv,ARMAConv,
                                global_mean_pool,GATConv,ChebConv,GCNConv)
from utils import TwoDGrid
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



# read dataset
dataset = TwoDGrid(root='dataset/2Dgrid', pre_transform=None)

# it consists of just one graph
train_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# ntask bandpass:0,  lowpass:1, highpass:2  
ntask=0


class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(1, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64)


        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)

        nn4 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv4 = GINConv(nn4,train_eps=True)
        self.bn4 = torch.nn.BatchNorm1d(64)        
        
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):

        x=data.x            
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x=self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x=self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x=self.bn3(x)

        x = F.relu(self.conv4(x, edge_index))
        x=self.bn4(x)        
        
        return self.fc2(x) 


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(1, 64*5, cached=False)
        self.conv2 = GCNConv(64*5, 64*5, cached=False)
        self.conv3 = GCNConv(64*5, 64, cached=False)  
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))        
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.conv1 = torch.nn.Linear(1, 64)   
        self.conv2 = torch.nn.Linear(64, 64) 
        self.conv3 = torch.nn.Linear(64, 64)      
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))       
        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self,S=5):
        super(ChebNet, self).__init__()
        
        self.conv1 = ChebConv(1, 64,S)    
        self.conv2 = ChebConv(64, 64,S) 
        self.conv3 = ChebConv(64, 64,S)
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x
              
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))   
        x = F.relu(self.conv2(x, edge_index)) 
        x = F.relu(self.conv3(x, edge_index))   
           
        return self.fc2(x) 



class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(1, 8, heads=8,concat=True, dropout=0.0)  
        self.conv2 = GATConv(64, 8, heads=8,concat=True, dropout=0.0) 
        self.conv3 = GATConv(64, 8, heads=8,concat=True, dropout=0.0)  
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x
          
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.elu(self.conv3(x, data.edge_index))  
        
        return self.fc2(x) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChebNet().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def visualize(tensor):
    y=tensor.detach().cpu().numpy()
    y=np.reshape(y,(95,95))
    plt.imshow(y.T);plt.colorbar();plt.show()


def train(epoch):
    model.train()
    ns=0
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)        
        lss= torch.square(data.m*(pre- data.y[:,ntask:ntask+1])).sum()        
        
        lss.backward()
        optimizer.step()        

        L+=lss.item()

        a=pre[data.m==1]    
        b=data.y[:,ntask:ntask+1] 
        b=b[data.m==1] 
        r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())


        # if you want to see the image that GNN  produce
        # visualize(pre)
    return L,r2


for epoch in range(1, 3001):
    
    trloss,r2=train(epoch)      
    print('Epoch: {:02d}, loss: {:.4f}, R2: {:.4f}'.format(epoch,trloss,r2))
    

for data in train_loader:
    data = data.to(device)
    pre=model(data)
    visualize(pre*data.m)
a=1