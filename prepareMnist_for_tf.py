import numpy as np
from torch_geometric.datasets import MNISTSuperpixels
from utils_tf import *
from utils import DegreeMaxEigTransform
   
#select if node degree and location of superpixel region would be used by model or not.
#after any chnageing please remove MNIST/processed folder in order to preprocess changes again.
transform=DegreeMaxEigTransform(adddegree=True,addposition=False)

train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True, pre_transform=transform)
test_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=False, pre_transform=transform)

# how many supports of Chebnet and Cayleynet to be prepared
nkernel=5
#cayleynet's zoom param
h=1

################

n=70000
nmax=75
# number of node per graph
ND=75*np.ones((n,1)) 
# node feature matrix
FF=np.zeros((n,nmax,2))
# one-hot coding output matrix 
YY=np.zeros((n,10))
# Convolution kernels, supports
SP=np.zeros((n,2*nkernel+1,nmax,nmax),dtype=np.float32)

d=train_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 
    FF[i,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i,gtrt]=1
    # set chebnet kernel first nkernel is chebnet's support
    chebnet = chebyshev_polynomials(A, nkernel-1,st=True)  
    for j in range(0,nkernel):
        SP[i,j ,0:nd,0:nd]=chebnet[j].toarray()    
    # set gcn kernel nkernel-th support i sgcn support
    SP[i,nkernel ,0:nd,0:nd]= (normalize_adj(A + sp.eye(nd))).toarray() 

    # last nkernel is cayleynet's support
    r=int((nkernel-1)/2)
    cayley=cayley_polynomials(A,h,r)
    for j in range(0,nkernel):
        SP[i,j+1+nkernel ,0:nd,0:nd]=cayley[j]

d=test_dataset
for i in range(0,len(d)):
    print(i)
    nd=75
    A=np.zeros((nd,nd),dtype=np.float32)        
    A[d[i].edge_index[0],d[i].edge_index[1]]=1 
    FF[i+60000,:,:]=d[i].x.numpy()
    gtrt=d[i].y.numpy()[0]
    YY[i+60000,gtrt]=1
    # set chebnet kernel
    chebnet = chebyshev_polynomials(A, nkernel-1,st=True)  
    for j in range(0,nkernel):
        SP[i+60000,j ,0:nd,0:nd]=chebnet[j].toarray()    
    # set gcn kernel
    SP[i+60000,nkernel ,0:nd,0:nd]= (normalize_adj(A + sp.eye(nd))).toarray()

    # last nkernel is cayleynet's support
    r=int((nkernel-1)/2)
    cayley=cayley_polynomials(A,h,r)
    for j in range(0,nkernel):
        SP[i+60000,j+1+nkernel ,0:nd,0:nd]=cayley[j]

np.save('supports',SP)
np.save('feats',FF)
np.save('output',YY)
np.save('nnodes',ND)


