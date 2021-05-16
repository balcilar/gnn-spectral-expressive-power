import numpy as np
from skimage.morphology import watershed
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio


def makegraph(im,npts=50):
    res=im.shape[0]
    I=np.zeros((res,res))    
    px=np.random.uniform(low=0,high=res,size=npts).astype(int)
    py=np.random.uniform(low=0,high=res,size=npts).astype(int)


    [X,Y]=np.meshgrid(range(res),range(res))
    sX=np.reshape(X,(res*res,1))
    sY=np.reshape(Y,(res*res,1))

    dx=sX-np.expand_dims(px,1).T
    dy=sY-np.expand_dims(py,1).T
    d=np.sqrt(dx*dx+dy*dy)
    dmin=d.min(1)
    D=np.reshape(dmin,(res,res))

    for i in range(npts):
        I[py[i],px[i]]=i+1
    labels = watershed(D, I, mask=np.ones((res,res)))

    labels-=1

    y1=labels[0:-1,:]
    y2=labels[1::,:]
    x1=labels[:,0:-1]
    x2=labels[:,1::]

    ucode=np.unique(np.reshape(y1,(res*(res-1),1))*npts+np.reshape(y2,(res*(res-1),1)))
    ucode2=np.unique(np.reshape(x1,(res*(res-1),1))*npts+np.reshape(x2,(res*(res-1),1)))
    ucode=np.unique(np.hstack((ucode,ucode2)))

    src=(ucode/npts).astype(np.int)
    trg=np.mod(ucode,npts)
    A=np.zeros((npts,npts))
    A[src,trg]=1
    A[trg,src]=1
    A=A*(np.ones((npts,npts))-np.eye(npts))    
    F=np.zeros((npts,4))
    for i in range(npts):
        F[i,:]=[im[labels==i].mean(), A[i,:].sum(),  Y[labels==i].mean(), X[labels==i].mean()] 

    # F : nptsx4 keeps node signal, node degree, vertical pos, horizontal pos
    # A : nptsxnpts adjacency matrix.
    return F,A

res=100
[X,Y]=np.meshgrid(range(2*res),range(2*res))

FF=[]
AA=[]
YY=[]
for i in range(2500):
    print(i)

    # generate class1. frequency is in [2-2.5] or [4-4.5]
    if np.random.uniform()<0.5:
        f= np.random.uniform(low=2,high=2.5)
    else:
        f= np.random.uniform(low=4,high=4.5)

    im=np.sin(X/X.shape[0]*4*f*np.pi)
    r= np.random.uniform(low=-90,high=90)
    imr=ndimage.rotate(im, r)
    rx,ry=imr.shape
    dx= int(np.random.uniform(low=0,high=10))
    im=imr[dx+int((rx-res)/2):dx+int((rx-res)/2)+res,int((ry-res)/2):int((ry-res)/2)+res]
    F,A=makegraph(im,npts=200)
    # if the graph is not connected, generate new to be sure that graph is connected
    while A.sum(1).min()==0:
        F,A=makegraph(im,npts=200)
    FF.append(F)
    AA.append(A)
    YY.append(f)

    # generate class2. frequency is in [1-1.9] or [2.6-3.9] or [4.6-5]
    f= np.random.uniform(low=1,high=5)
    while  not (f<1.9 or (f>2.6 and f<3.9) or f>4.6):
        f= np.random.uniform(low=1,high=5)

    im=np.sin(X/X.shape[0]*4*f*np.pi)
    r= np.random.uniform(low=-90,high=90)
    imr=ndimage.rotate(im, r)
    rx,ry=imr.shape
    dx= int(np.random.uniform(low=0,high=10))
    im=imr[dx+int((rx-res)/2):dx+int((rx-res)/2)+res,int((ry-res)/2):int((ry-res)/2)+res]
    F,A=makegraph(im,npts=200)
    # if the graph is not connected, generate new to be sure that graph is connected
    while A.sum(1).min()==0:
        F,A=makegraph(im,npts=200)
    FF.append(F)
    AA.append(A)
    YY.append(f)


# save Feature matrix, adjacency matrix and frequency that we generate pattern.
sio.savemat('bandpassgraph.mat',{'F':FF,'A':AA,'Y':YY})

# plot last image and its correspoded graph.
plt.imshow(im);plt.colorbar();plt.show()

import pygsp
G = pygsp.graphs.Graph(A)
G.set_coordinates(F[:,2::])
G.plot(F[:,0]);plt.show()

