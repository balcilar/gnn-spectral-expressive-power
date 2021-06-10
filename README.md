# Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective

This repository consists of codes of ["Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective"](https://openreview.net/forum?id=-qh0M9XWxnv) paper published in ICLR2021.

## Theoretical and Empirical Frequency Responses
Here are our theoretical findings on frequency responses of GNNs

![Sample image](images/freqresponse.jpg?raw=true "Title")


Studied GNN's spectral analysis codes are in the spectral_analysis_codes folder. Those codes were written in Matlab.
You can briefly run each script for each method independently and see the empirical frequency responses on the Cora graph and the theoretical ones.

Chebnet's first 5 convolution support's frequency response
```
>chebnet_spect_analysis
```

![Sample image](images/cheb.jpg?raw=true "Title")

CayleyNet's first 7 convolution support's frequency response where zoom parameter is 1.
```
>cayleynet_spect_analysis
```

![Sample image](images/cayley.jpg?raw=true "Title")

GIN's frequency response under epsilon for -2,-1,0 and 1
```
>gin_spect_analysis
```
![Sample image](images/gin.jpg?raw=true "Title")

GCN's frequency response
```
>gcn_spect_analysis
```
![Sample image](images/gcn.jpg?raw=true "Title")

GAT's frequency response
```
>gat_spect_analysis
```
![Sample image](images/gat.jpg?raw=true "Title")


## Datasets
In this research, we introduced two new datasets and one common dataset. 

### 2D-grid graph
For the spectral expressive power test, we introduced 2D-grid graph consist of 95x95 resolution and a 4-neighborhood regular grid graph. Each node refers to the pixel in the image. We prepared low-pass, high-pass, and band-pass filtering results as the output of the filter learning task.
You can load that dataset and visualize it by following the Matlab script
```
load dataset/2Dgrid/raw/2Dgrid
subplot(2,2,1);imagesc(reshape(F,[95 95]));axis equal
title('given images');
subplot(2,2,2);imagesc(reshape(Y(:,1),[95 95]));axis equal
title('band-pass images');
subplot(2,2,3);imagesc(reshape(Y(:,2),[95 95]));axis equal
title('low-pass images');
subplot(2,2,4);imagesc(reshape(Y(:,3),[95 95]));axis equal
title('high-pass images');
```

![Sample image](images/filter.jpg?raw=true "Title")

You can also read the dataset using Pytorch-geometric by following Python script. The dataset consists of just one graph, its adjacency, pixel value, 3 different target value and mask which we exclude 2 pixels borderline.
```
from utils import TwoDGrid
dataset = TwoDGrid(root='dataset/2Dgrid', pre_transform=None)
print(len(dataset))
print(dataset[0])

1
Data(edge_index=[2, 358685], m=[9025, 1], x=[9025, 1], y=[9025, 3])
```

### BandClass
Another introduced dataset for spectral expressive power is the BandClass dataset, which consists of 3K train, 1K validation, and 1K test planar graph, each has 200 nodes. Problem is a binary classification problem where the ground truth of classes was determined by the frequency on the graph. You can visualize two sample graphs using gspbox library by following the Matlab script. 

```
load dataset/bandclass/raw/bandclass
i=21;
G=gsp_graph(squeeze(A(i,:,:)),squeeze(coor(i,:,:)));
figure;gsp_plot_signal(G,F(i,:))
i=1000;
G=gsp_graph(squeeze(A(i,:,:)),squeeze(coor(i,:,:)));
figure;gsp_plot_signal(G,F(i,:))
```
![Sample image](images/graph.jpg?raw=true "Title")

You can read this dataset using Pytorch-geometric by following Python script. Note that even though the dataset includes the node coordinates, we neglected it in order to make the problem harder and more realistic in terms of graph research.
```
from utils import BandClassDataset
dataset = BandClassDataset(root='dataset/bandclass', pre_transform=None)
print(len(dataset))
print(dataset[0])

5000
Data(edge_index=[2, 1074], x=[200, 1], y=[1])
```

### MNIST-75
Last, we used MNIST superpixel dataset in our analysis. You can download and extract the "train.pt" and "test.pt" files into dataset/MNIST/raw/ folder. Here is the most recent link of Mnist-75.
https://graphics.cs.tu-dortmund.de/fileadmin/ls7-www/misc/cvpr/mnist_superpixels.tar.gz

Sample 2 graph in this dataset can be found in the following figure.
![Sample image](images/mnist46.jpg?raw=true "Title")

The dataset can be load using Pytorch-geometric by the following script. Given sample transform can add node degree and node coordinate to the pixel value as node features. However, we do not use superpixel coordinate in our analysis in order to make the problem harder and more realistic in terms of graph research. 

```
from torch_geometric.datasets import MNISTSuperpixels
from utils import DegreeMaxEigTransform
transform=DegreeMaxEigTransform(adddegree=True,addposition=False)
train_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=True, pre_transform=transform)
test_dataset = MNISTSuperpixels(root='dataset/MNIST/', train=False, pre_transform=transform)
print(len(train_dataset))
print(train_dataset[0])
print(len(test_dataset))
print(test_dataset[0])

60000
Data(edge_index=[2, 1399], lmax=1.2635252, pos=[75, 2], x=[75, 2], y=[1])
10000
Data(edge_index=[2, 1405], lmax=1.2760227, pos=[75, 2], x=[75, 2], y=[1])
```

# Performance Analysis of GNNs

## 2DGrid Dataset Results

Briefly launch the appendix_I.py script. You set the method selection hardcoded. You can select your method and launch the script.

	python appendix_I.py

## BandClass Dataset Results

Briefly launch the appendix_J.py script. You set the method selection hardcoded. You can select your method and launch the script.

	python appendix_J.py

## MNIST Dataset Results

There is two different Mnist implementations, one is using pytorch_geometric. Basically you can launch the mnist_code.py script. You set the method selection hardcoded. You can select your method out of ChebNet, GIN, GAT, GCN, MLP and launch the script.  

	python mnist_code.py
	
For tensorflow1 implementation, we do not use any GNN library. First you need to run the data preparation code. It will read the Mnist-75 dataset, extract adjacency, calculates the CayleyNet, Chebnet and GCN supports and save the supports into disk in numpy matrix format. It takes a while. But it should be run just once.

	python prepareMnist_for_tf.py
Later, you can launch the code written by tensorflow1 as follows.

	python mnist_tf_code.py
We have notice that, since the Mnist-75 graphs are quite dense, full matrix multiplication implementation (our tf1 implementation) may be faster than edge based multiplication. Thus tf1 implementation is quite faster. You can run tf1 implementation for CayleyNet, GCN and MLP in addition to Chebnet. Around 50th epoch, Chebnet with 5 convolution supports in tf1 version is converged.

	Epoch: 46, trainloss: 0.2266, Val: 0.1991,val acc 0.9378, Test: 0.2447, test acc 0.9212 besttest: 9266 
	Epoch: 47, trainloss: 0.2228, Val: 0.1995,val acc 0.9384, Test: 0.2558, test acc 0.9181 besttest: 9266 
	Epoch: 48, trainloss: 0.2245, Val: 0.2076,val acc 0.9320, Test: 0.2656, test acc 0.9175 besttest: 9266 
	Epoch: 49, trainloss: 0.2209, Val: 0.1840,val acc 0.9428, Test: 0.2260, test acc 0.9276 besttest: 9276 
	Epoch: 50, trainloss: 0.2222, Val: 0.2054,val acc 0.9348, Test: 0.2450, test acc 0.9207 besttest: 9276


## Requirements
These libraries' versions are not strictly needed. But these are the configurations of our test machine. Also, all dependencies of pytorch-geometric are needed.
- Python==3.8.5
- pytorch==1.5.1
- pytorch_geometric==1.6.1
- numpy==1.19.1
- scipy==1.5.2
- matplotlib==3.3.1

For tensorflow codes our test environment has following libraries' versions
- Python==3.6.5
- tensorflow-gpu==1.15.0
- numpy==1.17.4
- matplotlib==3.1.2
- scipy==1.3.1
- networkx==2.4
- pickle==4.0

## Citation

Please cite this paper if you use codes and/or datasets in your work,

	@inproceedings{
	balcilar2021analyzing,
	title={Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective},
	author={Muhammet Balcilar and Guillaume Renton and Pierre H{\'e}roux and Benoit Ga{\"u}z{\`e}re and S{\'e}bastien Adam and Paul Honeine},
	booktitle={International Conference on Learning Representations},
	year={2021},
	url={https://openreview.net/forum?id=-qh0M9XWxnv}
	}

  
## License
MIT License
