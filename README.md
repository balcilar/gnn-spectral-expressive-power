# Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective

This repository consist of codes of ["Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective"](https://openreview.net/forum?id=-qh0M9XWxnv) paper published in ICLR2021.

## Theoretical and Empirical Frequency Responses
Here is the our theoretical findings on frequency responses of GNNs

![Sample image](images/freqresponsetable.jpg?raw=true "Title")


Studied GNN's spectral analysis codes are in spectral_analysis_result folder. Those codes were written in Matlab.
You can briefly run each script for each method independently and see the empirical and theoretical frequency responses on Cora graph.
For Chebnet's frequency response
```
>chebnet_spectral_analysis
```

![Sample image](images/cheb.jpg?raw=true "Title")

For GIN's frequency response under epsilon for -2,-1,0 and 1
```
>gin_spectral_analysis
```
![Sample image](images/gin.jpg?raw=true "Title")

For GCN's frequency response
```
>gcn_spectral_analysis
```
![Sample image](images/gcn.jpg?raw=true "Title")

For GAT's frequency response
```
>gat_spectral_analysis
```
![Sample image](images/gat.jpg?raw=true "Title")


## Datasets
In this research, we introduced two different new datasets and one common dataset. 

### 2D-grid graph
One named 2D-grid graph consist of 95x95 resolution and 4-neighborhood regular grid graph. Each nodes refers the pixel in the image. We prepared low-pass, high-pass and band-pass filtering results as output of filter learning task.
You can load that dataset and viualize it by follwing matlab script
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

You can also read the dataset using Pytorch-geometric by following Python script. Dataset consist of just one graph, its adjacency, pixel value, 3 different target value and mask which we exclude 2 pixel borderline.
```
from utils import TwoDGrid
dataset = TwoDGrid(root='dataset/2Dgrid', pre_transform=None)
print(len(dataset))
print(dataset[0])

1
Data(edge_index=[2, 358685], m=[9025, 1], x=[9025, 1], y=[9025, 3])
```

### BandClass
Another introduced dataset is BandClass dataset, which consist of 3K train, 1K validation and 1K test planar graph, each has 200 nodes. Problem is binary classification problem where the ground truth of classes were determined by the frequency on the graph. You can visualize two sample graph using gspbox library by following Matlab script.

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

You can read this dataset using Pytorch-geometric by following Python script.
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

Sample 2 graph in this dataset can be found in following figure.
![Sample image](images/mnist46.jpg?raw=true "Title")

The dataset can be load using Pytorch-geometric by following script. Given sample transform can add node degree and node coordinate to the pixel value as a node features. However we do not use superpixel coordinate in our analysis in order to make the problem harder and more realiztic in terms of graph research. 

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

## MNIST Dataset Results

Briefly launch the mnist_code.py script. You set the method selection hard coded. You can select your method and launch the script.  

	python mnist_code.py
	

## MNIST Dataset Results

Briefly launch the appendix_I.py script. You set the method selection hard coded. You can select your method and launch the script.

	python appendix_I.py

## BandClass Dataset Results

Briefly launch the appendix_J.py script. You set the method selection hard coded. You can select your method and launch the script.

	python appendix_J.py


## Requirements
These libraries versions are not stricly needed. But these are the configurations in our test machine. Also all dependencies of pytorch-geometric are needed.
- Python==3.8.5
- pytorch==1.5.1
- pytorch_geometric==1.6.1
- numpy==1.19.1
- scipy==1.5.2
- matplotlib==3.3.1


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
