# Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective

This repository consist of ICLR2021 paper on theoretical and empirical spectral analysis of GNNs

## Theoretical and Empirical Frequency Responses
Studied GNN's spectral analysis codes are in spectral_analysis_result folder. Those codes were written in Matlab.
You can briefly run each script for each method independently and see the empirical and theoretical frequency responses on Cora graph.
For instance, Chebnet's frequency response
```
>chebnet_spectral_analysis
```

![Sample image](images/cheb.jpg?raw=true "Title")

GIN's frequency response
```
>gin_spectral_analysis
```
![Sample image](images/gin.jpg?raw=true "Title")

## Datasets
In this research, we introduced two different dataset. One named 2D-grid graph consist of 95x95 resolution and 4-neighborhood regular grid graph. Each nodes refers the pixel in the image. We prepared low-pass, high-pass and band-pass filtering results as output of filter learning task.
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

Last, we used MNIST superpixel dataset in our analysis. You can download and extract the "train.pt" and "test.pt" files into dataset/MNIST/raw/ folder. Here is the most recent link of Mnist-75.
https://graphics.cs.tu-dortmund.de/fileadmin/ls7-www/misc/cvpr/mnist_superpixels.tar.gz

Sample 2 graph in this dataset can be found in following figure.
![Sample image](images/mnist46.jpg?raw=true "Title")


