# Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective

This repository consist of ICLR2021 paper on theoretical and empirical spectral analysis of GNNs

## Theoretical and Empirical Result of GNNs
Studied GNN's spectral analysis codes are in spectral_analysis_result folder. Those codes were written in Matlab.
You can briefly run each script for each method independently and see the empirical and theoretical frequency responses on Cora graph.
For instance, Chebnet's frequency response
```
>cayley_spectral_analysis
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


## MNIST

https://graphics.cs.tu-dortmund.de/fileadmin/ls7-www/misc/cvpr/mnist_superpixels.tar.gz
