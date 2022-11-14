## Incorporating Dynamicity of Transportation Network with Multi-Weight Traffic Graph Convolutional Network for Traffic Forecasting (Shin and Yoon, 2020)

![image](https://user-images.githubusercontent.com/31876093/141113772-9e008ac3-1bcd-476f-a450-aaf3112aeeff.png)

This is a PyTorch implmentation of Multi-Weight Traffic Graph Conovlutional (MW-TGC) Network in the following paper:

> Yuyol Shin, Yoonjin Yoon. 2020. Incorporating Dynamicity of Transportation Network with Multi-Weight Traffic Graph Convolutional Network for Traffic Forecasting. IEEE Transactions on Intelligent Transportation Systems. ![link](https://ieeexplore.ieee.org/abstract/document/9239873)


## The Dataset
We used speed data of Seoul, South Korea to test the performance of the proposed model. 
The original dataset is obtained from TOPIS ![link](https://topis.seoul.go.kr/)
We processed the dataset and defined two study areas - (1) _urban-core_, and (2) _urban-mix_
_Urban-core_, located in Gangnam-gu, Seoul, is populated with rather homogeneous 304 links.
_Urban-mix_ is an expansion of _Urban-core_, and it is populated with heterogeneous 1,007 links.

![image](https://user-images.githubusercontent.com/31876093/141130819-768855b5-e9b3-4533-b3b6-ef348c687b3d.png)


## Experimental Results 
The model showed an increased performance compared to state-of-the-art models in both datasets. 
Note that the performance gain was larger in _Urban-mix_ than _Urban-core_

![image](https://user-images.githubusercontent.com/31876093/141131113-fb60b85e-5256-4ddd-a661-b8bcacdbbf20.png)

![image](https://user-images.githubusercontent.com/31876093/141131237-9c356c69-b94f-4910-acf5-a2b2ed6b888a.png)


 
