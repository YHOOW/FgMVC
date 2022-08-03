% Ref: 
% [1] K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
% [2] Multi-view K-Means Clustering with AdaptiveSparse Memberships and Weight Allocation.(TKDE2020)
% https://github.com/xujinglin/MVASM

% Matlab implementation of Fine-grained multi-view clustering with robust
% multi-prototypes representation (FgMVC)

clear ; clc; close all;
%% load data
load handwritten.mat;
groundtruth=Y;
class=numel(unique(groundtruth));  

numview = length(X);    n=size(Y,1);
%--------- parameters ---------%
M_subcluster=150;  % Number of sub-clusters 
k_nearest=5;       % Number of neighbors
%---------Data Normalization-----%
for v=1:numview
    X{v}=zscore(X{v}')';
end
%%
%---------FgMVC------------%
[label_out,Obj,W,BiGraph,Center,laMM]=FgMVC(X,class,M_subcluster, k_nearest,numview);
%--------- results ----------%
[ACC,NMI,purity]=ClusteringMeasure(Y,label_out)       




