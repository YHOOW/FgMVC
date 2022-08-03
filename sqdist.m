% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.*a,1); bb = sum(b.*b,1); ab = a'*b; 
d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

