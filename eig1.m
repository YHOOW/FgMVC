% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function [eigvec, eigval, eigval_full] = eig1(A, class, isMax, B)
% The optimal solution F to the problem is formed by the c eigenvectors of
% L corresponding to the (class) smallest eigenvalues
%-------------------------------------------------------------------------------------
if nargin < 2
    class = size(A,1);
    isMax = 1;
elseif class > size(A,1)
    class = size(A,1);
end
%-------------------------------------------------------------------------------------
if nargin < 3
    isMax = 1;
end

if nargin < 4
    B = eye(size(A,1));
end

A = (A+A')/2;  

[v d] = eig(A,B);
d = diag(d);
d = abs(d);
%-------------------------------------------------------------------------------------
if isMax == 0
    [d1, idx] = sort(d );
%--------------------------------------------------------------------------------------
else
    [d1, idx] = sort(d,'descend');
end

idx1 = idx(1:class);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);