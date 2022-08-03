% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function [clusternum, label] = struG2la(Z)
    [n,m] = size(Z);
    SS0 = sparse(n+m,n+m);
    SS0(1:n,n+1:end) = Z; 
    SS0(n+1:end,1:n) = Z';
    %SS0=[0,Z;Z',0]
    [clusternum, label] = graphconncomp(SS0);
    label = label(1:n)';  

end