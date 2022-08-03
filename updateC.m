%Ref:
% Multi-view K-Means Clustering with AdaptiveSparse Memberships and Weight Allocation.(TKDE2020)
% https://github.com/xujinglin/MVASM
function Center = updateC(X,Z)
% X:Samples
% Z:Connection probability matrix of samples and subcluster centers
[M_subcluster,~] = size(X);
Cup = X * Z';          
Cdown = sum(Z,2);     
Center = Cup./repmat(Cdown',M_subcluster,1); 
end