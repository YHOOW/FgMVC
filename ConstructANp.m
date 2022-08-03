% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function [Z,Alpha,MultiDis, id, tmp]= ConstructANp(A, B, k_nearest, numview,W,Q)
N_sample = size(A{1},2);%Number of samples
M_subcluster=size(B{1},2);%Number of subclusters
MultiDis = zeros(N_sample,M_subcluster);

for p = 1:numview
    Dis{p} = pdist2(A{p}',B{p}');
    if N_sample>M_subcluster
        for j=1:M_subcluster
            Dis{p}(:,j)=W{p}(j)*Q{p}(j)*Dis{p}(:,j);
        end
    elseif N_sample<M_subcluster  % When A is subcluster centers and B is samples
        for i=1:N_sample
            Dis{p}(i,:)=W{p}(i)*Q{p}(i)*Dis{p}(i,:);
        end
    end
end
for p=1:numview
    MultiDis = MultiDis+Dis{p};% Sum of weighted distances
end

di=zeros(N_sample,k_nearest+1);
distXt = MultiDis;
id=di;
for i = 1:k_nearest+1
    [di(:,i),id(:,i)] = min(distXt, [], 2);% Obtain the distance between each sample and the nearest (class) subcluster centers and the subcluster coordinates 
    temp = (id(:,i)-1)*N_sample+[1:N_sample]';% Record the coordinates of the subcluster corresponding to the minimum distance
    distXt(temp) = 1e100; % Find the second smallest distance by assigning a very large value to the minimum distance. 
end
clear distXt temp
id(:,end)=[];
Alpha=0.5*(k_nearest*di(:,k_nearest+1)-sum(di(:,1:k_nearest),2));%Update parameter Alpha adaptively
ver=version;
if(str2double(ver(1:3))>=9.1)
    tmp = (di(:,k_nearest+1)-di(:,1:k_nearest))./(2*Alpha+eps); % for the newest version(>=9.1) of MATLAB
else
    tmp =  bsxfun(@rdivide,bsxfun(@minus,di(:,k_nearest+1),di(:,1:k_nearest)),2*Alpha+eps); % for old version(<9.1) of MATLAB
end
Z = sparse(repmat([1:N_sample],1,k_nearest),id(:),tmp(:),N_sample,M_subcluster);% Converted to a sparse connection probability matrix
return
