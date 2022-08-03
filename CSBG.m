% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function [outlabel,BiGraph,isCov,objective,laMM]=CSBG(X,class,Center,k_nearest,numview,W,Q)
% X:Samples
% class:Number of clusters 
% Center:Subcluster center
% k_nearest:Number of neighbors
% numview: Number of views
% W:The weight of subclusters in each view
% Q：Instrumental variable

NITER=30;% Number of subiterations
zr=10e-5;
N_sample=size(X{1},2);
M_subcluster=size(Center{1},2);

[Z,Alpha,MultiDis,id]=ConstructANp(X,Center,k_nearest,numview,W,Q);
% Z: Connection probability matrix
% Alpha: parameter
% MultiDis: Sum of weighted distances
% id: Coordinates of the leading k_nearest nearest subclusters for each sample
[ZT,AlphaT,MultiDisT,idT]=ConstructANp(Center,X,k_nearest,numview,W,Q);
Z0=(Z+ZT')/2; % Initialize the connection probability matrix

alpha=mean(Alpha);
alphaT=mean(AlphaT);
lambda=(alpha+alphaT)/2;% Initialize lambda parameter
[BiGraph,U,V,eigval,D1,D2]=svd2uv(Z0,class);% Initialize the matrix F=[U;V]
if sum(eigval(1:class)) > class*(1-zr)
    error('The original graph has more than %d connected component£¬ Please set knearest larger', class);
end

dxi = zeros(N_sample,k_nearest);
for i = 1:N_sample
    dxi(i,:) = MultiDis(i,id(i,:));
end
dxiT = zeros(M_subcluster,k_nearest);
for i = 1:M_subcluster
    dxiT(i,:) = MultiDisT(i,idT(i,:));
end
Ater=0;
for iter = 1:NITER % The connection probability matrix Z and matrix F=[U;V] are updated iteratively until matrix Z has (class) connected components. 
    U1 = D1*U;
    V1 = D2*V;
    %% ------Update the connection probability matrix
    dist = sqdist(U1',V1');  % only local distances need to be computed. speed will be increased using C
    tmp1 = zeros(N_sample,k_nearest); 
    for i = 1:N_sample
        dfi = dist(i,id(i,:));
        ad = -(dxi(i,:)+lambda*dfi)/(2*alpha);   
        tmp1(i,:) = EProjSimplex_new(ad);
    end
    Z = sparse(repmat([1:N_sample],1,k_nearest),id(:),tmp1(:),N_sample,M_subcluster);% Converted to a sparse connection probability matrix 

    tmp2 = zeros(M_subcluster,k_nearest);
    for i = 1:M_subcluster
        dfiT = dist(idT(i,:),i);
        ad =  (dxiT(i,:)-0.5*lambda*dfiT'); 
        tmp2(i,:) = EProjSimplex_new(ad);
    end 
    ZT = sparse(repmat([1:M_subcluster],1,k_nearest),idT(:),tmp2(:),M_subcluster,N_sample); 
    BiGraph = (Z+ZT')/2;
    
    U_old = U;
    V_old = V;
    %% -------Update matrix F=[U;V]
    [BiGraph, U, V, eigval, D1, D2] = svd2uv(BiGraph, class);
    %% -------Determine whether the subiteration converges
    fn1 = sum(eigval(1:class));
    fn2 = sum(eigval(1:class+1));
    if fn1 < class-zr % the number of block is less than (class)
        Ater=0;
        lambda = 2*lambda;
    elseif fn2 > class+1-zr % the number of block is more than (class)
        Ater = 0;
        lambda= lambda/2;  
        U = U_old; V = V_old;
    else
        Ater=Ater+1;
        if(Ater==2)
            break;
        end
    end
end
fprintf('csbg loop:%d\n',iter)
laMM=id(:,1);
[clusternum, outlabel] = struG2la(BiGraph);% Compute the connected components of a bipartite graph
% clusternum: Number of connected components
% outlabel: Clustering results
if clusternum ~=  class
    sprintf('Can not find the correct cluster number: %d', class)
end
isCov=[Ater==2];
%-------Objective Function
objective = loss(MultiDis,BiGraph,alpha,lambda,U,V);
end

