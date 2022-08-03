% Ref:
% K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.(KDD2019)
% https://github.com/CHLWR/KDD2019_K-Multiple-Means
function [Z, U, V, eigval, D1z, D2z] = svd2uv(Z, class)
    [N_sample,M_subcluster] = size(Z);
    ver=version;
    if(str2double(ver(1:3))>=9.1)
        Z = Z./sum(Z,2);
        % for the newest version(>=9.1) of MATLAB
%---------------------------------------------------------------------------------------
    else
        Z = bsxfun(@rdivide, Z, sum(Z,2));% for old version(<9.1) of MATLAB
%--------------------------------------------------------------------------------------
    end
    z1 = sum(Z,2); 
    D1z = spdiags(1./sqrt(z1),0,N_sample,N_sample); 
    z2 = sum(Z,1); 
    D2z = spdiags(1./sqrt(z2'),0,M_subcluster,M_subcluster);
    Z1 = D1z*Z*D2z;
    LZ = full(Z1'*Z1);
    [V, eigval, ~]=eig1(LZ,class+1);
    V = V(:,1:class);
    U = (Z1*V)./(ones(N_sample,1)*sqrt(eigval(1:class))');

    U = sqrt(2)/2*U; 
    V = sqrt(2)/2*V;

end
