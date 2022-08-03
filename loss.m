function objective = loss(MultiDis,Z,alpha,lambda,U,V )
    Nsample = size(Z,1);
    Msubcluster = size(Z,2);
    a1 = sum(Z,2);
    a2 = sum(Z,1);
    st = sum(sum(MultiDis.*Z));
    at = alpha*sum(sum(Z.^2));
    Da = spdiags( [ 1./sqrt(a1) ;1./sqrt(a2')],0,Nsample+Msubcluster,Nsample+Msubcluster);
    SS = sparse(Nsample+Msubcluster,Nsample+Msubcluster); 
    SS(1:Nsample,Nsample+1:end) = Z; 
    SS(Nsample+1:end,1:Nsample) = Z';
    ft = lambda*trace([U; V]'*(eye(Nsample+Msubcluster)-Da*SS*Da )*[U;V]);
    objective = st+ at + ft;
end