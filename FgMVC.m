function [outlabel,outObjective,W,BiGraph,Center,laMM]=FgMVC(X,class,M_subcluster,k_nearest,numview)
% Input:
% X: Samples
% class: Number of clusters 
% M_subcluster: Number of subclusters
% k_nearest: Number of neighbors
% numview: Number of views

% Output:
% outlabel: Clustering results
% outObjective: Objective function value
% W: Weight
% BiGraph: Bipartite graph (connection probability matrix)
% Center: Subcluster center
% laMM: Labels that partition samples into subclusters

%% --------initialization
maxIter=15; % Maximum iterations
for p = 1:numview
    %-----Initialize Weight
    W{p}=ones(1,M_subcluster)/numview;
    %-----Initialize subcluster center
    [~,k_center] = kmeans(X{p}', M_subcluster, 'emptyaction', 'singleton', 'replicates', 1, 'display', 'off');
    Center{p} = k_center';
end
for p=1:numview %-----Initialize auxiliary variable Q
    DCC{p}=pdist2(Center{p}',Center{p}');
    for j=1:M_subcluster
        Q{p}(j)=1/(sum(DCC{p}(j,:))+eps);
    end
end
%%
%----Update the bipartite graph and determine whether there is (class) connected components
[outlabel,BiGraph,isCov,objective,laMM]=CSBG(X,class,Center,k_nearest,numview,W,Q);
% isCov: Determine whether the bipartite graph has (class) connected components
iter=0;
OBJ=[];
while(iter<maxIter)
    iter=iter+1;
    if isCov
        OBJ=[OBJ objective];
        objective_old=objective;
        %% -------Update subcluster center
        for p = 1:numview
            Center{p} = updateC(X{p},BiGraph');
        end
        %% ------Update weight
        for p=1:numview  
            DCC{p}=pdist2(Center{p}',Center{p}');
        end
        for j=1:M_subcluster
            idx=find(BiGraph(:,j)~=0);
            for p=1:numview
                  Disxc{p}=sum(pdist2(X{p}(:,idx)',Center{p}(:,j)').*full(BiGraph(idx,j)));
            end
            for p=1:numview
                disc=sum(DCC{p}(j,:));
                %------Update auxiliary variable
                Q{p}(j)=1/(disc+eps);
                %------Update weight
                W{p}(j)=0.5 * sqrt(disc)/(sqrt(Disxc{p})+eps);
            end 
        end
        %% -------Normalize Weight       
           for j=1:M_subcluster
               ww=0;
               for p=1:numview
                   ww=ww+W{p}(j);
               end
               for v=1:numview  
                   W{v}(j)=W{v}(j)/(ww+eps);   
               end
           end
        %% ----Update bipartite graph
        [outlabel,BiGraph,isCov,objective,laMM]=CSBG(X,class,Center,k_nearest,numview,W,Q);
        %% ----Convergence
        if abs(objective_old-objective)<1e-6
            fprintf('converge\n')
            break;
        end
    else
        %----If bipartite graph has no (class) connected components
        for p = 1:numview %-----Reinitialize the subcluster center
            [~,k_center] = kmeans(X{p}', M_subcluster, 'emptyaction', 'singleton', 'replicates', 1, 'display', 'off');
            Center{p} = k_center';
        end
        [outlabel,BiGraph,isCov,objective,laMM]=CSBG(X,class,Center,k_nearest,numview,W,Q);   
    end
    fprintf('loop:%d\n',iter)
end
outObjective = OBJ;
end