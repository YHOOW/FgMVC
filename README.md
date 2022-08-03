# Fine-grained multi-view clustering with robust multi-prototypes representation (FgMVC)
Matlab implementation of Fine-grained multi-view clustering with robust multi-prototypes representation (FgMVC)

This package is implementing the method in Applied Intelligence 2022 paper: Fine-grained multi-view clustering with robust multi-prototypes representation. 
(Please cite this paper) Please contact 02713@zjhu.edu.cn if you have any questions. 

Thanks a lot for the relevant code provided by Chenglong Wang and Jinglin Xu, and this work is inspired by K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters (KDD2019).   
The updateC.m and pdist2.m functions are derived from the open source code of Multi-View k-means Clustering with AdaptiveSparse Memberships and Weight Allocation (TKDE2020). 

To show my respect for these studies.

# Short demo
main.m: you can run this main file, and Its performance will be recorded automatically. 

# Dataset: Handwritten
Here you can run the main.m function get Handwritten clustering results. 

# Parameters: 
k_nearest: Number of neighbors
M_subcluster: Number of subclusters

If you want to obtain clustering results for other datasets, you need to set these two parameters.

# Reference:
If you find this code useful in your research, please cite the paper.

Yin, H., Wang, G., Hu, W. et al. Fine-grained multi-view clustering with robust multi-prototypes representation. Appl Intell (2022). https://doi.org/10.1007/s10489-022-03898-2
