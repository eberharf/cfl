# Some Notes on Clustering

## Available Clustering Options 

- Any clustering method that comes from the [Scikit-learn clustering module](https://scikit-learn.org/stable/modules/clustering.html) 
- Shared Nearest Neighbor clustering (implemented by us inside of the `cfl.clustering_methods` submodule) 
- Any other module that uses the same interface as a Scikit-learn clusterer

## Recommendations for Choosing a Clusterer 

(eventually, we will have some tools available to help users automatically select the clustering method/parameters for their dataset, but those do not exist yet)

DBSCAN and KMeans are the two clustering methods we've worked with the most, so unless you have a reason to choose another, maybe stick with one of those.

KMeans
    - Advantages: only one parameter to tune, meaning of parameter (# of clusters) is pretty intuitive 
    - Potential Disadvantages: can only detect globular clusters, forces the user to choose the number of clusters (a goal of CFL is to detect number of macrovariables without supervision)

DBSCAN 
    - Advantages: does not force you to pre-define number of clusters, can detect clusters of any shape, you can maybe get away with only tuning one parameter (eps)
    - Disadvantages: has two  parameters (eps and min_samples) (even though eps is more important to tune than min_samples), can be tricky to tune well, may not correctly distinguish two clusters if there is overlap between the clusters 

---
**NOTE**

Shared Nearest Neighbor (SNN) clustering is a derivative of DBSCAN designed to perform well on high dimensional data

---
