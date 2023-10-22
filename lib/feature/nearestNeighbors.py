from sklearn.neighbors import (
    NearestNeighbors,
    KDTree,
    NearestCentroid
)

def NearestNeighborsFe(
        xMatrixLike,
        yMatrix = None,
        **kwargs
    )-> NearestNeighbors:

    return NearestNeighbors(
        **kwargs
    ).fit(
        xMatrixLike,
        yMatrix
    )

def KDTreeFe(
        xMatrixLike,
        k = 1,
        leaf_size = 40,
        metric = "minkowski",
        sample_weight = None,
        return_distance = True,
        dualtree = False,
        breadth_first = False,
        sort_results = True,   
        **kwargs
    )-> KDTree:

    kdt = KDTree(
        xMatrixLike,
        leaf_size,
        metric,
        sample_weight, 
        **kwargs
    )

    return kdt.query(
        xMatrixLike, 
        k, 
        return_distance,
        dualtree,
        breadth_first,
        sort_results
    )

def NearestCentroidFe(
        xMatrixLike, 
        yMatrix,
        metric = "euclidean",
        shrink_threshold = None
    )-> NearestCentroid:

    return NearestCentroid(
        metric,
        shrink_threshold
    ).fit(
        xMatrixLike,
        yMatrix
    )
