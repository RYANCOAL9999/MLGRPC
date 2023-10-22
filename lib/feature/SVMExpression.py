from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    SVC
)

def LinearSVCFe(
        xMatrixLike:any, 
        yMatrix:any, 
        sample_weight = None,  
        **kwargs
    )->LinearSVC:

    return LinearSVC(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix, 
        sample_weight
    )

def LinearSVRFe(
        xMatrixLike, 
        yMatrix,
        sample_weight = None,  
        **kwargs
    )->LinearSVR:

    return LinearSVR(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix, 
        sample_weight
    )

def SVCFe(
        xMatrixLike, 
        yMatrix,
        sample_weight = None,  
        **kwargs
    )->SVC:

    return SVC(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix, 
        sample_weight
    )
