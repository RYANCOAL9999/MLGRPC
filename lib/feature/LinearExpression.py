from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoLars,
    BayesianRidge,
    TweedieRegressor,
    SGDClassifier,
    ElasticNet
)

def LinearFe(
        xMatrixLike,  
        yMatrix,
        sample_weight = None, 
        **kwargs
    )->LinearRegression:

    return LinearRegression(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def RidgeFe(
        xMatrixLike,  
        yMatrix,
        alpha = 1, 
        sample_weight = None, 
        **kwargs
    )->Ridge:

    return Ridge(
        alpha, 
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def RidgeCVFe(
        xMatrixLike, 
        yMatrix,
        alphas = 1,
        sample_weight = None,  
        **kwargs
    )->RidgeCV:

    return RidgeCV(
        alphas, 
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def LassoFe(
        xMatrixLike, 
        yMatrix,
        alpha = 1, 
        sample_weight = None, 
        **kwargs
    )->Lasso:

    return Lasso(
        alpha, 
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def LassoLarsFe(
        xMatrixLike, 
        yMatrix,
        alpha = 1,
        sample_weight = None,  
        **kwargs
    )->LassoLars:

    return LassoLars(
        alpha, 
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def BayesianRidgeFe(
        xMatrixLike, 
        yMatrix,
        sample_weight = None,  
        **kwargs
    )->BayesianRidge:

    return BayesianRidge(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def TweedieRegressorFe(
        xMatrixLike, 
        yMatrix,
        sample_weight = None,  
        **kwargs
    )->TweedieRegressor:

    return TweedieRegressor(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def SGDClassifierFe(
        xMatrixLike, 
        yMatrix,
        sample_weight = None,  
        **kwargs
    )->SGDClassifier:

    return SGDClassifier(
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

def ElasticNetFe(
        xMatrixLike, 
        yMatrix,
        alpha = 1,
        sample_weight = None,  
        **kwargs
    )->ElasticNet:

    return ElasticNet(
        alpha,
        **kwargs
    ).fit(
        xMatrixLike, 
        yMatrix,
        sample_weight
    )

