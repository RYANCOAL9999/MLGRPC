from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron

def PolynomialFe(
        xMatrixLike,
        yMatrix = None,
        degree = 2,
        interaction_only = False,
        include_bias = True,
        order = "C",
        **kwargs
    )-> PolynomialFeatures:

    return PolynomialFeatures(
        degree,
        interaction_only,
        include_bias,
        order, 
    ).fit_transform(
        xMatrixLike,
        yMatrix,
        **kwargs
    )

def PolynomialTransformFe(
        xMatrixLike,
        yMatrix = None,
        sample_weight = None,
        degree = 2,
        interaction_only = False,
        include_bias = True,
        order = "C",
        coef_init = None,
        intercept_init = None,
        **kwargs
    )->Perceptron:

    y = xMatrixLike[:, 0] ^ xMatrixLike[:, 1]

    x = PolynomialFeatures(
        xMatrixLike,
        yMatrix = yMatrix,
        degree = degree,
        interaction_only = interaction_only,
        include_bias = include_bias,
        order = order
    ).astype(int)

    regularExpression = Perceptron(
        **kwargs
    )

    if regularExpression is None: return None
    
    return regularExpression.fit(
        x, 
        y,
        coef_init,
        intercept_init,
        sample_weight
    )
