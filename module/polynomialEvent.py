from server_manager import ServerManager
from lib.feature.polynomialFeatures import (
    PolynomialFe,
    PolynomialTransformFe
)
from lib.proto.py.polynomial_features_pb2 import (
    PolynomialService,
    PolynomialFeaturesReply,
    PolynomialFeaturesRequest,
    PolynomialFeaturesFitTransformReply,
    PolynomialFeaturesFitTransformRequest
)

class PolynomialEvent(PolynomialService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def PolynomialFeaturesTrigger(
            self, 
            request,
            context
        ) -> PolynomialFeaturesReply:

        if not isinstance(request, PolynomialFeaturesRequest):
            raise ValueError("Invalid request. Expected PolynomialFeaturesRequest.")

        request = PolynomialFeaturesRequest(request)

        response = PolynomialFeaturesReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = PolynomialFeaturesReply(
            PolynomialFe(
                x,
                y if not y else None,
                request.degree,
                request.interaction_only,
                request.include_bias,
                request.order,
                **request.kwargs
            )
        )

        return response
    
    def PolynomialFeaturesFitTransformTrigger(
            self, 
            request,
            context
        ) -> PolynomialFeaturesFitTransformReply:

        if not isinstance(request, PolynomialFeaturesFitTransformRequest):
            raise ValueError("Invalid request. Expected PolynomialFeaturesFitTransformRequest.")

        request = PolynomialFeaturesFitTransformRequest(request)

        response = PolynomialFeaturesFitTransformReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = PolynomialFeaturesFitTransformReply(
            PolynomialTransformFe(
                x,
                y if not y else None,
                request.sample_weight,
                request.degree,
                request.interaction_only,
                request.include_bias,
                request.order,
                request.coef_init,
                request.intercept_init,
                **request.kwargs
            )
        )

        return response
