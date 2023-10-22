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

class PolynomialEvents(PolynomialService):

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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(request.key, x_train, y_train, x_test, y_test)

        model = PolynomialFe(
            x,
            y if not y else None,
            request.degree,
            request.interaction_only,
            request.include_bias,
            request.order,
            **request.kwargs
        )

        response = PolynomialFeaturesReply(
            matrix = model
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(request.key, x_train, y_train, x_test, y_test)

        model = PolynomialTransformFe(
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

        response = PolynomialFeaturesFitTransformReply(
            t_ = model.t_,
            n_iter_ = model.n_iter_,
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            loss_function_ = model.loss_function_,
            intercept_ = model.intercept_,
            coef_ = model.coef_,
            classes_ = model.classes_
        )

        return response
