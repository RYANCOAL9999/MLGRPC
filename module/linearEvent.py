# import statsmodels.api as sm

from server_manager import ServerManager
from lib.feature.linearExpression import (
    BayesianRidgeFe,
    ElasticNetFe,
    LassoFe,
    LassoLarsFe,
    LinearFe, 
    RidgeFe, 
    RidgeCVFe,
    SGDClassifierFe,
    TweedieRegressorFe
)
from lib.proto.py.linear_expression_pb2 import (
    LinearService,
    LinearRegressionReply,
    LinearRegressionRequest,
    LinearRidgeReply,
    LinearRegressionRequest,
    LinearRidgeCVReply,
    LinearRidgeCVRequest,
    LassoExpressionReply,
    LassoExpressionRequest,
    LassoLarsLassoExpressionReply,
    LassoLarsLassoExpressionRequest,
    BayesianRidgeReply,
    BayesianRidgeRequest,
    TweedieRegressorReply,
    TweedieRegressorRequest,
    SGDClassifierReply,
    SGDClassifierRequest,
    ElasticNetReply,
    ElasticNetRequest
)

class LinearEvent(LinearService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager
    
    def LinearRegressionTrigger(
            self, 
            request, 
            context
        ) -> LinearRegressionReply:

        if not isinstance(request, LinearRegressionRequest):
            raise ValueError("Invalid request. Expected LinearRegressionRequest.")

        request = LinearRegressionRequest(request)

        response = LinearRegressionReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        # ols_m = sm.OLS(y, sm.add_constant(x)).fit()
        
        # ols_m.summary()

        response = LinearRegressionReply(
            LinearFe(
                x,
                y,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response
    
    def LinearRidgeTrigger(
            self, 
            request, 
            context
        ) -> LinearRidgeReply:

        if not isinstance(request, LinearRegressionRequest):
            raise ValueError("Invalid request. Expected LinearRegressionRequest.")

        request = LinearRegressionRequest(request)

        response = LinearRidgeReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LinearRidgeReply(
            RidgeFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response
    
    def LinearRidgeCVTrigger(
            self, 
            request, 
            context
        ) -> LinearRidgeCVReply:

        if not isinstance(request, LinearRidgeCVRequest):
            raise ValueError("Invalid request. Expected LinearRidgeCVRequest.")

        request = LinearRidgeCVRequest(request)

        response = LinearRidgeCVReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LinearRidgeCVReply(
            RidgeCVFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response

    
    def LassoExpressionTrigger(
            self, 
            request, 
            context 
        ) -> LassoExpressionReply:

        if not isinstance(request, LassoExpressionRequest):
            raise ValueError("Invalid request. Expected LassoExpressionRequest.")

        request = LassoExpressionRequest(request)

        response = LassoExpressionReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LassoExpressionReply(
            LassoFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )
        
        return response
    
    def LassoLarsLassoExpressionTrigger(
            self, 
            request, 
            context   
        ) -> LassoLarsLassoExpressionReply:

        if not isinstance(request, LassoLarsLassoExpressionRequest):
            raise ValueError("Invalid request. Expected LassoLarsLassoExpressionRequest.")

        request = LassoLarsLassoExpressionRequest(request)

        response = LassoLarsLassoExpressionReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LassoLarsLassoExpressionReply(
            LassoLarsFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response

    def BayesianRidgeTrigger(
            self, 
            request, 
            context 
        ) -> BayesianRidgeReply:

        if not isinstance(request, BayesianRidgeRequest):
            raise ValueError("Invalid request. Expected BayesianRidgeRequest.")

        request = BayesianRidgeRequest(request)

        response = BayesianRidgeReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = BayesianRidgeReply(
            BayesianRidgeFe(
                x,
                y,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response

    def TweedieRegressorTrigger(
            self, 
            request, 
            context 
        ) -> TweedieRegressorReply:

        if not isinstance(request, LassoLarsLassoExpressionRequest):
            raise ValueError("Invalid request. Expected LassoLarsLassoExpressionRequest.")

        request = TweedieRegressorRequest(request)

        response = TweedieRegressorReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = TweedieRegressorReply(
            TweedieRegressorFe(
                x,
                y,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response

    def SGDClassifierTrigger(
            self, 
            request, 
            context 
        ) -> SGDClassifierReply:

        if not isinstance(request, SGDClassifierRequest):
            raise ValueError("Invalid request. Expected SGDClassifierRequest.")

        request = SGDClassifierRequest(request)

        response = SGDClassifierReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = SGDClassifierReply(
            SGDClassifierFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response
    
    def ElasticNetTrigger (
            self, 
            request, 
            context 
        ) -> ElasticNetReply:

        if not isinstance(request, ElasticNetRequest):
            raise ValueError("Invalid request. Expected ElasticNetRequest.")

        request = ElasticNetRequest(request)

        response = ElasticNetReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = ElasticNetReply(
            ElasticNetFe(
                x,
                y,
                request.alpha,
                request.sample_weight,
                **request.kwargs
            )
        )

        return response
