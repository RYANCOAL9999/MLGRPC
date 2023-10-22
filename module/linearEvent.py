# import statsmodels.api as sm

from server_manager import ServerManager
from lib.feature.LinearExpression import (
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

        request = LinearRegressionRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        # ols_m = sm.OLS(y, sm.add_constant(x)).fit()
        
        # ols_m.summary()

        return LinearFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )
    
    def LinearRidgeTrigger(
            self, 
            request, 
            context
        ) -> LinearRidgeReply:

        request = LinearRegressionRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return RidgeFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )
    
    def LinearRidgeCVTrigger(
            self, 
            request, 
            context
        ) -> LinearRidgeCVReply:

        request = LinearRidgeCVRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return RidgeCVFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )
    
    def LassoExpressionTrigger(
            self, 
            request, 
            context 
        ) -> LassoExpressionReply:

        request = LassoExpressionRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )
        
        return LassoFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )
    
    def LassoLarsLassoExpressionTrigger(
            self, 
            request, 
            context   
        ) -> LassoLarsLassoExpressionReply:

        request = LassoLarsLassoExpressionRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return LassoLarsFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

    def BayesianRidgeTrigger(
            self, 
            request, 
            context 
        ) -> BayesianRidgeReply:

        request = BayesianRidgeRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return BayesianRidgeFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )

    def TweedieRegressorTrigger(
            self, 
            request, 
            context 
        ) -> TweedieRegressorReply:

        request = TweedieRegressorRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return TweedieRegressorFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )

    def SGDClassifierTrigger(
            self, 
            request, 
            context 
        ) -> SGDClassifierReply:

        request = SGDClassifierRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return SGDClassifierFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )
    
    def ElasticNetTrigger (
            self, 
            request, 
            context 
        ) -> ElasticNetReply:

        request = ElasticNetRequest(request)

        return ElasticNetFe(
            None,
            None,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )
