# import statsmodels.api as sm
from server_manager import ServerManager
from lib.feature.linearExpressions import (
    bayesianRidgeFe,
    elasticNetFe,
    lassoFe,
    lassoLarsFe,
    linearFe, 
    ridgeFe, 
    ridgeCVFe,
    sgdClassifierFe,
    tweedieRegressorFe
)
from lib.proto.py.linear_expression_pb2_grpc import LinearService
from lib.proto.py.linear_expression_pb2 import (
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

class LinearEvents(LinearService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager
    
    def LinearRegressionEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        # ols_m = sm.OLS(y, sm.add_constant(x)).fit()
        
        # ols_m.summary()

        model = linearFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )

        score = self.manager.showScore(x, y, model)

        y_predict = self.manager.showPredict(model, x_test)

        meanAbsoluteError = self.manager.showMeanAbsoluteError(y_test, y_predict)

        response = LinearRegressionReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            intercept_ = model.intercept_,
            singular_ = model.singular_,
            rank_ = model.rank_,
            coef_ = model.coef_,
            score = score,
            predict = y_predict,
            meanAbsoluteError = meanAbsoluteError
        )

        return response
    
    def LinearRidgeEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = ridgeFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

        response = LinearRidgeReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            n_iter = model.n_iter_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response
    
    def LinearRidgeCVEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = ridgeCVFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

        response = LinearRidgeCVReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            best_score_ = model.best_score_,
            alpha = model.alpha,
            intercept_ = model.intercept_,
            coef_ = model.coef_,
            cv_values_ = model.cv_values_,
        )

        return response

    
    def LassoExpressionEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = lassoFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

        response = LassoExpressionReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            n_iter_ = model.n_iter_,
            intercept_ = model.intercept_,
            sparse_coef_ = model.sparse_coef_,
            dual_gap_ = model.dual_gap_,
            coef_ = model.coef_,
        )
        
        return response
    
    def LassoLarsLassoExpressionEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = lassoLarsFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

        response = LassoLarsLassoExpressionReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            n_iter_ = model.n_iter_,
            coef_ = model.coef_,
            coef_path_ = model.coef_path_,
            active_ = model.active_,
            alphas_ = model.alphas_
        )

        return response

    def BayesianRidgeEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = bayesianRidgeFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )

        response = BayesianRidgeReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            x_scale_ = model.X_scale_,
            x_offset_ = model.X_offset_,
            n_iter_ = model.n_iter_,
            score_ = model.scores_,
            simga_ = model.sigma_,
            lambda_ = model.lambda_,
            alpha_ = model.alpha_,
            intercept_ = model.intercept_,
            coef = model.coef_,
        )

        return response

    def TweedieRegressorEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = tweedieRegressorFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )
        
        response = TweedieRegressorReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            n_iter_ = model.n_iter_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response

    def SGDClassifierEvent(
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = sgdClassifierFe(
            x,
            y,
            request.sample_weight,
            **request.kwargs
        )

        response = SGDClassifierReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            t_ = model.t_,
            classes_ = model.classes_,
            loss_function_ = model.loss_function_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response
    
    def ElasticNetEvent (
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

        x_train, y_train, x_test, y_test = self.manager.generateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random
        )

        x, y = self.manager.chooseDictData(
            request.key, 
            x_train, 
            y_train, 
            x_test, 
            y_test
        )

        model = elasticNetFe(
            x,
            y,
            request.alpha,
            request.sample_weight,
            **request.kwargs
        )

        response = ElasticNetReply(
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            dual_gap_ = model.dual_gap_,
            n_iter_ = model.n_iter_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response
