from server_manager import ServerManager
from lib.feature.svmExpressions import (
    LinearSVCFe,
    LinearSVRFe,
    SVCFe
)

from lib.proto.py.svm_expression_pb2 import (
    SVMService,
    LinearSVCReply,
    LinearSVCRequest,
    LinearSVRReply,
    LinearSVRRequest,    
    SVCReply,
    SVCRequest
)

class SVMEvents(SVMService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def LinearSVCTrigger(
            self, 
            request, 
            context
        ) -> LinearSVCReply:

        if not isinstance(request, LinearSVCRequest):
            raise ValueError("Invalid request. Expected LinearSVCRequest.")

        request = LinearSVCRequest(request)

        response = LinearSVCReply()

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

        model = LinearSVCFe(
            x,
            y,
            **request.kwargs
        )

        response = LinearSVCReply(
            n_iter_ = model.n_iter_,
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            classes_ = model.classes_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response
    
    def LinearSVRTrigger(
            self, 
            request, 
            context
        ) -> LinearSVRReply:

        if not isinstance(request, LinearSVRRequest):
            raise ValueError("Invalid request. Expected LinearSVRRequest.")

        request = LinearSVRRequest(request)

        response = LinearSVRReply()

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

        model = LinearSVRFe(
            x,
            y,
            **request.kwargs
        )

        response = LinearSVRReply(
            n_iter_ = model.n_iter_,
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            intercept_ = model.intercept_,
            coef_ = model.coef_
        )

        return response
    
    def SVCTrigger(
            self, 
            request, 
            context
        ) -> SVCReply:

        if not isinstance(request, SVCRequest):
            raise ValueError("Invalid request. Expected SVCRequest.")

        request = SVCRequest(request)

        response = SVCReply()

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

        model = SVCFe(
            x,
            y,
            **request.kwargs
        )   

        response = SVCReply(
            shape_fit_ = model.shape_fit_,
            probB_ = model.probB_,
            probA_ = model.probA_,
            n_support_ = model.n_support_,
            support_vectors_ = model.support_vectors_,
            support_ = model.support_,
            n_iter_ = model.n_iter_,
            feature_names_in_ = model.feature_names_in_,
            n_features_in_ = model.n_features_in_,
            intercept_ = model.intercept_,
            fit_status_ = model.fit_status_,
            dual_coef_ = model.dual_coef_,
            coef_ = model.coef_,
            classes_ = model.classes_,
            class_weight_ = model.class_weight_
        )

        return response
    
    


