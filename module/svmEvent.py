from server_manager import ServerManager
from lib.feature.svmExpression import (
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

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LinearSVCReply(
            LinearSVCFe(
                x,
                y,
                **request.kwargs
            )
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

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = LinearSVRReply(
            LinearSVRFe(
                x,
                y,
                **request.kwargs
            )
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

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = SVCReply(
            SVCFe(
                x,
                y,
                **request.kwargs
            )   
        )

        return response
    
    


