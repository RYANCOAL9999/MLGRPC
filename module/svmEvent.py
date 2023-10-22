from server_manager import ServerManager
from lib.feature.SVMExpression import (
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

        request = LinearSVCRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return LinearSVCFe(
            x,
            y,
            **request.kwargs
        )
    
    def LinearSVRTrigger(
            self, 
            request, 
            context
        ) -> LinearSVRReply:

        request = LinearSVRRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return LinearSVRFe(
            x,
            y,
            **request.kwargs
        )
    
    def SVCTrigger(
            self, 
            request, 
            context
        ) -> SVCReply:

        request = SVCRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return SVCFe(
            x,
            y,
            **request.kwargs
        )
    
    


