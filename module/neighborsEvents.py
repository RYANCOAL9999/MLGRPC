from server_manager import ServerManager

from lib.feature.NearestNeighbors import (
    NearestNeighborsFe,
    KDTreeFe,
    NearestCentroidFe
)

from lib.proto.py.nearest_neighbors_pb2 import (
    NeighborsService,
    NearestNeighborsReply,
    NearestNeighborsRequest,
    KDTreeReply,
    KDTreeRequest,
    NearestCentroidReply,
    NearestCentroidRequest
)

class NeighborsEvents(NeighborsService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def NearestNeighborsTrigger(
            self, 
            request, 
            context
        ) -> NearestNeighborsReply:

        if not isinstance(request, NearestNeighborsRequest):
            raise ValueError("Invalid request. Expected NearestNeighborsRequest.")

        request = NearestNeighborsRequest(request)

        response = NearestNeighborsReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = NearestNeighborsReply(
            NearestNeighborsFe(
                x,
                y if not y else None,
                **request.kwargs
            )
        )

        return response
    
    def KDTreeTrigger(
            self,
            request, 
            context
        ) -> KDTreeReply:

        if not isinstance(request, KDTreeRequest):
            raise ValueError("Invalid request. Expected KDTreeRequest.")

        request = KDTreeRequest(request)

        response = KDTreeReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = KDTreeReply(
            KDTreeFe(
                x,
                request.k,
                request.leaf_size,
                request.metric,
                request.sample_weight,
                request.return_distance,
                request.dualtree,
                request.breadth_first,
                request.sort_results,   
                **request.kwargs
            )
        )

        return response
    
    def NearestCentroidTrigger(
            self,
            request, 
            context  
        ) -> NearestCentroidReply:

        if not isinstance(request, NearestCentroidRequest):
            raise ValueError("Invalid request. Expected NearestCentroidRequest.")

        request = NearestCentroidRequest(request)

        response = NearestCentroidReply()

        if not self.manager.reversed(): return response

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        response = NearestCentroidReply(
                NearestCentroidFe(
                x,
                y,
                request.metric,
                request.shrink_threshold
            )
        )

        return response




