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

        request = NearestNeighborsRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return NearestNeighborsFe(
            x,
            y if not y else None,
            **request.kwargs
        )
    
    def KDTreeTrigger(
            self,
            request, 
            context
        ) -> KDTreeReply:

        request = KDTreeRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return KDTreeFe(
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
    
    def NearestCentroidTrigger(
            self,
            request, 
            context  
        ) -> NearestCentroidReply:

        request = NearestCentroidRequest(request)

        dataSet = self.manager.getDataSet()

        x, y = self.manager.gerateTrainData(
            dataSet.drop(request.x_drop_data, axis=1), 
            dataSet[request.y_drop_data], 
            test_size=request.size, 
            random_state=request.random,
            key = request.key
        )

        return NearestCentroidFe(
            x,
            y,
            request.metric,
            request.shrink_threshold
        )




