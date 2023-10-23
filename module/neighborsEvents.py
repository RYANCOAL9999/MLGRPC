from server_manager import ServerManager
from lib.feature.nearestNeighbors import (
    nearestNeighborsFe,
    kdTreeFe,
    nearestCentroidFe
)
from lib.proto.py.nearest_neighbors_pb2_grpc import NeighborsService
from lib.proto.py.nearest_neighbors_pb2 import (
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

    def NearestNeighborsEvent(
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

        model = nearestNeighborsFe(
            x,
            y if not y else None,
            **request.kwargs
        )

        response = NearestNeighborsReply(
            n_samples_fit_ = model.n_samples_fit_,
	        feature_names_in_ = model.feature_names_in_,
	        n_features_in_ = model.n_features_in_,
	        effective_metric_params_ = model.effective_metric_params_,
	        effective_metric_ = model.effective_metric_
        )

        return response
    
    def KDTreeEvent(
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

        model = kdTreeFe(
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

        response = KDTreeReply(
            matrix = model
        )

        return response
    
    def NearestCentroidEvent(
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

        model = nearestCentroidFe(
            x,
            y,
            request.metric,
            request.shrink_threshold
        )

        response = NearestCentroidReply(
	        feature_names_in_ = model.feature_names_in_,
	        n_features_in_ = model.n_features_in_,
	        classes_ = model.classes_,
	        centroids_ = model.centroids_
        )

        return response




