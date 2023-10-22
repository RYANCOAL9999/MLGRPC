from module import neighborsEvents
from lib.proto.py.nearest_neighbors_pb2_grpc import add_NeighborsServiceServicer_to_server

def control(server, server_manager)-> None:
    add_NeighborsServiceServicer_to_server(
        neighborsEvents(server_manager),
        server
    )