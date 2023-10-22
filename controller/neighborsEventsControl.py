from module.neighborsEvents import NeighborsEvents
from lib.proto.py.nearest_neighbors_pb2_grpc import add_NeighborsServiceServicer_to_server

def control(server, server_manager)-> None:
    add_NeighborsServiceServicer_to_server(
        NeighborsEvents(server_manager),
        server
    )