from module.linearEvents import LinearEvents
from lib.proto.py.linear_expression_pb2_grpc import add_LinearServiceServicer_to_server

def control(server, server_manager)-> None:
    add_LinearServiceServicer_to_server(
        LinearEvents(server_manager),
        server
    )
