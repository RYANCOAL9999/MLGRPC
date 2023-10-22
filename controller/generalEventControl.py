from module.generalEvents import GeneralEvents
from lib.proto.py.general_expression_pb2_grpc import add_generalServiceServicer_to_server

def control(server, server_manager)->None:
    add_generalServiceServicer_to_server(
        GeneralEvents(server_manager),
        server
    )
