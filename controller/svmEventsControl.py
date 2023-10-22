from module.svmEvents import SVMEvents
from lib.proto.py.svm_expression_pb2_grpc import add_SVMServiceServicer_to_server

def control(server, server_manager)-> None:
    add_SVMServiceServicer_to_server(
        SVMEvents(server_manager),
        server
    )