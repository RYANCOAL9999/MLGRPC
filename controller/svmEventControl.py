from module import svmEvent
from lib.proto.py.svm_expression_pb2_grpc import add_SVMServiceServicer_to_server

def control(server, server_manager)-> None:
    add_SVMServiceServicer_to_server(
        svmEvent(server_manager),
        server
    )