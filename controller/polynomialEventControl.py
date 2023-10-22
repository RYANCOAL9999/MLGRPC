from module import polynomialEvent
from lib.proto.py.polynomial_features_pb2_grpc import add_PolynomialServiceServicer_to_server

def control(server, server_manager)-> None:
    add_PolynomialServiceServicer_to_server(
        polynomialEvent(server_manager),
        server
    )