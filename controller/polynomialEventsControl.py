from module.polynomialEvents import PolynomialEvent
from lib.proto.py.polynomial_features_pb2_grpc import add_PolynomialServiceServicer_to_server

def control(server, server_manager)-> None:
    add_PolynomialServiceServicer_to_server(
        PolynomialEvent(server_manager),
        server
    )