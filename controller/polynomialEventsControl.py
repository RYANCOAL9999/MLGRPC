from module.polynomialEvents import PolynomialEvents
from lib.proto.py.polynomial_features_pb2_grpc import add_PolynomialServiceServicer_to_server

def control(server, server_manager)-> None:
    add_PolynomialServiceServicer_to_server(
        PolynomialEvents(server_manager),
        server
    )