from module.fileEvents import FileEvents
from lib.proto.py.file_handler_pb2_grpc import add_FileServiceServicer_to_server

def control(server, server_manager)->None:
    add_FileServiceServicer_to_server(
        FileEvents(server_manager),
        server
    )
