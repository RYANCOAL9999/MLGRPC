from server_manager import ServerManager

from lib.proto.py.file_handler_pb2 import (
    FileService,
    FileUploadRequest,
    FileUploadReply
)

class FileEvents(FileService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def fileUploadTrigger(
            self, 
            request, 
            context
        ) -> FileUploadReply:

        if not isinstance(request, FileUploadRequest):
            raise ValueError("Invalid request. Expected LinearRegressionRequest.")
        
        request = FileUploadRequest(request)

        access = self.manager.accessDF(request.url)

        return FileUploadReply(access)
