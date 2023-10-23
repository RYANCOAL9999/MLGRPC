from server_manager import ServerManager

from lib.proto.py.file_handler_pb2 import (
    FileService,
    FileEventReply,
    FileUploadRequest,
    FileDeleteRequest
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
        ) -> FileEventReply:

        if not isinstance(request, FileUploadRequest):
            raise ValueError("Invalid request. Expected LinearRegressionRequest.")
        
        request = FileUploadRequest(request)

        result = self.manager.accessDF(request.url)

        return FileEventReply(
            access = result
        )
    
    def fileDeleteTrigger(
            self,
            request, 
            context
        ) -> FileEventReply:

        # if not isinstance(request, FileDeleteRequest):
            # raise ValueError("Invalid request. Expected LinearRegressionRequest.")
        
        request = FileDeleteRequest(request)

        result = self.manager.deleteDF(request.fileName)

        return FileEventReply(
            access = result
        )
