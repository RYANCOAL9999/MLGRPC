from server_manager import ServerManager
from lib.proto.py.general_expression_pb2 import DataFrame
from lib.proto.py.general_expression_pb2_grpc import GeneralService

class GeneralEvents(GeneralService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def HeaderEvent(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showhead()

        response = DataFrame(
            data = model.data
        )

        return response
    
    def InfoEvent(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showInfo()

        response = DataFrame(
            data = model.data
        )
        
        return response
    
    def DescriblerEvent(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showDescrible()

        response = DataFrame(
            data = model.data
        )
        
        return response

