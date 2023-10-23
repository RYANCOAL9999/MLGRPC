

from lib.proto.py.general_expression_pb2 import (
    GeneralService,
    DataFrame
)
from server_manager import ServerManager

class GeneralEvents(GeneralService):

    def __init__(
            self, 
            serverManager : ServerManager
        )->None:
        self.manager = serverManager

    def HeaderEventTriggered(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showhead()

        response = DataFrame(
            data = model.data
        )

        return response
    
    def InfoEventTriggered(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showInfo()

        response = DataFrame(
            data = model.data
        )
        
        return response
    
    def DescribeEventTriggered(
            self, 
            context
        )->DataFrame:

        if not self.manager.reversed(): return response

        model = self.manager.showDescrible()

        response = DataFrame(
            data = model.data
        )
        
        return response
