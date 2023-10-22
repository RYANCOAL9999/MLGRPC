import os
import grpc
import asyncio
import logging

from server_manager import ServerManager

# Coroutines to be invoked when the event loop is shutting down.
_cleanup_coroutines = []
async def server_graceful_shutdown(server) -> None:
    logging.info("Starting graceful shutdown...")
    # Shuts down the server with 5 seconds of grace period. During the
    # grace period, the server won't accept new connections and allow
    # existing RPCs to continue within the grace period.
    await server.stop(5)

async def serve() -> None:
    """
    Serves the gRPC server.
    """

    server_manager = ServerManager()

    gRPCKey = str(os.environ['GRPCKEY'])

    port = int(os.environ['PORT'])

    address = "[::]:"

    server = grpc.server()

    await server_manager.start(server, gRPCKey)

    listen_addr = address + port

    # grpc.ssl_server_credentials(...) if os.environ['SSL'] else None
    server_credentials = None # grpc.ssl_server_credentials(...) 

    server.add_secure_port(listen_addr, server_credentials)  # Use secure port instead of insecure port

    logging.info("Starting server on %s", listen_addr)

    await server.start()

    _cleanup_coroutines.append(server_graceful_shutdown(server))

    await server.wait_for_termination()

if __name__ == "__main__":

    logging.basicConfig(
        filename = None,
        filemode = "",
        format = "",
        datefmt = None,
        style = "",
        level = logging.INFO,
        stream = None,
        handlers = None,
        force = None,
        encoding = None,
        errors = None
    )

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(serve())
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()