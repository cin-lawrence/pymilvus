import copy
from typing import Awaitable, Dict

from ...orm.connections import BaseConnections
from ..client.grpc_handler import GrpcHandler


class Connections(BaseConnections[GrpcHandler, Awaitable[None]]):
    """Async Connections."""

    async def _disconnect(self, alias: str, remove_connection: bool):
        if alias in self._connected_alias:
            await self._connected_alias.pop(alias).close()
        if remove_connection:
            self._alias.pop(alias, None)

    async def _connect(self, alias: str, kwargs_copy: Dict, **kwargs):
        gh = GrpcHandler(**kwargs)

        await gh._channel_ready()
        # TODO: ReconnectHandler
        # if kwargs.get("keep_alive", False):
        #     gh.register_state_change_callback(
        #         ReconnectHandler(self, alias, kwargs_copy).reconnect_on_idle
        #     )
        kwargs.pop("password")
        kwargs.pop("token", None)
        kwargs.pop("db_name", "")

        self._connected_alias[alias] = gh
        self._alias[alias] = copy.deepcopy(kwargs)


# Singleton Mode in Python
connections = Connections()
