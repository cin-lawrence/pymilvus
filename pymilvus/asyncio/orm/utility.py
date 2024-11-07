"""
Source: pymilvus.orm.utility
"""

from .connections import connections


def _get_connection(alias: str):
    return connections._fetch_handler(alias)


def get_server_type(using: str = "default"):
    """Get the server type. Now, it will return "zilliz" if the connection related to
        an instance on the zilliz cloud, otherwise "milvus" will be returned.

    :param using: Alias to the connection. Default connection is used if this is not specified.
    :type  using: str

    :return: The server type.
    :rtype: str
    """
    return _get_connection(using).get_server_type()
