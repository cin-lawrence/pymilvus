"""
Intercept an asynchronous gRPC channel with aio interceptors.
Reference:
- https://github.com/grpc/grpc/blob/master/src/python/grpcio/grpc/_interceptor.py
- https://github.com/grpc/grpc/blob/master/src/python/grpcio/grpc/aio/_interceptor.py
"""

from typing import Optional, Union

import grpc


def intercept_aio_channel(
    channel: grpc.aio._channel.Channel,
    *interceptors: Optional[
        Union[
            grpc.aio.UnaryUnaryClientInterceptor,
            grpc.aio.UnaryStreamClientInterceptor,
            grpc.aio.StreamStreamClientInterceptor,
            grpc.aio.StreamUnaryClientInterceptor,
        ]
    ],
) -> grpc.aio._channel.Channel:
    """Intercept an asynchronous gRPC channel with aio interceptors."""
    if interceptors is not None:
        for interceptor in interceptors:
            if isinstance(interceptor, grpc.aio.UnaryUnaryClientInterceptor):
                channel._unary_unary_interceptors.append(interceptor)
            elif isinstance(interceptor, grpc.aio.UnaryStreamClientInterceptor):
                channel._unary_stream_interceptors.append(interceptor)
            elif isinstance(interceptor, grpc.aio.StreamUnaryClientInterceptor):
                channel._stream_unary_interceptors.append(interceptor)
            elif isinstance(interceptor, grpc.aio.StreamStreamClientInterceptor):
                channel._stream_stream_interceptors.append(interceptor)
            else:
                raise ValueError(
                    f"Interceptor {interceptor} must be "
                    + f"{grpc.aio.UnaryUnaryClientInterceptor.__name__} or "
                    + f"{grpc.aio.UnaryStreamClientInterceptor.__name__} or "
                    + f"{grpc.aio.StreamUnaryClientInterceptor.__name__} or "
                    + f"{grpc.aio.StreamStreamClientInterceptor.__name__}. "
                )
    return channel
