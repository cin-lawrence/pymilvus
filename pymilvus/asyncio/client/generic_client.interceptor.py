""" "
Asynchronous generic client interceptor for gRPC.
Reference: https://github.com/grpc/grpc/blob/master/examples/python/interceptors/headers/generic_client_interceptor.py
"""

from typing import Any, List

import grpc
import grpc.aio


class GenericAioClientInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,
    grpc.aio.UnaryStreamClientInterceptor,
    grpc.aio.StreamUnaryClientInterceptor,
    grpc.aio.StreamStreamClientInterceptor,
):
    def __init__(self, interceptor_function):
        """Initialize the interceptor with a custom function."""
        self._fn = interceptor_function

    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept unary-unary async RPC calls."""
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,))
        )
        response = await continuation(new_details, next(new_request_iterator))
        return await postprocess(response) if postprocess else response

    async def intercept_unary_stream(self, continuation, client_call_details, request):
        """Intercept unary-stream async RPC calls."""
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,))
        )
        response_it = await continuation(new_details, next(new_request_iterator))
        return await postprocess(response_it) if postprocess else response_it

    async def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """Intercept stream-unary async RPC calls."""
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator
        )
        response = await continuation(new_details, new_request_iterator)
        return await postprocess(response) if postprocess else response

    async def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        """Intercept stream-stream async RPC calls."""
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator
        )
        response_it = await continuation(new_details, new_request_iterator)
        return await postprocess(response_it) if postprocess else response_it


def header_adder_aio_interceptor(headers: List, values: List) -> GenericAioClientInterceptor:
    def intercept_call(
        client_call_details: Any,
        request_iterator: Any,
    ):
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        for item in zip(headers, values):
            metadata.append(item)
        client_call_details = grpc.aio.ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            grpc.aio.Metadata.from_tuple(tuple(metadata)),
            client_call_details.credentials,
            client_call_details.wait_for_ready,
        )
        return client_call_details, request_iterator, None

    return GenericAioClientInterceptor(intercept_call)
