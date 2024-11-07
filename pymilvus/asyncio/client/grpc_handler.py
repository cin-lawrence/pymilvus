import copy
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Optional, Union

import grpc
from grpc._cython import cygrpc

from pymilvus.client import entity_helper, ts_utils, utils
from pymilvus.client.abstract import CollectionSchema, MutationResult, SearchResult
from pymilvus.client.check import check_pass_param
from pymilvus.client.constants import ITERATOR_SESSION_TS_FIELD
from pymilvus.client.grpc_handler import GrpcHandler as BaseGrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.client.types import DataType, ExtraList, LoadState, Status, get_cost_extra
from pymilvus.client.utils import check_status, is_successful, len_of
from pymilvus.exceptions import (
    DescribeCollectionException,
    ErrorCode,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import common_pb2, milvus_pb2_grpc
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.settings import Config

from .generic_client_interceptor import GenericAioClientInterceptor, header_adder_aio_interceptor
from .interceptor import intercept_aio_channel


# TODO: Implement BaseGrpcHandler
class GrpcHandler(BaseGrpcHandler):
    _insecure_channel = staticmethod(grpc.aio.insecure_channel)
    _secure_channel = staticmethod(grpc.aio.secure_channel)
    _channel: grpc.aio.Channel
    _stub: milvus_pb2_grpc.MilvusServiceStub

    def __init__(
        self,
        uri: str = Config.GRPC_URI,
        host: str = "",
        port: str = "",
        channel: Optional[grpc.aio.Channel] = None,
        **kwargs,
    ) -> None:
        self._channel = channel

        addr = kwargs.get("address")
        self._address = addr if addr is not None else self.__get_address(uri, host, port)
        self._log_level = None
        self._request_id = None
        self._user = kwargs.get("user")
        self._set_authorization(**kwargs)
        self._setup_db_interceptor(kwargs.get("db_name"))
        self._setup_grpc_channel()
        self.callbacks: List[Callable] = []

    # TODO: @retry_on_rpc_failure()
    async def create_collection(
        self, collection_name: str, fields: List, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.create_collection_request(collection_name, fields, **kwargs)
        status = await self._stub.CreateCollection(request, timeout=timeout)
        check_status(status)

    # TODO: @retry_on_rpc_failure()
    async def rename_collections(
        self,
        old_name: str,
        new_name: str,
        new_db_name: str = "",
        timeout: Optional[float] = None,
    ):
        check_pass_param(collection_name=new_name, timeout=timeout)
        check_pass_param(collection_name=old_name)
        if new_db_name:
            check_pass_param(db_name=new_db_name)
        request = Prepare.rename_collections_request(old_name, new_name, new_db_name)
        response = await self._stub.RenameCollection(request, timeout=timeout)
        check_status(response)

    # TODO: @retry_on_rpc_failure()
    async def describe_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.describe_collection_request(collection_name)
        response = await self._stub.DescribeCollection(request, timeout=timeout)
        status = response.status

        if is_successful(status):
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.code, status.reason, status.error_code)

    # TODO: @retry_on_rpc_failure()
    async def has_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.describe_collection_request(collection_name)
        response = await self._stub.DescribeCollection(request, timeout=timeout)

        # For compatibility with Milvus less than 2.3.2, which does not support status.code.
        if (
            response.status.error_code == common_pb2.UnexpectedError
            and "can't find collection" in response.status.reason
        ):
            return False

        if response.status.error_code == common_pb2.CollectionNotExists:
            return False

        if is_successful(response.status):
            return True

        if response.status.code == ErrorCode.COLLECTION_NOT_FOUND:
            return False

        raise MilvusException(
            response.status.code, response.status.reason, response.status.error_code
        )

    # TODO: @retry_on_rpc_failure()
    async def list_collections(self, timeout: Optional[float] = None):
        request = Prepare.show_collections_request()
        response = await self._stub.ShowCollections(request, timeout=timeout)
        status = response.status
        check_status(status)
        return list(response.collection_names)

    # TODO: @retry_on_rpc_failure()
    async def drop_collection(self, collection_name: str, timeout: Optional[float] = None):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.drop_collection_request(collection_name)

        status = await self._stub.DropCollection(request, timeout=timeout)
        check_status(status)

    # TODO: @retry_on_rpc_failure()
    async def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        index_param = Prepare.get_collection_stats_request(collection_name)
        response = await self._stub.GetCollectionStatistics(index_param, timeout=timeout)
        status = response.status
        check_status(status)
        return response.stats

    # TODO: @retry_on_rpc_failure()
    async def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            collection_name=collection_name, replica_number=replica_number, timeout=timeout
        )
        refresh = kwargs.get("refresh", kwargs.get("_refresh", False))
        resource_groups = kwargs.get("resource_groups", kwargs.get("_resource_groups"))
        load_fields = kwargs.get("load_fields", kwargs.get("_load_fields"))
        skip_load_dynamic_field = kwargs.get(
            "skip_load_dynamic_field", kwargs.get("_skip_load_dynamic_field", False)
        )

        request = Prepare.load_collection(
            "",
            collection_name,
            replica_number,
            refresh,
            resource_groups,
            load_fields,
            skip_load_dynamic_field,
        )
        status = await self._stub.LoadCollection(request, timeout=timeout)
        check_status(status)

    # TODO: @retry_on_rpc_failure()
    async def release_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.release_collection("", collection_name)
        status = await self._stub.ReleaseCollection(request, timeout=timeout)
        check_status(status)

    # TODO: @retry_on_rpc_failure()
    async def get_load_state(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        request = Prepare.get_load_state(collection_name, partition_names)
        response = await self._stub.GetLoadState(request, timeout=timeout)
        check_status(response.status)
        return LoadState(response.state)

    # TODO: @retry_on_rpc_failure()
    async def get_loading_progress(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        is_refresh: bool = False,
    ):
        request = Prepare.get_loading_progress(collection_name, partition_names)
        response = await self._stub.GetLoadingProgress(request, timeout=timeout)
        check_status(response.status)
        if is_refresh:
            return response.refresh_progress
        return response.progress

    # TODO: @retry_on_rpc_failure()
    async def insert_rows(
        self,
        collection_name: str,
        entities: Union[Dict, List[Dict]],
        partition_name: Optional[str] = None,
        schema: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        request = await self._prepare_row_insert_request(
            collection_name, entities, partition_name, schema, timeout=timeout, **kwargs
        )
        response = await self._stub.Insert(request=request, timeout=timeout)
        check_status(response.status)
        ts_utils.update_collection_ts(collection_name, response.timestamp)
        return MutationResult(response)

    # TODO: @retry_on_rpc_failure()
    async def upsert_rows(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(entities, dict):
            entities = [entities]
        request = await self._prepare_row_upsert_request(
            collection_name, entities, partition_name, timeout=timeout, **kwargs
        )
        response = await self._stub.Upsert(request, timeout=timeout)
        check_status(response.status)
        m = MutationResult(response)
        ts_utils.update_collection_ts(collection_name, m.timestamp)
        return m

    # TODO: @retry_on_rpc_failure()
    # TODO: Is it good than default Milvus's async search
    async def search(
        self,
        collection_name: str,
        data: Union[List[List[float]], utils.SparseMatrixInputType],
        anns_field: str,
        param: Dict,
        limit: int,
        expression: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            limit=limit,
            round_decimal=round_decimal,
            anns_field=anns_field,
            search_data=data,
            partition_name_array=partition_names,
            output_fields=output_fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp"),
            timeout=timeout,
        )

        request = Prepare.search_requests_with_expr(
            collection_name,
            data,
            anns_field,
            param,
            limit,
            expression,
            partition_names,
            output_fields,
            round_decimal,
            **kwargs,
        )
        return await self._execute_search(request, timeout, round_decimal=round_decimal, **kwargs)

    # TODO: @retry_on_rpc_failure()
    async def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if output_fields is not None and not isinstance(output_fields, list):
            raise ParamError(message="Invalid query format. 'output_fields' must be a list")
        request = Prepare.query_request(
            collection_name, expr, output_fields, partition_names, **kwargs
        )

        response = await self._stub.Query(request, timeout=timeout)
        if Status.EMPTY_COLLECTION in {response.status.code, response.status.error_code}:
            return []
        check_status(response.status)

        num_fields = len(response.fields_data)
        # check has fields
        if num_fields == 0:
            raise MilvusException(message="No fields returned")

        # check if all lists are of the same length
        it = iter(response.fields_data)
        num_entities = len_of(next(it))
        if not all(len_of(field_data) == num_entities for field_data in it):
            raise MilvusException(message="The length of fields data is inconsistent")

        _, dynamic_fields = entity_helper.extract_dynamic_field_from_result(response)

        results = []
        for index in range(num_entities):
            entity_row_data = entity_helper.extract_row_data_from_fields_data(
                response.fields_data, index, dynamic_fields
            )
            results.append(entity_row_data)

        extra_dict = get_cost_extra(response.status)
        extra_dict[ITERATOR_SESSION_TS_FIELD] = response.session_ts
        return ExtraList(results, extra=extra_dict)

    # TODO: @retry_on_rpc_failure()
    async def delete(
        self,
        collection_name: str,
        expression: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        try:
            req = Prepare.delete_request(
                collection_name,
                partition_name,
                expression,
                consistency_level=kwargs.get("consistency_level", 0),
                param_name=kwargs.get("param_name"),
            )
            response = await self._stub.Delete(req, timeout=timeout)
            check_status(response.status)
            m = MutationResult(response)
            ts_utils.update_collection_ts(collection_name, m.timestamp)
        except Exception as err:
            raise err from err
        else:
            return m

    # TODO: @retry_on_rpc_failure()
    async def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        # for historical reason, index_name contained in kwargs.
        index_name = kwargs.pop("index_name", Config.IndexName)
        copy_kwargs = copy.deepcopy(kwargs)

        collection_desc = await self.describe_collection(
            collection_name, timeout=timeout, **copy_kwargs
        )

        valid_field = False
        for fields in collection_desc["fields"]:
            if field_name != fields["name"]:
                continue
            valid_field = True
            if fields["type"] not in {
                DataType.FLOAT_VECTOR,
                DataType.BINARY_VECTOR,
                DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR,
                DataType.SPARSE_FLOAT_VECTOR,
            }:
                break

        if not valid_field:
            raise MilvusException(message=f"cannot create index on non-existed field: {field_name}")

        index_param = Prepare.create_index_request(
            collection_name, field_name, params, index_name=index_name
        )

        status = await self._stub.CreateIndex(index_param, timeout=timeout)
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    # TODO: @retry_on_rpc_failure()
    async def load_partitions_progress(
        self, collection_name: str, partition_names: List[str], timeout: Optional[float] = None
    ):
        """Return loading progress of partitions"""
        progress = self.get_loading_progress(collection_name, partition_names, timeout)
        return {
            "loading_progress": f"{progress:.0f}%",
        }

    def _header_adder_interceptor(self, header, value) -> GenericAioClientInterceptor:
        return header_adder_aio_interceptor(header, value)

    def _setup_db_interceptor(self, db_name: Optional[str] = None):
        if db_name is None:
            self._db_interceptor = None
        else:
            check_pass_param(db_name=db_name)
            self._db_interceptor = self._header_adder_interceptor(["dbname"], [db_name])

    def _setup_identifier_interceptor(self, user: str, timeout: int = 10):
        host = socket.gethostname()
        self._identifier = self.__internal_register(user, host, timeout=timeout)
        self._identifier_interceptor = self._header_adder_interceptor(
            ["identifier"], [str(self._identifier)]
        )
        self._final_channel: grpc.aio.Channel = intercept_aio_channel(
            self._final_channel, self._identifier_interceptor
        )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    def _setup_grpc_channel(self):
        """Create a ddl grpc channel"""
        if self._channel is None:
            opts = [
                (cygrpc.ChannelArgKey.max_send_message_length, -1),
                (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_time_ms", 55000),
            ]
            if not self._secure:
                self._channel = self._insecure_channel(
                    self._address,
                    options=opts,
                )
            else:
                if self._server_name != "":
                    opts.append(("grpc.ssl_target_name_override", self._server_name))

                root_cert, private_k, cert_chain = None, None, None
                if self._server_pem_path != "":
                    with Path(self._server_pem_path).open("rb") as f:
                        root_cert = f.read()
                elif (
                    self._client_pem_path != ""
                    and self._client_key_path != ""
                    and self._ca_pem_path != ""
                ):
                    with Path(self._ca_pem_path).open("rb") as f:
                        root_cert = f.read()
                    with Path(self._client_key_path).open("rb") as f:
                        private_k = f.read()
                    with Path(self._client_pem_path).open("rb") as f:
                        cert_chain = f.read()

                creds = grpc.ssl_channel_credentials(
                    root_certificates=root_cert,
                    private_key=private_k,
                    certificate_chain=cert_chain,
                )
                self._channel = self._secure_channel(
                    self._address,
                    creds,
                    options=opts,
                )

        # avoid to add duplicate headers.
        self._final_channel = self._channel
        if self._authorization_interceptor:
            self._final_channel = intercept_aio_channel(
                self._final_channel, self._authorization_interceptor
            )
        if self._db_interceptor:
            self._final_channel = intercept_aio_channel(self._final_channel, self._db_interceptor)
        if self._log_level:
            log_level_interceptor = self._header_adder_interceptor(["log_level"], [self._log_level])
            self._final_channel = intercept_aio_channel(self._final_channel, log_level_interceptor)
            self._log_level = None
        if self._request_id:
            request_id_interceptor = self._header_adder_interceptor(
                ["client_request_id"], [self._request_id]
            )
            self._final_channel = intercept_aio_channel(self._final_channel, request_id_interceptor)
            self._request_id = None

        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    async def _channel_ready(self):
        if self._channel is None:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message="No channel in handler, please setup grpc channel first",
            )

        await self._channel.channel_ready()

    async def _prepare_row_insert_request(
        self,
        collection_name: str,
        entity_rows: Union[List[Dict], Dict],
        partition_name: Optional[str] = None,
        schema: Optional[Dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(entity_rows, dict):
            entity_rows = [entity_rows]

        if not isinstance(schema, dict):
            schema = await self.describe_collection(collection_name, timeout=timeout)

        fields_info = schema.get("fields")
        enable_dynamic = schema.get("enable_dynamic_field", False)

        return Prepare.row_insert_param(
            collection_name,
            entity_rows,
            partition_name,
            fields_info,
            enable_dynamic=enable_dynamic,
        )

    async def _prepare_row_upsert_request(
        self,
        collection_name: str,
        rows: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if not isinstance(rows, list):
            raise ParamError(message="None rows, please provide valid row data.")

        fields_info, enable_dynamic = await self._get_info(collection_name, timeout, **kwargs)
        return Prepare.row_upsert_param(
            collection_name,
            rows,
            partition_name,
            fields_info,
            enable_dynamic=enable_dynamic,
        )

    async def _execute_search(
        self, request: milvus_types.SearchRequest, timeout: Optional[float] = None, **kwargs
    ):
        try:
            response = await self._stub.Search(request, timeout=timeout)
            check_status(response.status)
            round_decimal = kwargs.get("round_decimal", -1)
            return SearchResult(
                response.results,
                round_decimal,
                status=response.status,
                # session_ts=response.session_ts, # TODO: Check in next Milvus version
            )
        except Exception as e:
            raise e from e

    async def _get_info(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        schema = kwargs.get("schema")
        if not schema:
            schema = await self.describe_collection(collection_name, timeout=timeout)

        fields_info = schema.get("fields")
        enable_dynamic = schema.get("enable_dynamic_field", False)

        return fields_info, enable_dynamic
