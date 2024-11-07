"""MilvusClient for dealing with simple workflows."""

import logging
from uuid import uuid4
from typing import Optional

from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import (
    ExceptionsMessage,
    ExtraList,
    LoadState,
    OmitZeroDict,
    construct_cost_extra,
)
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    ParamError,
    PrimaryKeyException,
)
from pymilvus.milvus_client.index import IndexParams
from pymilvus.orm.collection import CollectionSchema
from pymilvus.orm.types import DataType
from pymilvus.settings import Config

from .orm.connections import connections
from .orm.utility import get_server_type

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MilvusClient:
    """The Async Milvus Client"""

    # TODO: Can we keep the same interface as sync MilvusClient?
    def __init__(self, conn_alias: str = Config.MILVUS_CONN_ALIAS) -> None:
        self._using = conn_alias
        self.is_self_hosted = bool(get_server_type(using=self._using) == "milvus")

    @classmethod
    async def connect(
        cls,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        **kwargs,
    ) -> "MilvusClient":
        # Async connection creation task
        using = await cls._create_connection(uri, user, password, db_name, token, **kwargs)
        return cls(using)

    async def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        primary_field_name: str = "id",  # default is "id"
        id_type: str = "int",  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "COSINE",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        schema: Optional[CollectionSchema] = None,
        index_params: Optional[IndexParams] = None,
        **kwargs,
    ):
        if schema is None:
            if dimension is None:
                raise ValueError("Need to provide either schema or dimension")

            return await self._fast_create_collection(
                collection_name,
                dimension,
                primary_field_name=primary_field_name,
                id_type=id_type,
                vector_field_name=vector_field_name,
                metric_type=metric_type,
                auto_id=auto_id,
                timeout=timeout,
                **kwargs,
            )

        return await self._create_collection_with_schema(
            collection_name, schema, index_params, timeout=timeout, **kwargs
        )

    @classmethod
    def create_schema(cls, **kwargs):
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def prepare_index_params(cls, field_name: str = "", **kwargs):
        return IndexParams(field_name, **kwargs)

    async def rename_collection(
        self,
        old_name: str,
        new_name: str,
        target_db: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.rename_collections(old_name, new_name, target_db, timeout=timeout, **kwargs)

    async def describe_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        return await conn.describe_collection(collection_name, timeout=timeout, **kwargs)

    async def has_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return await conn.has_collection(collection_name, timeout=timeout, **kwargs)

    async def list_collections(self, **kwargs):
        conn = self._get_connection()
        return await conn.list_collections(**kwargs)

    async def drop_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Delete the collection stored in this object"""
        conn = self._get_connection()
        await conn.drop_collection(collection_name, timeout=timeout, **kwargs)

    async def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> Dict:
        conn = self._get_connection()
        stats = await conn.get_collection_stats(collection_name, timeout=timeout)
        result = {stat.key: stat.value for stat in stats}
        if "row_count" in result:
            result["row_count"] = int(result["row_count"])
        return result

    async def load_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Loads the collection."""
        conn = self._get_connection()
        try:
            await conn.load_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error("Failed to load collection: %s", collection_name)
            raise ex from ex

    async def release_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            await conn.release_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error("Failed to load collection: %s", collection_name)
            raise ex from ex

    async def get_load_state(
        self,
        collection_name: str,
        partition_name: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        conn = self._get_connection()
        partition_names = None
        if partition_name:
            partition_names = [partition_name]
        try:
            state = await conn.get_load_state(
                collection_name, partition_names, timeout=timeout, **kwargs
            )
        except Exception as ex:
            raise ex from ex

        ret = {"state": state}
        if state == LoadState.Loading:
            progress = await conn.get_loading_progress(
                collection_name, partition_names, timeout=timeout
            )
            ret["progress"] = progress

        return ret

    async def refresh_load(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        kwargs.pop("_refresh", None)
        conn = self._get_connection()
        await conn.load_collection(collection_name, timeout=timeout, _refresh=True, **kwargs)

    async def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Insert data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.

        Raises:
            DataNotMatchException: If the data has missing fields an exception will be thrown.
            MilvusException: General Milvus error on insert.

        Returns:
            Dict: Number of rows that were inserted and the inserted primary key list.
        """
        # If no data provided, we cannot input anything
        if isinstance(data, dict):
            data = [data]

        msg = "wrong type of argument 'data', "
        msg += f"expected 'Dict' or list of 'Dict', got '{type(data).__name__}'"

        if not isinstance(data, list):
            raise TypeError(msg)

        if len(data) == 0:
            return {"insert_count": 0, "ids": []}

        conn = self._get_connection()
        # Insert into the collection.
        try:
            res = await conn.insert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout
            )
        except Exception as ex:
            raise ex from ex
        return OmitZeroDict(
            {
                "insert_count": res.insert_count,
                "ids": res.primary_keys,
                "cost": res.cost,
            }
        )

    async def upsert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Upsert data into the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.

        Raises:
            DataNotMatchException: If the data has missing fields an exception will be thrown.
            MilvusException: General Milvus error on upsert.

        Returns:
            Dict: Number of rows that were upserted.
        """
        # If no data provided, we cannot input anything
        if isinstance(data, dict):
            data = [data]

        msg = "wrong type of argument 'data', "
        msg += f"expected 'Dict' or list of 'Dict', got '{type(data).__name__}'"

        if not isinstance(data, list):
            raise TypeError(msg)

        if len(data) == 0:
            return {"upsert_count": 0}

        conn = self._get_connection()
        # Upsert into the collection.
        try:
            res = await conn.upsert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout, **kwargs
            )
        except Exception as ex:
            raise ex from ex

        return OmitZeroDict(
            {
                "upsert_count": res.upsert_count,
                "cost": res.cost,
            }
        )

    async def search(
        self,
        collection_name: str,
        data: Union[List[List], List],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        **kwargs,
    ) -> List[List[Dict]]:
        """Search for a query vector/vectors.

        In order for the search to process, a collection needs to have been either provided
        at init or data needs to have been inserted.

        Args:
            data (Union[List[list], list]): The vector/vectors to search.
            limit (int, optional): How many results to return per search. Defaults to 10.
            filter(str, optional): A filter to use for the search. Defaults to None.
            output_fields (List[str], optional): List of which field values to return. If None
                specified, only primary fields including distances will be returned.
            search_params (dict, optional): The search params to use for the search.
            timeout (float, optional): Timeout to use, overrides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: The collection being searched doesn't exist. Need to insert data first.

        Returns:
            List[List[dict]]: A nested list of dicts containing the result data. Embeddings are
                not included in the result data.
        """
        conn = self._get_connection()
        try:
            res = await conn.search(
                collection_name,
                data,
                anns_field or "",
                search_params or {},
                expression=filter,
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to search collection: %s", collection_name)
            raise ex from ex

        ret = []
        for hits in res:
            query_result = []
            for hit in hits:
                query_result.append(hit.to_dict())
            ret.append(query_result)

        return ExtraList(ret, extra=construct_cost_extra(res.cost))

    async def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        ids: Optional[Union[List, str, int]] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict]:
        """Query for entries in the Collection.

        Args:
            filter (str): The filter to use for the query.
            output_fields (List[str], optional): List of which field values to return. If None
                specified, all fields excluding vector field will be returned.
            partitions (List[str], optional): Which partitions to perform query. Defaults to None.
            timeout (float, optional): Timeout to use, overrides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts, vectors are not included.
        """
        if filter and not isinstance(filter, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

        if filter and ids is not None:
            raise ParamError(message=ExceptionsMessage.AmbiguousQueryFilterParam)

        if isinstance(ids, (int, str)):
            ids = [ids]

        conn = self._get_connection()
        try:
            schema_dict = await conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        if ids:
            filter = self._pack_pks_expr(schema_dict, ids)

        if not output_fields:
            output_fields = ["*"]
            vec_field_name = self._get_vector_field_name(schema_dict)
            if vec_field_name:
                output_fields.append(vec_field_name)

        try:
            res = await conn.query(
                collection_name,
                expr=filter,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to query collection: %s", collection_name)
            raise ex from ex

        return res

    async def get(
        self,
        collection_name: str,
        ids: Union[List, str, int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict]:
        """Grab the inserted vectors using the primary key from the Collection.

        Due to current implementations, grabbing a large amount of vectors is slow.

        Args:
            ids (str): The pk's to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
            timeout (float, optional): Timeout to use, overrides the client level assigned at
                init. Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts with keys {pk_field, vector_field}
        """
        if not isinstance(ids, list):
            ids = [ids]

        if len(ids) == 0:
            return []

        conn = self._get_connection()
        try:
            schema_dict = await conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        if not output_fields:
            output_fields = ["*"]
            vec_field_name = await self._get_vector_field_name(schema_dict)
            if vec_field_name:
                output_fields.append(vec_field_name)

        expr = self._pack_pks_expr(schema_dict, ids)
        try:
            res = await conn.query(
                collection_name,
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to get collection: %s", collection_name)
            raise ex from ex

        return res

    async def delete(
        self,
        collection_name: str,
        ids: Optional[Union[List, str, int]] = None,
        timeout: Optional[float] = None,
        filter: Optional[str] = "",
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Union[Dict, List]:
        """Delete entries in the collection by their pk or by filter.

        Starting from version 2.3.2, Milvus no longer includes the primary keys in the result
        when processing the delete operation on expressions.
        This change is due to the large amount of data involved.
        The delete interface no longer returns any results.
        If no exceptions are thrown, it indicates a successful deletion.
        However, for backward compatibility, If the primary_keys returned from old
        Milvus(previous 2.3.2) is not empty, the list of primary keys is still returned.

        Args:
            ids (list, str, int): The pk's to delete. Depending on pk_field type it can be int
                or str or alist of either. Default to None.
            filter(str, optional): A filter to use for the deletion. Defaults to empty.
            timeout (int, optional): Timeout to use, overrides the client level assigned at init.
                Defaults to None.

        Returns:
            dict | list: Number of rows that were deleted OR list of primary keys that were deleted.
        """
        pks = kwargs.get("pks", [])
        if isinstance(pks, (int, str)):
            pks = [pks]

        for pk in pks:
            if not isinstance(pk, (int, str)):
                msg = "wrong type of argument pks, "
                msg += f"expect list, int or str, got '{type(pk).__name__}'"
                raise TypeError(msg)

        if ids is not None:
            if isinstance(ids, (int, str)):
                pks.append(ids)
            elif isinstance(ids, list):
                for id in ids:
                    if not isinstance(id, (int, str)):
                        msg = "wrong type of argument ids, "
                        msg += f"expect list, int or str, got '{type(id).__name__}'"
                        raise TypeError(msg)
                pks.extend(ids)
            else:
                msg = "wrong type of argument ids, "
                msg += f"expect list, int or str, got '{type(ids).__name__}'"
                raise TypeError(msg)

        expr = ""
        conn = self._get_connection()
        if pks:
            try:
                schema_dict = await conn.describe_collection(
                    collection_name, timeout=timeout, **kwargs
                )
            except Exception as ex:
                logger.error("Failed to describe collection: %s", collection_name)
                raise ex from ex

            expr = self._pack_pks_expr(schema_dict, pks)

        if filter:
            if expr:
                raise ParamError(message=ExceptionsMessage.AmbiguousDeleteFilterParam)

            if not isinstance(filter, str):
                raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

            expr = filter

        ret_pks = []
        try:
            res = await conn.delete(
                collection_name,
                expr,
                partition_name,
                timeout=timeout,
                param_name="filter or ids",
                **kwargs,
            )
            if res.primary_keys:
                ret_pks.extend(res.primary_keys)
        except Exception as ex:
            logger.error("Failed to delete primary keys in collection: %s", collection_name)
            raise ex from ex

        if ret_pks:
            return ret_pks

        return OmitZeroDict({"delete_count": res.delete_count, "cost": res.cost})

    async def create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        for index_param in index_params:
            await self._create_index(collection_name, index_param, timeout=timeout, **kwargs)

    async def close(self):
        await connections.disconnect(self._using)

    @staticmethod
    async def _create_connection(
        uri: str,
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        **kwargs,
    ) -> str:
        """Create the connection to the Milvus server."""
        # TODO: Implement reuse with new uri style
        using = uuid4().hex

        try:
            await connections.connect(using, user, password, db_name, token, uri=uri, **kwargs)
        except Exception as ex:
            logger.error("Failed to create new connection using: %s", using)
            raise ex from ex
        else:
            logger.debug("Created new connection using: %s", using)
            return using

    async def _fast_create_collection(
        self,
        collection_name: str,
        dimension: int,
        primary_field_name: str = "id",  # default is "id"
        id_type: Union[DataType, str] = DataType.INT64,  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "COSINE",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if dimension is None:
            msg = "missing required argument: 'dimension'"
            raise TypeError(msg)
        if "enable_dynamic_field" not in kwargs:
            kwargs["enable_dynamic_field"] = True

        schema = self.create_schema(auto_id=auto_id, **kwargs)

        if id_type in ("int", DataType.INT64):
            pk_data_type = DataType.INT64
        elif id_type in ("string", "str", DataType.VARCHAR):
            pk_data_type = DataType.VARCHAR
        else:
            raise PrimaryKeyException(message=ExceptionsMessage.PrimaryFieldType)

        pk_args = {}
        if "max_length" in kwargs and pk_data_type == DataType.VARCHAR:
            pk_args["max_length"] = kwargs["max_length"]

        schema.add_field(primary_field_name, pk_data_type, is_primary=True, **pk_args)
        vector_type = DataType.FLOAT_VECTOR
        schema.add_field(vector_field_name, vector_type, dim=dimension)
        schema.verify()

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        try:
            await conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        index_params = IndexParams()
        index_params.add_index(vector_field_name, "", "", metric_type=metric_type)
        await self.create_index(collection_name, index_params, timeout=timeout)
        await self.load_collection(collection_name, timeout=timeout)

    async def _create_collection_with_schema(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        schema.verify()

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        try:
            await conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        if index_params:
            await self.create_index(collection_name, index_params, timeout=timeout)
            await self.load_collection(collection_name, timeout=timeout)

    async def _create_index(
        self, collection_name: str, index_param: Dict, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            params = index_param.pop("params", {})
            field_name = index_param.pop("field_name", "")
            index_name = index_param.pop("index_name", "")
            params.update(index_param)
            await conn.create_index(
                collection_name,
                field_name,
                params,
                timeout=timeout,
                index_name=index_name,
                **kwargs,
            )
            logger.debug("Successfully created an index on collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create an index on collection: %s", collection_name)
            raise ex from ex

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    def _extract_primary_field(self, schema_dict: Dict) -> Dict:
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("is_primary", None) is not None:
                return field_dict

        return {}

    def _get_vector_field_name(self, schema_dict: Dict):
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("type", None) == DataType.FLOAT_VECTOR:
                return field_dict.get("name", "")
        return ""

    def _pack_pks_expr(self, schema_dict: Dict, pks: List) -> str:
        primary_field = self._extract_primary_field(schema_dict)
        pk_field_name = primary_field["name"]
        data_type = primary_field["type"]

        # Varchar pks need double quotes around the values
        if data_type == DataType.VARCHAR:
            ids = ["'" + str(entry) + "'" for entry in pks]
            expr = f"""{pk_field_name} in [{','.join(ids)}]"""
        else:
            ids = [str(entry) for entry in pks]
            expr = f"{pk_field_name} in [{','.join(ids)}]"
        return expr
