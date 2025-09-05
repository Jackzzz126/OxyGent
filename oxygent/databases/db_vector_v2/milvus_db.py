"""Milvus向量数据库实现"""
# oxygent/databases/db_vector_v2/milvus_db.py
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

from .base_vector_db import BaseVectorDB


class MilvusDB(BaseVectorDB):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Milvus连接
        :param config: 配置字典，包含host、port、user、password等
        """
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.user = config.get("user", "")
        self.password = config.get("password", "")
        self.alias = config.get("alias", "default")
        self._connect()

    def _connect(self): # tcp的长连接
        """建立Milvus连接"""
        if not connections.has_connection(self.alias):
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        index_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        if utility.has_collection(collection_name, self.alias):
            return True

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]

        # 添加元数据字段（动态字段）
        schema = CollectionSchema(
            fields=fields,
            description="Vector collection for knowledge base",
            enable_dynamic_field=True
        )

        # 创建集合
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )

        # 创建索引
        if index_params:
            collection.create_index(
                field_name="vector",
                index_params=index_params,
                metric_type=metric_type
            )
        return True

    async def insert(
        self,
        collection_name: str,
        vectors: Union[List[np.ndarray], np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        collection = Collection(collection_name, using=self.alias)
        if isinstance(vectors, list):
            vectors = np.array(vectors)

        # 生成默认ID
        if not ids:
            ids = [f"vec_{i}" for i in range(len(vectors))]

        # 构建插入数据
        data = [ids, vectors.tolist()]
        if metadata:
            for i in range(len(metadata)):
                data.append(metadata[i])

        # 执行插入
        result = collection.insert(data)
        collection.flush()
        return result.primary_keys

    async def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        collection = Collection(collection_name, using=self.alias)
        collection.load()

        # 构建过滤条件
        expr = None
        if filter:
            expr = " and ".join([f"{k} == '{v}'" for k, v in filter.items()])

        # 执行搜索
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=kwargs.get("search_params", {"metric_type": "L2", "params": {}}),
            limit=limit,
            expr=expr,
            output_fields=["*"]
        )

        # 格式化结果
        output = []
        for hit in results[0]:
            output.append({
                "id": hit.id,
                "score": hit.score,
                "vector": hit.entity.get("vector"),
                "metadata": {k: v for k, v in hit.entity.get_properties().items() if k != "vector"}
            })
        return output

    # 其他方法（delete/update/get等）实现略