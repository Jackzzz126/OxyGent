"""Qdrant向量数据库实现"""
# oxygent/databases/db_vector_v2/qdrant_db.py
from typing import List, Dict, Any, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from .base_vector_db import BaseVectorDB


class QdrantDB(BaseVectorDB):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Qdrant连接
        :param config: 配置字典，包含host、port、api_key等
        """
        self.client = QdrantClient(
            host=config.get("host", "localhost"),
            port=config.get("port", 6333),
            api_key=config.get("api_key")
        )

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",** kwargs
    ) -> bool:
        # 检查集合是否存在
        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return True

        # 创建集合
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=metric_type.upper()  # Qdrant要求大写（L2/IP/COSINE）
            ),
            **kwargs
        )
        return True

    async def insert(
        self,
        collection_name: str,
        vectors: Union[List[np.ndarray], np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        if isinstance(vectors, list):
            vectors = np.array(vectors)

        # 生成默认ID
        if not ids:
            ids = [str(i) for i in range(len(vectors))]

        # 构建点数据
        points = []
        for i in range(len(vectors)):
            point = PointStruct(
                id=ids[i],
                vector=vectors[i].tolist(),
                payload=metadata[i] if metadata else None
            )
            points.append(point)

        # 执行插入
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        return ids

    async def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,** kwargs
    ) -> List[Dict[str, Any]]:
        # 构建过滤条件
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(
                    key=k,
                    match=MatchValue(value=v)
                ) for k, v in filter.items()
            ]
            qdrant_filter = Filter(
                must=conditions
            )

        # 执行搜索
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            filter=qdrant_filter,
            **kwargs
        )

        # 格式化结果
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "vector": hit.vector,
                "metadata": hit.payload
            } for hit in results
        ]

    # 其他方法实现略