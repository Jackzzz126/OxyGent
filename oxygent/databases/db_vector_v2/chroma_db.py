# oxygent/databases/db_vector_v2/chroma_db.py
"""Chroma向量数据库实现"""
from typing import List, Dict, Any, Optional, Union

import numpy as np
import chromadb
from chromadb.api.async_client import AsyncClient
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.config import Settings

from oxygent.databases.db_vector.base_vector_db import BaseVectorDB


class ChromaDB(BaseVectorDB):
    """Chroma向量数据库适配器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Chroma连接
        :param config: 配置信息，包含persist_directory, host, port等
        """
        self.client: Optional[AsyncClient] = None
        self.persist_directory = config.get("persist_directory")
        self.host = config.get("host")
        self.port = config.get("port")
        self.collections: Dict[str, ChromaCollection] = {}

    async def _connect(self):
        """建立连接"""
        if not self.client:
            if self.host and self.port:
                # 连接远程服务
                self.client = await chromadb.AsyncClient(
                    host=self.host,
                    port=self.port
                )
            else:
                # 本地模式
                settings = Settings(
                    persist_directory=self.persist_directory,
                    is_persistent=self.persist_directory is not None
                )
                self.client = await chromadb.AsyncClient(settings=settings)

    async def _get_collection(self, collection_name: str, create_if_not_exists: bool = False,
                              dimension: int = None, metric_type: str = None) -> ChromaCollection:
        """获取集合实例"""
        if collection_name in self.collections:
            return self.collections[collection_name]

        await self._connect()
        assert self.client is not None, "Chroma client not initialized"

        # 检查集合是否存在
        collections = await self.client.list_collections()
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            if not create_if_not_exists:
                raise ValueError(f"Collection {collection_name} does not exist")

            # 创建新集合
            metadata = {}
            if metric_type:
                metadata["hnsw:space"] = metric_type.lower()  # Chroma使用小写
            self.collections[collection_name] = await self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
        else:
            self.collections[collection_name] = await self.client.get_collection(
                name=collection_name
            )

        return self.collections[collection_name]

    async def create_collection(
            self,
            collection_name: str,
            dimension: int,
            metric_type: str = "L2",
            **kwargs
    ) -> bool:
        # Chroma会自动处理维度，不需要显式指定
        metric_map = {
            "L2": "l2",
            "IP": "ip",
            "COSINE": "cosine"
        }
        chroma_metric = metric_map.get(metric_type, "l2")

        try:
            await self._get_collection(
                collection_name,
                create_if_not_exists=True,
                metric_type=chroma_metric
            )
            return True
        except Exception:
            return False

    async def insert(
            self,
            collection_name: str,
            vectors: Union[List[np.ndarray], np.ndarray],
            ids: Optional[List[str]] = None,
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        collection = await self._get_collection(collection_name)

        # 处理向量格式
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if not isinstance(vectors, list):
            vectors = [vectors]

        # 生成ID
        if not ids:
            ids = [f"vec_{i}" for i in range(len(vectors))]

        # 处理元数据
        metadatas = metadata if metadata is not None else [{} for _ in range(len(vectors))]

        # 执行插入
        await collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas
        )
        return ids

    async def search(
            self,
            collection_name: str,
            query_vector: np.ndarray,
            limit: int = 10,
            filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        collection = await self._get_collection(collection_name)

        # 执行搜索
        results = await collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=limit,
            where=filter,
            include=["embeddings", "metadatas", "distances"]
        )

        # 处理结果
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "vector": results["embeddings"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i] or {}
            })
        return hits

    async def delete(
            self,
            collection_name: str,
            ids: List[str],
            filter: Optional[Dict[str, Any]] = None
    ) -> int:
        collection = await self._get_collection(collection_name)

        if ids:
            await collection.delete(ids=ids)
            return len(ids)
        elif filter:
            # 先查询符合条件的ID
            results = await collection.query(
                where=filter,
                include=[],
                n_results=10000
            )
            ids_to_delete = results["ids"][0] if results["ids"] else []
            if ids_to_delete:
                await collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        return 0

    async def update(
            self,
            collection_name: str,
            ids: List[str],
            vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        collection = await self._get_collection(collection_name)

        # 处理向量格式
        embeddings = None
        if vectors is not None:
            if isinstance(vectors, np.ndarray):
                embeddings = vectors.tolist()
            if not isinstance(embeddings, list):
                embeddings = [embeddings]

        # 执行更新
        await collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata
        )
        return len(ids)

    async def get(
            self,
            collection_name: str,
            ids: List[str],
            filter: Optional[Dict[str, Any]] = None,
            include_vectors: bool = True
    ) -> List[Dict[str, Any]]:
        collection = await self._get_collection(collection_name)

        include = ["metadatas"]
        if include_vectors:
            include.append("embeddings")

        # 执行查询
        if ids:
            results = await collection.get(
                ids=ids,
                include=include
            )
        elif filter:
            results = await collection.query(
                where=filter,
                include=include,
                n_results=10000
            )
        else:
            return []

        # 处理结果
        result_list = []
        if ids:
            # get方法返回格式处理
            for i in range(len(results["ids"])):
                result_list.append({
                    "id": results["ids"][i],
                    "vector": results.get("embeddings", [None])[i] if include_vectors else None,
                    "metadata": results.get("metadatas", [{}])[i] or {}
                })
        else:
            # query方法返回格式处理
            for i in range(len(results["ids"][0])):
                result_list.append({
                    "id": results["ids"][0][i],
                    "vector": results.get("embeddings", [[None]])[0][i] if include_vectors else None,
                    "metadata": results.get("metadatas", [{}])[0][i] or {}
                })
        return result_list

    async def count(
            self,
            collection_name: str,
            filter: Optional[Dict[str, Any]] = None
    ) -> int:
        collection = await self._get_collection(collection_name)

        if filter:
            results = await collection.query(
                where=filter,
                include=[],
                n_results=1
            )
            return len(results["ids"][0]) if results["ids"] else 0
        else:
            return await collection.count()

    async def drop_collection(self, collection_name: str) -> bool:
        await self._connect()
        assert self.client is not None, "Chroma client not initialized"

        try:
            await self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            return True
        except Exception:
            return False

    async def list_collections(self) -> List[str]:
        await self._connect()
        assert self.client is not None, "Chroma client not initialized"

        collections = await self.client.list_collections()
        return [col.name for col in collections]