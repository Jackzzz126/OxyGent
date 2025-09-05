"""Vearch向量数据库实现（修复版）"""
import asyncio
import json
import random
from typing import List, Dict, Any, Optional, Union

import httpx
import numpy as np

from oxygent.databases.db_vector_v2.base_vector_db import BaseVectorDB


class VearchDB(BaseVectorDB):
    """Vearch向量数据库适配器（遵循官方API规范）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Vearch连接
        :param config: 配置信息，包含master_url, router_url, db_name等
        """
        # 修复1：移除URL末尾斜杠，避免路径拼接错误
        self.master_url = config.get("master_url", "http://localhost:8817").rstrip("/")
        self.router_url = config.get("router_url", "http://localhost:8816").rstrip("/")
        self.db_name = config.get("db_name", "default_db")
        # 修复2：增加超时设置，避免网络延迟导致的JSON截断
        self.client = httpx.AsyncClient(timeout=30.0)
        # 从配置获取空间名称（兼容官方Demo的space概念）
        self.space_name = config.get("tool_df_space_name", "")

    async def _create_db_if_not_exists(self) -> bool:
        """创建数据库（如果不存在）- 遵循官方Demo的数据库创建规范"""
        try:
            # 检查数据库是否存在
            url = f"{self.master_url}/databases/{self.db_name}"
            response = await self.client.get(url)
            if response.status_code == 200:
                return True

            # 创建数据库（使用官方Demo的API路径和格式）
            create_url = f"{self.master_url}/databases"
            payload = {"name": self.db_name}
            response = await self.client.post(create_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"创建数据库失败：{str(e)}")
            return False

    async def create_collection(
            self,
            collection_name: str,
            dimension: int,
            metric_type: str = "L2",
            **kwargs
    ) -> bool:
        """创建集合（Space）- 严格遵循官方Demo的Schema规范"""
        # 确保数据库存在
        if not await self._create_db_if_not_exists():
            print("数据库创建失败，无法继续创建集合")
            return False

        # 修复3：统一度量类型格式（与官方Demo保持一致）
        metric_map = {
            "L2": "L2",
            "IP": "InnerProduct",
            "COSINE": "Cosine"
        }
        vearch_metric = metric_map.get(metric_type.upper(), "L2")

        # 修复4：使用官方Demo兼容的Space配置结构
        space_config = {
            "name": collection_name,
            "partitionNum": kwargs.get("partition_num", 1),  # 官方字段名使用驼峰式
            "replicaNum": kwargs.get("replica_num", 1),
            "fields": [
                {
                    "name": "vector",
                    "type": "vector",
                    "params": {
                        "dimension": dimension,
                        "metricType": vearch_metric,  # 官方字段名
                        "indexType": kwargs.get("index_type", "IVF_FLAT"),
                        "indexParam": {
                            "nlist": kwargs.get("nlist", 128)  # 仅保留官方支持的参数
                        }
                    }
                },
                {
                    "name": "metadata",
                    "type": "object",
                    "params": {"docValues": True}  # 驼峰式字段名
                }
            ]
        }

        # 修复5：使用官方Demo的API路径（/spaces而非/space）
        url = f"{self.master_url}/databases/{self.db_name}/spaces"
        try:
            # 打印配置用于调试（官方Demo的调试方式）
            print(f"创建集合配置: {json.dumps(space_config, indent=2)}")
            response = await self.client.post(url, json=space_config)

            # 修复6：同时检查状态码和响应内容（官方返回格式）
            print(f"创建集合响应: {response.status_code} - {response.text}")
            if response.status_code == 200:
                resp_data = response.json()
                return resp_data.get("code", 1) == 0
            return False
        except json.JSONDecodeError:
            print("Vearch返回无效JSON，可能是API路径错误")
            return False
        except Exception as e:
            print(f"创建集合异常: {str(e)}")
            return False

    async def insert(
            self,
            collection_name: str,
            vectors: Union[List[np.ndarray], np.ndarray],
            ids: Optional[List[str]] = None,
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """插入数据 - 适配官方批量插入格式"""
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        if not ids:
            ids = [f"vec_{random.getrandbits(64)}" for _ in range(len(vectors))]

        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]

        # 修复7：使用官方兼容的批量插入格式
        bulk_data = []
        for vec_id, vec, meta in zip(ids, vectors, metadata):
            bulk_data.append({
                "id": vec_id,
                "vector": vec,
                "metadata": meta
            })

        # 修复8：使用官方API路径（/bulk而非/_bulk）
        url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/bulk"
        try:
            response = await self.client.post(url, json={"docs": bulk_data})
            print(f"插入响应: {response.status_code} - {response.text}")

            if response.status_code == 200:
                resp_data = response.json()
                if resp_data.get("code") == 0:
                    return ids
            return []
        except Exception as e:
            print(f"插入异常: {str(e)}")
            return []

    async def search(
            self,
            collection_name: str,
            query_vector: np.ndarray,
            limit: int = 10,
            filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """搜索向量 - 遵循官方查询语法"""
        # 修复9：构建官方兼容的查询结构
        query = {
            "vector": {
                "field": "vector",
                "query": query_vector.tolist(),
                "topk": limit,
                "params": {"nprobe": kwargs.get("nprobe", 10)}
            },
            "fields": ["vector", "metadata"],
            "withDistance": True  # 官方参数，返回距离值
        }

        # 添加过滤条件（适配官方过滤语法）
        if filter:
            query["filter"] = {
                "conditions": [
                    {"field": f"metadata.{k}", "operator": "eq", "value": v}
                    for k, v in filter.items()
                ]
            }

        url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/search"
        try:
            response = await self.client.post(url, json=query)
            result = response.json()

            if result.get("code") != 0:
                print(f"搜索失败: {result.get('msg')}")
                return []

            # 格式化结果为统一格式
            hits = []
            for doc in result.get("data", {}).get("docs", []):
                hits.append({
                    "id": doc.get("id"),
                    "vector": doc.get("vector"),
                    "score": doc.get("distance"),  # 官方返回的是distance
                    "metadata": doc.get("metadata", {})
                })
            return hits
        except Exception as e:
            print(f"搜索异常: {str(e)}")
            return []

    async def delete(
            self,
            collection_name: str,
            ids: List[str],
            filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """删除数据 - 适配官方删除API"""
        if ids:
            # 修复10：使用官方批量删除API
            url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/delete"
            response = await self.client.post(url, json={"ids": ids})
            if response.status_code == 200:
                resp_data = response.json()
                return resp_data.get("data", {}).get("deleteCount", 0)
            return 0
        elif filter:
            # 通过过滤条件删除（先查询再删除）
            filter_query = {
                "filter": {
                    "conditions": [
                        {"field": f"metadata.{k}", "operator": "eq", "value": v}
                        for k, v in filter.items()
                    ]
                },
                "fields": [],
                "size": 10000
            }
            url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/query"
            response = await self.client.post(url, json=filter_query)
            result = response.json()
            if result.get("code") == 0:
                ids_to_delete = [doc.get("id") for doc in result.get("data", {}).get("docs", [])]
                return await self.delete(collection_name, ids_to_delete, None)
        return 0

    async def update(
            self,
            collection_name: str,
            ids: List[str],
            vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """更新数据 - 适配官方更新API"""
        if vectors is not None and isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        update_docs = []
        for i, vec_id in enumerate(ids):
            doc = {"id": vec_id}
            if vectors is not None and i < len(vectors):
                doc["vector"] = vectors[i]
            if metadata is not None and i < len(metadata):
                doc["metadata"] = metadata[i]
            update_docs.append(doc)

        url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/update"
        response = await self.client.post(url, json={"docs": update_docs})
        if response.status_code == 200:
            resp_data = response.json()
            return resp_data.get("data", {}).get("updateCount", 0)
        return 0

    async def get(
            self,
            collection_name: str,
            ids: List[str],
            filter: Optional[Dict[str, Any]] = None,
            include_vectors: bool = True
    ) -> List[Dict[str, Any]]:
        """查询数据 - 适配官方查询API"""
        result = []
        fields = ["metadata"]
        if include_vectors:
            fields.append("vector")

        if ids:
            url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/query"
            response = await self.client.post(url, json={
                "ids": ids,
                "fields": fields
            })
            if response.status_code == 200:
                resp_data = response.json()
                for doc in resp_data.get("data", {}).get("docs", []):
                    result.append({
                        "id": doc.get("id"),
                        "vector": doc.get("vector") if include_vectors else None,
                        "metadata": doc.get("metadata", {})
                    })
        elif filter:
            # 通过过滤条件查询
            url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/query"
            response = await self.client.post(url, json={
                "filter": {
                    "conditions": [
                        {"field": f"metadata.{k}", "operator": "eq", "value": v}
                        for k, v in filter.items()
                    ]
                },
                "fields": fields,
                "size": 10000
            })
            if response.status_code == 200:
                resp_data = response.json()
                for doc in resp_data.get("data", {}).get("docs", []):
                    result.append({
                        "id": doc.get("id"),
                        "vector": doc.get("vector") if include_vectors else None,
                        "metadata": doc.get("metadata", {})
                    })
        return result

    async def count(
            self,
            collection_name: str,
            filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """统计数量 - 适配官方统计API"""
        url = f"{self.router_url}/databases/{self.db_name}/spaces/{collection_name}/count"
        if filter:
            payload = {
                "filter": {
                    "conditions": [
                        {"field": f"metadata.{k}", "operator": "eq", "value": v}
                        for k, v in filter.items()
                    ]
                }
            }
            response = await self.client.post(url, json=payload)
        else:
            response = await self.client.get(url)

        if response.status_code == 200:
            resp_data = response.json()
            return resp_data.get("data", {}).get("count", 0)
        return 0

    async def drop_collection(self, collection_name: str) -> bool:
        """删除集合 - 适配官方删除API"""
        url = f"{self.master_url}/databases/{self.db_name}/spaces/{collection_name}"
        response = await self.client.delete(url)
        print(f"删除集合响应: {response.status_code} - {response.text}")
        if response.status_code == 200:
            return response.json().get("code", 1) == 0
        return False

    async def list_collections(self) -> List[str]:
        """列出所有集合 - 适配官方列表API"""
        url = f"{self.master_url}/databases/{self.db_name}/spaces"
        response = await self.client.get(url)
        if response.status_code == 200:
            resp_data = response.json()
            return [space["name"] for space in resp_data.get("data", {}).get("spaces", [])]
        return []
