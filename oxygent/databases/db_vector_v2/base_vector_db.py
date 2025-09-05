# oxygent/databases/db_vector_v2/base_vector_db.py
"""向量数据库抽象基类模块"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import numpy as np

from oxygent.databases.base_db import BaseDB

logger = logging.getLogger(__name__)


class BaseVectorDB(BaseDB, ABC):
    """向量数据库抽象基类，定义核心接口规范"""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        **kwargs
    ) -> bool:
        """
        创建向量集合（表）
        :param collection_name: 集合名称
        :param dimension: 向量维度
        :param metric_type: 距离度量方式（L2/IP/COSINE等）
        :param kwargs: 数据库特定参数（如索引配置）
        :return: 创建成功返回True
        """
        pass

    @abstractmethod
    async def insert(
        self,
        collection_name: str,
        vectors: Union[List[np.ndarray], np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        插入向量数据
        :param collection_name: 集合名称
        :param vectors: 向量列表或数组
        :param ids: 可选，向量唯一标识
        :param metadata: 可选，向量关联的元数据
        :return: 插入成功的ID列表
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        :param collection_name: 集合名称
        :param query_vector: 查询向量
        :param limit: 返回结果数量
        :param filter: 元数据过滤条件
        :param kwargs: 数据库特定参数（如搜索参数）
        :return: 搜索结果列表，包含id、向量、分数、元数据
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        ids: List[str],
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        删除向量数据
        :param collection_name: 集合名称
        :param ids: 要删除的ID列表（与filter二选一）
        :param filter: 元数据过滤条件（与ids二选一）
        :return: 删除的记录数
        """
        pass

    @abstractmethod
    async def update(
        self,
        collection_name: str,
        ids: List[str],
        vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        更新向量或元数据
        :param collection_name: 集合名称
        :param ids: 要更新的ID列表
        :param vectors: 可选，新向量值
        :param metadata: 可选，新元数据
        :return: 更新的记录数
        """
        pass

    @abstractmethod
    async def get(
        self,
        collection_name: str,
        ids: List[str],
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        获取向量数据
        :param collection_name: 集合名称
        :param ids: 要获取的ID列表（与filter二选一）
        :param filter: 元数据过滤条件（与ids二选一）
        :param include_vectors: 是否返回向量数据
        :return: 向量记录列表
        """
        pass

    @abstractmethod
    async def count(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        统计记录数量
        :param collection_name: 集合名称
        :param filter: 过滤条件
        :return: 记录总数
        """
        pass

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> bool:
        """
        删除集合
        :param collection_name: 集合名称
        :return: 删除成功返回True
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        列出所有集合
        :return: 集合名称列表
        """
        pass