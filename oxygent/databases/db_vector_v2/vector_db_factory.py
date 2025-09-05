# oxygent/databases/db_vector_v2/vector_db_factory.py
"""向量数据库工厂类"""
from typing import Dict, Type, Any, List
from .base_vector_db import BaseVectorDB
from .vearch_db import VearchDB
from .milvus_db import MilvusDB
from .chroma_db import ChromaDB
from .qdrant_db import QdrantDB
from .es_db import EsVectorDB


class VectorDBFactory:
    """向量数据库工厂，用于创建不同类型的向量数据库实例"""

    _db_classes: Dict[str, Type[BaseVectorDB]] = {
        "vearch": VearchDB,
        "milvus": MilvusDB,
        "chroma": ChromaDB,
        "qdrant": QdrantDB,
        "es": EsVectorDB
    }

    @classmethod
    def create(cls, db_type: str, config: Dict[str, Any]) -> BaseVectorDB:
        """
        创建向量数据库实例
        :param db_type: 数据库类型
        :param config: 数据库配置
        :return: 向量数据库实例
        """
        db_class = cls._db_classes.get(db_type.lower())
        if not db_class:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")
        return db_class(config)

    @classmethod
    def supported_dbs(cls) -> List[str]:
        """返回支持的数据库类型列表"""
        return list(cls._db_classes.keys())