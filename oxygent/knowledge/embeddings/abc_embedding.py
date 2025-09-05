
# oxygent/knowledge/embeddings/abc_embedding.py
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class AbstractEmbedding(ABC):
    """Embedding模型核心接口，与具体平台解耦"""

    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        """生成文本嵌入向量
        Args:
            texts: 待嵌入的文本列表
        Returns:
            形状为 [len(texts), embedding_dim] 的numpy数组
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量维度"""
        return 0

    @property
    @abstractmethod
    def platform(self) -> str:
        """模型所属平台标识（如huggingface, modelscope等）"""
        return ""