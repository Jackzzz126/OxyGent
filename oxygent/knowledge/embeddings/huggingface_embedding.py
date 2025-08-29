
from .abc_embedding import AbstractEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class HuggingFaceEmbedding(AbstractEmbedding):
    """Hugging Face平台Embedding实现"""

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: List[str]) -> np.ndarray:
        # 同步模型异步包装
        import asyncio
        loop = asyncio.get_event_loop()
        # 正确：将参数作为关键字参数传递（使用 *args 和 **kwargs 分离）
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                prompt_name="query",  # 直接传递字符串参数
                convert_to_numpy=True
            )
        )
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def platform(self) -> str:
        return "huggingface"