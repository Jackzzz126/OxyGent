
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
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts, True  # 启用L2归一化
        )
        return np.array(embeddings)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def platform(self) -> str:
        return "huggingface"