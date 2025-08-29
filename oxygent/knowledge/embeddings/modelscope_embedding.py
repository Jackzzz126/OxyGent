
from .abc_embedding import AbstractEmbedding
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors.nlp import SentenceEmbeddingTransformersPreprocessor
import numpy as np
from typing import List


class ModelScopeEmbedding(AbstractEmbedding):
    """魔搭社区（ModelScope）Embedding实现"""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model = Model.from_pretrained(model_id, device=device)
        self.preprocessor = SentenceEmbeddingTransformersPreprocessor(model_dir=self.model.model_dir)
        self.pipeline = pipeline(
            task="text-embedding",
            model=self.model,
            preprocessor=self.preprocessor
        )
        # 获取维度（通过测试嵌入计算）
        test_emb = self.pipeline("test")["text_embedding"]
        self._dimension = len(test_emb)

    async def embed(self, texts: List[str]) -> np.ndarray:
        # 同步模型异步包装
        import asyncio
        loop = asyncio.get_event_loop()

        def _encode():
            results = [self.pipeline(text)["text_embedding"] for text in texts]
            return np.array(results)

        embeddings = await loop.run_in_executor(None, _encode)
        # 确保L2归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def platform(self) -> str:
        return "modelscope"