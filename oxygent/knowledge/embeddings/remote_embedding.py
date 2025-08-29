
from .abc_embedding import AbstractEmbedding
import httpx
import numpy as np
import base64
from typing import List


class RemoteEmbedding(AbstractEmbedding):
    """远程API Embedding实现（兼容OpenAI/自定义服务）"""

    def __init__(self, api_url: str, api_key: str = None, dimension: int = None):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._dimension = dimension  # 可选：提前指定维度

    async def embed(self, texts: List[str]) -> np.ndarray:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                headers=self.headers,
                json={"input": texts}
            )
            response.raise_for_status()
            data = response.json()

            # 兼容OpenAI格式
            if "data" in data:
                embeddings = [item["embedding"] for item in data["data"]]
            else:
                # 自定义API格式
                embeddings = data.get("embeddings", [])

            # 自动获取维度（如果未指定）
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])

            return np.array(embeddings)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ValueError("Dimension not initialized, call embed first")
        return self._dimension

    @property
    def platform(self) -> str:
        return "remote"