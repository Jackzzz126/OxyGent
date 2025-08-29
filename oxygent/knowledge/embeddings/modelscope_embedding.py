from .abc_embedding import AbstractEmbedding
from modelscope.models import Model
from transformers import AutoTokenizer
import numpy as np
from typing import List
import torch


class ModelScopeEmbedding(AbstractEmbedding):
    """魔搭社区（ModelScope）Embedding实现"""

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model = Model.from_pretrained(model_id, device=device)
        # 加载Qwen专用Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Qwen模型需要设置padding_side为left，并且添加pad_token
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用eos_token作为pad_token
        self.device = device
        # 获取维度（通过测试嵌入计算）
        test_emb = self._encode_single("test")
        self._dimension = len(test_emb)

    def _encode_single(self, text: str) -> np.ndarray:
        """适配Qwen3-Embedding模型的嵌入提取逻辑"""
        # 预处理文本，添加Qwen模型要求的格式
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512  # Qwen模型通常有长度限制
        )
        # 将输入转移到模型设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 模型推理（关闭梯度计算提高效率）
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Qwen3-Embedding从logits中提取最后一个token的特征
            if "logits" not in outputs:
                available_keys = list(outputs.keys())
                raise KeyError(f"模型输出中找不到 logits，可用键为：{available_keys}")

            # 取最后一个token的logits作为嵌入（Qwen生成式模型的常见做法）
            # logits形状: [batch_size, seq_len, vocab_size]
            last_token_logits = outputs["logits"][:, -1, :].squeeze()  # 取最后一个token的logits
            if isinstance(last_token_logits, torch.Tensor):
                last_token_logits = last_token_logits.cpu().numpy()

        return last_token_logits

    async def embed(self, texts: List[str]) -> np.ndarray:
        # 同步模型异步包装
        import asyncio
        loop = asyncio.get_event_loop()

        def _encode():
            return np.array([self._encode_single(text) for text in texts])

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