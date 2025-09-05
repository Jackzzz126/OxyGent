#
# # oxygent/knowledge/embeddings/modelscope_embedding.py
# from .abc_embedding import AbstractEmbedding
# from modelscope.models import Model
# from transformers import AutoTokenizer
# import numpy as np
# from typing import List
# import torch
#
#
# # 抽象 app sdk api, @gechunfa1
# class ModelScopeEmbedding(AbstractEmbedding):
#     """魔搭社区（ModelScope）Embedding实现"""
#
#     def __init__(self, model_id: str, device: str = "cpu"):
#         self.model = Model.from_pretrained(model_id, device=device)
#         # 加载Qwen专用Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         # Qwen模型需要设置padding_side为left，并且添加pad_token
#         self.tokenizer.padding_side = "left"
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用eos_token作为pad_token
#         self.device = device
#         # 获取维度（通过测试嵌入计算）
#         test_emb = self._encode_single("test")
#         self._dimension = len(test_emb)
#
#     def _encode_single(self, text: str) -> np.ndarray:
#         """适配Qwen3-Embedding模型的嵌入提取逻辑"""
#         # 预处理文本，添加Qwen模型要求的格式
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512  # Qwen模型通常有长度限制
#         )
#         # 将输入转移到模型设备
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#
#         # 模型推理（关闭梯度计算提高效率）
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#
#             # Qwen3-Embedding从logits中提取最后一个token的特征
#             if "logits" not in outputs:
#                 available_keys = list(outputs.keys())
#                 raise KeyError(f"模型输出中找不到 logits，可用键为：{available_keys}")
#
#             # 取最后一个token的logits作为嵌入（Qwen生成式模型的常见做法）
#             # logits形状: [batch_size, seq_len, vocab_size]
#             last_token_logits = outputs["logits"][:, -1, :].squeeze()  # 取最后一个token的logits
#             if isinstance(last_token_logits, torch.Tensor):
#                 last_token_logits = last_token_logits.cpu().numpy()
#
#         return last_token_logits
#
#     async def embed(self, texts: List[str]) -> np.ndarray:
#         # 同步模型异步包装
#         import asyncio
#         loop = asyncio.get_event_loop()
#
#         def _encode():
#             return np.array([self._encode_single(text) for text in texts])
#
#         embeddings = await loop.run_in_executor(None, _encode)
#         # 确保L2归一化
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         return embeddings / norms
#
#     @property
#     def dimension(self) -> int:
#         return self._dimension
#
#     @property
#     def platform(self) -> str:
#         return "modelscope"


from typing import List, Union
import asyncio
import numpy as np
import torch
from modelscope.models import Model
from modelscope.pipelines import pipeline
from transformers import AutoTokenizer
from .abc_embedding import AbstractEmbedding


class ModelScopeEmbedding(AbstractEmbedding):
    """最终修复版：适配Qwen3-Embedding的CausalLMOutputWithPast输出格式"""

    def __init__(self, model_id: str, device: str = "cpu"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self._dimension = 0
        self._use_pipeline = "Qwen3-Embedding" not in model_id  # Qwen专用逻辑标记

        # 初始化模型和处理器
        self._init_model()
        # 设置维度（优先硬编码，确保正确性）
        self._set_dimension()

    def _init_model(self):
        """根据模型类型选择合适的加载方式"""
        if self._use_pipeline:
            # 通用模型使用ModelScope流水线
            self.embed_pipeline = pipeline(
                task="text-embedding",
                model=self.model_id,
                device=self.device,
                model_revision="master"
            )
        else:
            # Qwen3-Embedding专用加载逻辑
            self.model = Model.from_pretrained(self.model_id, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Qwen模型特殊配置
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Qwen3-Embedding模型加载完成（设备：{self.device}）")

    def _set_dimension(self):
        """维度设置策略：硬编码优先，自动检测兜底"""
        model_dim_map = {
            "Qwen/Qwen3-Embedding-0.6B": 1024,
            "Qwen/Qwen3-Embedding-1.8B": 1536,
            "text2vec-large-chinese": 1024,
            "BAAI/bge-large-zh": 1024,
            "BAAI/bge-base-zh": 768
        }

        if self.model_id in model_dim_map:
            self._dimension = model_dim_map[self.model_id]
            print(f"✅ 硬编码维度设置：{self._dimension}（模型：{self.model_id}）")
        else:
            try:
                test_emb = asyncio.run(self.embed(["维度检测"]))[0]
                self._dimension = len(test_emb)
                print(f"✅ 自动检测维度：{self._dimension}（模型：{self.model_id}）")
            except Exception as e:
                raise RuntimeError(f"维度检测失败：{str(e)}")

        if self._dimension <= 0:
            raise ValueError(f"无效的模型维度：{self._dimension}")

    def _encode_qwen(self, text: str) -> np.ndarray:
        """终极修复：适配CausalLMOutputWithPast输出格式"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # 关键修复：处理CausalLMOutputWithPast类型输出
            if hasattr(outputs, 'logits'):
                # 从logits中提取最后一个token的特征
                last_token_logits = outputs.logits[:, -1, :]  # 形状: [1, vocab_size]
                embedding = last_token_logits.squeeze().cpu().numpy()  # 转为一维数组
            elif isinstance(outputs, tuple):
                # 兼容元组类型输出
                embedding = outputs[0].squeeze().cpu().numpy()
            else:
                # 最后的兜底方案
                embedding = outputs.squeeze().cpu().numpy()

        # 确保输出维度正确（1024维）
        if len(embedding) != self._dimension:
            raise RuntimeError(
                f"Qwen模型输出维度异常：实际{len(embedding)}维，预期{self._dimension}维"
            )
        return embedding

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """统一嵌入接口：返回list[list[float]]"""
        if not texts:
            return []

        def sync_embed():
            if self._use_pipeline:
                results = self.embed_pipeline(texts)
                return [result["embedding"] for result in results]
            else:
                embeddings = [self._encode_qwen(text) for text in texts]
                # 执行L2归一化
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized = (embeddings / norms).tolist()  # 转为list[float]
                return normalized

        return await asyncio.to_thread(sync_embed)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def platform(self) -> str:
        return "modelscope"


