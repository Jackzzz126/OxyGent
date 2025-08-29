import asyncio
from typing import List

import numpy as np
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from oxygent.knowledge.embeddings.abc_embedding import AbstractEmbedding
from oxygent.knowledge.embeddings.modelscope_embedding import ModelScopeEmbedding
from oxygent.knowledge.embeddings.huggingface_embedding import  HuggingFaceEmbedding

# 假设所有类都在embeddings模块中

async def embedding_func(embedding: AbstractEmbedding, texts: List[str]):
    """测试嵌入模型的通用函数"""
    print(f"\n测试 {embedding.platform} 嵌入模型...")
    print(f"模型维度: {embedding.dimension}")

    # 生成嵌入
    embeddings = await embedding.embed(texts)

    # 输出结果信息
    print(f"生成的嵌入形状: {embeddings.shape}")
    print(f"第一个向量前5个值: {embeddings[0][:5].round(4)}")
    print(f"第一个向量的L2范数: {np.linalg.norm(embeddings[0]):.6f}")  # 应接近1.0


async def main():
    # 测试文本
    test_texts = [
        "人工智能正在改变世界",
        "自然语言处理是人工智能的重要分支",
        "Embedding将文本转换为向量表示"
    ]

    # 1. 测试HuggingFace嵌入模型
    hf_embedding = HuggingFaceEmbedding(
        model_name_or_path="all-MiniLM-L6-v2",  # 轻量级通用模型
        device="cpu"  # 可选"cuda"启用GPU加速
    )
    await embedding_func(hf_embedding, test_texts)

    # 2. 测试ModelScope嵌入模型
    ms_embedding = ModelScopeEmbedding(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        # model_id="iic/nlp_gte_sentence-embedding_chinese-base",
        device="cpu"
    )
    await embedding_func(ms_embedding, test_texts)


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())