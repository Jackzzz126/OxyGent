
import asyncio
from oxygent import MAS, oxy, Config
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.knowledge.knowledge_base import KnowledgeBase
from oxygent.knowledge.kb_tools import KnowledgeBaseTool



# 知识库配置
oxy_space = [
    # # 向量数据库（使用现有实现）


    # 嵌入模型（使用现有实现）
    # oxy.EmbeddingModel(
    #     name="default_embedding",
    #     model_name="bge-large-en"
    # ),

    # 文档处理器
    DocumentProcessor(name="doc_processor"),

    # 知识库实例
    KnowledgeBase(
        name="product_kb",
        vector_db="kb_vector_db",
        doc_processor="doc_processor",
        embedding_model="default_embedding"
    ),

    # 知识库工具
    KnowledgeBaseTool(name="kb_tool", kb_name="product_kb"),

    # 应用智能体
    oxy.ReActAgent(
        name="kb_agent",
        is_master=True,
        llm_model="default_llm",
        tools=["kb_tool"],
        prompt="You are a knowledge base assistant. Use the kb_tool to answer user questions."
    ),

    # LLM模型
    oxy.HttpLLM(
        name="default_llm",
        api_key="YOUR_API_KEY",
        base_url="YOUR_BASE_URL",
        model_name=""
    )
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        # 添加示例文档
        await mas.call_oxy(
            callee="product_kb",
            arguments={
                "operation": "add_document",
                "file_path": "examples/data/product_manual.pdf",
                "owner": "admin"
            }
        )

        # 启动服务
        await mas.start_web_service(
            first_query="What are the main features of the product?"
        )


if __name__ == "__main__":
    asyncio.run(main())