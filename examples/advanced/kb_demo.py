
import asyncio
from oxygent import MAS, oxy, Config
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.knowledge.knowledge_base import KnowledgeBase
from oxygent.knowledge.kb_tools import KnowledgeBaseTool


#知识库-agent N:1

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
        prompt="You are a knowledge base assistant. Use the kb_tool to answer user questions.",
        #知识库
        kb={"milvus_table_01", "milvus_table_02", "milvus_table_03"}
    ),

    # LLM模型
    oxy.HttpLLM(
        name="default_llm",
        api_key="YOUR_API_KEY",
        base_url="YOUR_BASE_URL",
        model_name="",
    ),


    # db http server
    # jsf服务 -> mcp服务
    # support remote service and local service
    # 参考openai_llm方法调用流程 -> 实现OpenAILLM
    # 原始文档/分割之后的知识/向量/切分的规则(重点)
    # 知识库(存储方式: 结构化(mysql/es) + 非结构化(向量库)(nlp->向量)) M:N
    # 知识:chunk之后的数据(可以供召回的数据)
    # 原始的文档id(原始的知识id), 处理之后可以被检索的知识id(文档id)
    # 先考虑下一条知识里面包含各种模态数据(词向量化) -> 知识块(N条知识的组合)
    # 单条知识->embedding -> insert vector
    # query condition -> embedding -> 向量化之后的知识块 -> 得到原始数据
    # 混合检索
    oxy.knowledgeDb(
        name="milvus_table_01", # server_name
        url="localhost:1001", #知识库中的表名
        func_cls="" #后置方法
    )

    # rag-engine(kb)
]


# class knowledgeDb(Oxy):
#     call_remote:
#
#     call_local:



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