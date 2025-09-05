import asyncio
from oxygent import MAS
from oxygent.databases.db_vector_v2.vearch_db import VearchDB
from oxygent.knowledge.embeddings.modelscope_embedding import ModelScopeEmbedding
from oxygent.oxy.agents.knowledge_agent import KnowledgeBaseAgent, KnowledgeAugmentedReActAgent
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.oxy.llms.http_llm import HttpLLM
from oxygent.schemas import OxyResponse

# Vearch配置
vearch_config = {
    "router_url": "http://ocre-aigc-router.vectorbase.svc.lf09.n.jd.local",
    "master_url": "http://ocre-aigc-master.vectorbase.svc.lf09.n.jd.local",
    "db_name": "qqd_db_rag_base",
    "tool_df_space_name": "guojian31_bdagent_tool_desc_test1"
}


async def main():
    # 1. 初始化组件（保持不变）
    vearch_db = VearchDB(config=vearch_config)
    doc_processor = DocumentProcessor()
    embedding_model = ModelScopeEmbedding(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        device="cpu"
    )
    llm = HttpLLM(
        name="default_llm",
        api_key="EMPTY",
        base_url="http://aigc-custom-qwen25-vl-7b.jd.local/v1/chat/completions",
        model_name="qwen25vl-7b",
        llm_params={"temperature": 0.01}
    )

    # 2. 配置组件空间（保持不变）
    oxy_space = [
        llm,
        KnowledgeBaseAgent(
            name="test_kb_agent",
            vector_db=vearch_db,
            embedding_model=embedding_model,
            doc_processor=doc_processor,
            collection_name="test_kb_collection"  # 指定明确的集合名称
        ),
        KnowledgeAugmentedReActAgent(
            name="knowledge_agent",
            is_master=True,
            llm_model="default_llm",
            knowledge_agent_name="test_kb_agent",
            max_react_rounds=3,
            tools=[]
        )
    ]

    # 启动MAS
    async with MAS(oxy_space=oxy_space) as mas:
        # 测试1：添加文档
        print("=== 测试1：添加文档 ===")
        add_resp = await mas.call(
            callee="test_kb_agent",
            caller="user",
            arguments={
                "operation": "add_document",
                "file_path": "/Users/gechunfa1/Documents/jd-opensource-code/raw_ai/OxyGent/examples/knowledgebase/data/hongmeng_car.txt",
                "document_id": "doc_001"
            }
        )
        if isinstance(add_resp, OxyResponse):
            print(f"状态：{'成功' if add_resp.state == 1 else '失败'}")
            print(f"结果：{add_resp.output}")
        else:
            print(f"状态：成功")
            print(f"结果：{add_resp}")

        # 新增：调试-手动检查集合中的记录数
        print("\n=== 调试：检查集合记录数 ===")
        count_resp = await mas.call(
            callee="test_kb_agent",
            caller="user",
            arguments={"operation": "count_documents"}  # 需要新增count_documents操作
        )
        print(f"集合记录数：{count_resp}")

        # 测试2：回答问题（保持不变）
        print("\n=== 测试2：回答问题 ===")
        query_resp = await mas.call(
            callee="knowledge_agent",
            caller="user",
            arguments={"query": "问界汽车是什么？"}
        )
        print(f"回答：{query_resp}")

        # 测试3：列出文档（保持不变）
        print("\n=== 测试3：列出文档 ===")
        list_resp = await mas.call(
            callee="test_kb_agent",
            caller="user",
            arguments={"operation": "list_documents"}
        )
        if isinstance(list_resp, OxyResponse):
            print(f"状态：{'成功' if list_resp.state == 1 else '失败'}")
            doc_list = list_resp.output if list_resp.output else "无文档"
            print(f"文档ID列表：{doc_list}")
        else:
            print(f"状态：成功")
            doc_list = list_resp if list_resp else "无文档"
            print(f"文档ID列表：{doc_list}")


if __name__ == "__main__":
    # 禁用MAS启动时的logo和版本信息（可选，用于清理输出）
    import logging

    logging.getLogger("oxygent").setLevel(logging.WARNING)

    asyncio.run(main())