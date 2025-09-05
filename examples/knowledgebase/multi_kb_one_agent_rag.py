from oxygent import MAS, oxy, OxyRequest
from oxygent.databases.db_vector_v2.vearch_db import VearchDB
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.knowledge.embeddings.modelscope_embedding import ModelScopeEmbedding
from oxygent.knowledge.knowledge_base import KnowledgeBase

# 初始化多个知识库配置
kb_configs = [
    {
        "name": "jd_knowledge",
        "collection": "jd_knowledge",
        "vearch": {
            "router_url": "http://router1",
            "master_url": "http://master1",
            "db_name": "db1"
        }
    },
    {
        "name": "taobao_knowledge",
        "collection": "taobao_knowledge",
        "vearch": {
            "router_url": "http://router2",
            "master_url": "http://master2",
            "db_name": "db2"
        }
    }
]

# 初始化嵌入模型（Qwen3-Embedding-0.6B）
embedding_model = ModelScopeEmbedding(
    model_id="Qwen/Qwen3-Embedding-0.6B",
    device="cpu"
)

doc_processor = DocumentProcessor()

# 创建多个知识库实例
knowledge_bases = {}
for cfg in kb_configs:
    vearch_db = VearchDB(config=cfg["vearch"])
    knowledge_bases[cfg["name"]] = KnowledgeBase(
        vector_db=vearch_db,
        embedding_model=embedding_model,
        doc_processor=doc_processor,
        collection_name=cfg["collection"]
    )


# 多知识库检索处理函数
async def multi_kb_retrieval(oxy_request: OxyRequest):
    # 1. 存储所有知识库到共享数据
    oxy_request.shared_data["knowledge_bases"] = knowledge_bases

    # 2. 获取需要查询的知识库列表（从agent配置中读取）
    target_kbs = oxy_request.agent_config.get("kbs", [])
    if not target_kbs:
        target_kbs = list(knowledge_bases.keys())  # 默认查询所有

    # 3. 并行查询多个知识库
    all_knowledge = []
    for kb_name in target_kbs:
        kb = oxy_request.shared_data["knowledge_bases"][kb_name]
        await kb.initialize()
        result = await kb.retrieve(
            query=oxy_request.get_query(),
            top_k=3
        )
        if result["state"] == "completed" and result["output"]:
            all_knowledge.extend([item["content"] for item in result["output"]])

    # 4. 合并结果
    oxy_request.arguments["knowledge"] = "\n\n".join(all_knowledge) or "未检索到相关知识"
    return oxy_request


# 定义使用多知识库的Agent
oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key="EMPTY",
        base_url="http://aigc-custom-qwen25-vl-7b.jd.local/v1/chat/completions",
        model_name="qwen25vl-7b",
        llm_params={"temperature": 0.01}
    ),
    oxy.ChatAgent(
        name="QA_agent",
        llm_model="default_llm",
        prompt="你是一个乐于助人的助手！你可以参考以下的知识回答问题：\n${knowledge}",
        func_process_input=multi_kb_retrieval,
        agent_config={"kbs": ["jd_knowledge", "taobao_knowledge"]}  # 指定使用的知识库
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="京东的211时效是什么？")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())