
# examples/knowledgebase/rag_demo.py

from oxygent import MAS, oxy, OxyRequest
from oxygent.databases.db_vector_v2.vearch_db import VearchDB
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.knowledge.embeddings.modelscope_embedding import ModelScopeEmbedding
from oxygent.knowledge.knowledge_base import KnowledgeBase

# 初始化知识库的基础配置
vearch_config = {
    "router_url": "http://ocre-aigc-router.vectorbase.svc.lf09.n.jd.local",
    "master_url": "http://ocre-aigc-master.vectorbase.svc.lf09.n.jd.local",
    "db_name": "qqd_db_rag_base",
    "tool_df_space_name": "guojian31_bdagent_tool_desc_test1"
}

vearch_db = VearchDB(config=vearch_config)
doc_processor = DocumentProcessor()
embedding_model = ModelScopeEmbedding(
    model_id="Qwen/Qwen3-Embedding-0.6B",
    device="cpu"
)

# 在这里定义一个知识库的实例
knowledge_base = KnowledgeBase(
    vector_db=vearch_db,
    embedding_model=embedding_model,
    doc_processor=doc_processor,
    collection_name="jd_knowledge"
)

# 查询知识库的方法
async def add_knowledge(oxy_request: OxyRequest):

    await knowledge_base.initialize()
    # 1. 将知识库实例存入 shared_data（替代直接赋值属性）
    oxy_request.shared_data["knowledge_base"] = knowledge_base

    # 2. 获取用户查询（逻辑不变）
    query = oxy_request.get_query()

    # 3. 从 shared_data 中获取知识库实例并调用检索
    retrieve_result = await oxy_request.shared_data["knowledge_base"].retrieve(
        query=query,
        top_k=5
    )
    # 处理检索结果：提取content拼接为字符串（适配原prompt模板）
    if retrieve_result["state"] == "completed" and retrieve_result["output"]:
        # 从返回结果中提取每条知识的content字段
        knowledge_content = "\n\n".join(
            [item["content"] for item in retrieve_result["output"]]
        )
    else:
        # 检索失败或无结果时的默认提示
        knowledge_content = "未检索到相关知识，请根据常识回答。"

    # 将处理后的知识存入请求参数，供prompt模板使用
    oxy_request.arguments["knowledge"] = knowledge_content
    print("检索到的知识：\n", knowledge_content)  # 调试用
    return oxy_request


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
        prompt="你是一个乐于助人的助手！你可以参考以下的知识回答问题：\n${knowledge}",   # 占位符
        # 在这里进行知识库方法的调用
        func_process_input=add_knowledge,
    )

]

async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="京东的211时效是什么？")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
