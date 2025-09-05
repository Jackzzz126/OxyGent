
# examples/knowledgebase/rag_demo.py

import os
from oxygent import MAS, oxy, OxyRequest
from oxygent.oxy import ChatAgent

def add_knowledge(oxy_request: OxyRequest):
    def retrieval(query):
        """以下用字符匹配模拟，而生产环境中，通常根据query从知识库中召回topK条相关语料，可以增加多路召回逻辑"""
        knowledage_dict = {
            "211时效": "**京东211时效**：当日上午 11:00 前提交的现货订单(部分城市为上午 10:00 点前），当日送达；当日 23:00 前提交的现货订单，次日 15:00 前送达。(注：先货订单以提交时间点开始计算，先款订单以支付完成时间点计算)",
            "使命": "京东的使命是 “技术为本，让生活更美好”。",
            "愿景": "京东的愿景是成为全球最值得信赖的企业。",
            "价值观": "京东核心价值观是：客户为先、创新、拼搏、担当、感恩、诚信。",
        }
        return "\n\n".join([v for k, v in knowledage_dict.items() if k in query])

    oxy_request.arguments["knowledge"] = retrieval(oxy_request.get_query())
    print(oxy_request.arguments["knowledge"])

    # 手动拼接最终Prompt并打印
    prompt_template = "你是一个乐于助人的助手！你可以参考以下的知识回答问题：\n${knowledge}"
    final_prompt = prompt_template.replace("${knowledge}", oxy_request.arguments["knowledge"])
    print("\n=== 最终 Prompt ===")
    print(final_prompt)
    print("===================\n")

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
        func_process_input=add_knowledge,   # prompt里注入knowledge信息
        # kbs={"k1", "k2"}
    ),

    oxy.ChatAgent(
        name="QA_agent_2",
        llm_model="default_llm",
        prompt="你是经济学家:\n${knowledge}",
        func_process_input=add_knowledge,   # prompt里注入knowledge信息
    )

    # oxy.BaseKnowledge(
    #     name="k1",
    #     url="http:127.0.0.1:8091"
    # ),
    # oxy.BaseKnowledge(
    #     name="k2",
    #     url="http:127.0.0.1:8092"
    # ),

]

async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="京东的211时效是什么？")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
