import asyncio
import os

from pydantic import Field

from oxygent import MAS, Config, OxyRequest, OxyResponse, oxy
from oxygent.prompts import INTENTION_PROMPT
from oxygent.oxy.mcp_tools.streamable_mcp_client import StreamableMCPClient
Config.set_agent_llm_model("default_llm")
# Config.set_vearch_config({
#     "router_url": "${VEARCH_ROUTER_URL}",
#     "master_url": "${VEARCH_MASTER_URL}",
#     "db_name": "${VEARCH_NAME}",
#     "tool_df_space_name": "${TOOL_OF_SPACE_NAME}",
#     "embedding_model_url": "${EMBEDDING_MODEL_URL}"
# })

Config.set_vearch_config({
    "router_url": "http://ocre-aigc-router.vectorbase.svc.lf09.n.jd.local",
    "master_url": "http://ocre-aigc-master.vectorbase.svc.lf09.n.jd.local",
    "db_name": "qqd_db_rag_base",
    "tool_df_space_name": "guojian31_bdagent_tool_desc_test1",
    "embedding_model_url": "http://big-data-emb.jd.local/v2/models/embedding/infer"
})

async def workflow(oxy_request: OxyRequest):
    short_memory = oxy_request.get_short_memory()
    print("--- History record --- :", short_memory)
    master_short_memory = oxy_request.get_short_memory(master_level=True)
    print("--- History record-User layer --- :", master_short_memory)
    print("user query:", oxy_request.get_query(master_level=True))
    await oxy_request.send_message("msg")
    oxy_response = await oxy_request.call(
        callee="time_agent",
        arguments={"query": "What time is it now in Asia/Shanghai?"},
    )
    print("--- Current time --- :", oxy_response.output)
    oxy_response = await oxy_request.call(
        callee="default_llm",
        arguments={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "llm_params": {"temperature": 0.6},
        },
    )
    print(oxy_response.output)
    import re

    numbers = re.findall(r"\d+", oxy_request.get_query())
    if numbers:
        n = numbers[-1]
        oxy_response = await oxy_request.call(callee="calc_pi", arguments={"prec": n})
        return f"Save {n} positions: {oxy_response.output}"
    else:
        return "Save 2 positions: 3.14, or you could ask me to save how many positions you want."

def update_query(oxy_request: OxyRequest) -> OxyRequest:
    print(oxy_request.shared_data)
    user_query = oxy_request.get_query(master_level=True)
    current_query = oxy_request.get_query()
    print(user_query + "\n" + current_query)
    oxy_request.arguments["who"] = oxy_request.callee
    return oxy_request


def func_filter(payload: dict):
    payload["shared_data"] = {"userKey": "75064", "uuid": "7af2bbc072d849298af3a71a1a38b4d3"}
    print(payload)
    return payload


def format_output(oxy_response: OxyResponse) -> OxyResponse:
    oxy_response.output = "Answer: " + oxy_response.output
    return oxy_response


oxy_space = [
    oxy.HttpLLM(
        name="Default_LLM",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
    ),
    # oxy.HttpLLM(
    #     name = 'DeepSeek_V3',
    #     api_key = 'bffe6499813b4586a99ecba9f018e4da',
    #     base_url = 'http://easyalgo.jd.com/openapi/deepseek/v1/chat/completions',
    #     model_name = 'DeepSeek-V3',
    #     llm_params = {
    #         'temperature': 0.01
    #     },
    #     semaphore=4,
    #     timeout = 240
    # ),
    oxy.HttpLLM(
        name='OxyGen_LLM', # GPT-4o
        api_key = '0ff96343-972f-43be-94d2-5579a29d6912',
        base_url = 'http://gpt-proxy.jd.com/gateway/common',
        model_name = 'gpt-4o-0806',
        llm_params = {
            'temperature': 0.01
        },
        semaphore=4,
        timeout = 240
    ),
    oxy.ReActAgent(
        name="master_agent",
        sub_agents=[],
        tools=[],
        additional_prompt="You may get several types of tasks, please choose correct tools to finish tasks.",
        is_master=True,
        func_format_output=format_output,
        timeout=100,
        llm_model="OxyGen_LLM",
        top_k_tools=3,   # 仅召回 top 3 个工具
    ),
]


async def main():
    """
    new instance:
        Method 1: async with MAS(oxy_space=oxy_space) as mas:
        Method 2: mas = await MAS.create(oxy_space=oxy_space)
    call directly:
        await mas.call(callee = 'joke_tool', arguments = {'joke_type': 'comic'})
    call as a cli:
        await mas.start_cli_mode(first_query='Get what time it is and save in `log.txt` under `/local_file`')
        await mas.start_web_service(first_query='Get what time it is and save in `log.txt` under `/local_file`')
        await mas.start_batch_processing(['Hello', 'What time is it now', '200 positions of Pi', 'Get what time it is and save in `log.txt` under `/local_file`'])
    """
    async with MAS(oxy_space=oxy_space, func_filter=func_filter) as mas:
        await mas.start_contest_mode(
            competition_id='102',
            token="""a9fc93235a9344cfa08318a634198736@102""")


if __name__ == "__main__":
    asyncio.run(main())
