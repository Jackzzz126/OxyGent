import asyncio, os


from oxygent import MAS, Config, OxyResponse, oxy
Config.set_agent_llm_model("default_llm")


def format_output(oxy_response: OxyResponse) -> OxyResponse:
    oxy_response.output = oxy_response.output
    return oxy_response


oxy_space = [
    oxy.HttpLLM(
        name="OxyGent_LLM",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
    ),
    oxy.ReActAgent(
        name="master_agent",
        sub_agents=[],
        tools=[],
        prompt="""角色：你是一个猜题者，正在参与一个猜商品游戏。你不知道目标商品（<sku_name> 可能具体到型号）是什么，你的任务是通过提出是否疑问句来逐步缩小范围，最终猜出它。

规则：

1.提问形式：你每次只能向出题者提出一个简洁、明确的是非疑问句（即答案仅为“是”、“否”、“不确定”或“无法回答”的问题）。

2.提问策略：你的问题应具有策略性，通常从广泛类别（如“这是实物商品吗？”）开始，逐步过渡到具体属性（如“它需要电力驱动吗？”），以高效地锁定目标。

3.信息处理：你必须严格根据出题者之前的回答来构思后续问题。切勿臆造或忽略已知信息。

4.获胜条件：当你足够确信自己猜出答案时，请直接说出你认为的商品名称（例如：“它是手机吗？”）。这是唯一允许你不以提问形式进行交流的情况。

5.禁止行为：

∙不得询问或猜测与商品属性无关的内容。

∙不得要求出题者提供任何形式的提示或描述。

∙不要问相同的问题, 一个问题仅需问一次。

∙不得在提问中预先假设答案（例如，避免“它是不是一个很贵的XX？”这样的复合问题）。

最终目标：用尽可能少的问题猜中商品<sku_name>, 一定要先更宽泛更广大类别。

现在游戏开始，请提出你的第一个问题, 仅输出问题, 你的回答仅限提问形式。

输出格式：

Problem:{Your problem}""",
        is_master=True,
        func_format_output=format_output,
        timeout=100,
        llm_model="OxyGent_LLM",
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
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_contest_mode(
            token="""Your token""")


if __name__ == "__main__":
    asyncio.run(main())
