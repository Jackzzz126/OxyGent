import asyncio

from oxygent import MAS, oxy
from oxygent.utils.env_utils import get_env_var

master_prompt = """
你是一个学识渊博，态度友好，说话幽默的AI助理 
你勇于承认自己的无知，遇到没把握的问题时，会大方承认自己不知道
你一般总是说中文
"""

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=get_env_var("DEFAULT_LLM_API_KEY"),
        base_url=get_env_var("DEFAULT_LLM_BASE_URL"),
        model_name=get_env_var("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
        timeout=240,
    ),
    oxy.ReActAgent(
        name="master_agent",
        prompt = master_prompt,
        is_master=True,
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        history = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            user_in = input("User: ").strip()
            if user_in.lower() in {"exit", "quit", "q"}:
                break

            print(f"输入1：{user_in}")

            history.append({"role": "user", "content": user_in})
            print(f"输入2：{history}")
            result = await mas.call(
                callee="master_agent",
                arguments={"query": history},
            )
            assistant_out = result
            print(f"Assistant: {assistant_out}\n")
            history.append({"role": "assistant", "content": assistant_out})

# async def main():
#     async with MAS(oxy_space=oxy_space) as mas:
#         await mas.start_web_service(
#             first_query="Hello!"
#         )
if __name__ == "__main__":
    asyncio.run(main())