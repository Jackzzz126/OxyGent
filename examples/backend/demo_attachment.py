import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        is_multimodal_supported=True,
    ),
    oxy.ChatAgent(
        name="qa_agent",
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        payload = {
            "query": "Introduce the content of the file",
            "attachments": ["README.md"],
        }
        oxy_response = await mas.chat_with_agent(payload=payload)
        print("LLM: ", oxy_response.output)


if __name__ == "__main__":
    asyncio.run(main())
