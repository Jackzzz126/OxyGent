import asyncio
import os

from oxygent import MAS, Config, oxy, preset_tools

Config.set_agent_llm_model("default_llm")

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    preset_tools.sql_tools,
    oxy.ReActAgent(
        name="database_agent",
        desc="A tool that can analyze the database",
        tools=["sql_tools"],
    ),
    preset_tools.file_tools,
    oxy.ReActAgent(
        name="file_agent",
        desc="A tool that can operate the file system",
        tools=["file_tools"],
    ),
    oxy.ReActAgent(
        is_master=True,
        name="master_agent",
        sub_agents=["database_agent", "file_agent"],
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="hello"
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
delete from used_order_latest where create_time < '2025-11-06 00:00:00';
这是一个慢sql语句，执行以下操作：
1. 分析这个语句，提练出其中的表名
2. 连接数据库，根据上一步中的表名，查询表的结构及索引情况
3. 结合慢sql语句和表结构，索引情况，分析出慢sql语句可能得原因和优化建议
4. 生成一个result.csv文件,第一列是慢sql语句，第二例是表名，地三列是表结构，第四列是表的索引情况，第五列是慢sql慢的原因分析，第六列是优化建议
"""
