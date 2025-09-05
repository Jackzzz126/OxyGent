from oxygent.databases.db_vector_v2.vearch_db import VearchDB

# 配置参数（使用提供的现有配置）
vearch_config = {
    "router_url": "http://ocre-aigc-router.vectorbase.svc.lf09.n.jd.local",
    "master_url": "http://ocre-aigc-master.vectorbase.svc.lf09.n.jd.local",
    "db_name": "qqd_db_rag_base",
    # 可选：如果需要使用工具相关功能可以添加
    "tool_df_space_name": "guojian31_bdagent_tool_desc_test1",
    "embedding_model_url": "http://big-data-emb.jd.local/v2/models/embedding/infer"
}


# 实例化VearchDB对象
async def create_vearch_instance():
    # 初始化VearchDB实例
    vearch_db = VearchDB(config=vearch_config)

    # 可选：验证数据库连接（通过尝试列出集合）
    try:
        collections = await vearch_db.list_collections()
        print(f"成功连接到Vearch，现有集合: {collections}")
        return vearch_db
    except Exception as e:
        print(f"连接Vearch失败: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    import asyncio

    vearch_instance = asyncio.run(create_vearch_instance())
    # 现在可以使用vearch_instance调用各种方法
    # 例如：await vearch_instance.create_collection(...)