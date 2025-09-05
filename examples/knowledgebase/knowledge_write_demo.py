import asyncio
from oxygent import OxyRequest  # 导入OxyRequest构建请求参数
from oxygent.databases.db_vector_v2.vearch_db import VearchDB
from oxygent.knowledge.document_processor import DocumentProcessor
from oxygent.knowledge.embeddings.modelscope_embedding import ModelScopeEmbedding
from oxygent.knowledge.knowledge_base import KnowledgeBase, OxyState  # 导入OxyState用于状态判断


async def main():
    # --------------------------
    # 1. 初始化知识库依赖组件
    # --------------------------
    print("=" * 50)
    print("开始初始化知识库组件...")

    # Vearch数据库配置（根据实际环境调整）
    vearch_config = {
        "router_url": "http://ocre-aigc-router.vectorbase.svc.lf09.n.jd.local",
        "master_url": "http://ocre-aigc-master.vectorbase.svc.lf09.n.jd.local",
        "db_name": "qqd_db_rag_base",
        "tool_df_space_name": "guojian31_bdagent_tool_desc_test1"
    }

    try:
        # 初始化Vearch客户端
        vearch_db = VearchDB(config=vearch_config)
        print(f"✅ Vearch数据库客户端初始化成功（DB: {vearch_config['db_name']}）")

        # 在初始化知识库后添加
        create_result = await vearch_db.create_collection(
            collection_name="jd_knowledge_123",
            dimension=1024,
            metric_type="L2"
        )
        print(f"手动创建集合结果：{create_result}")  # 若为False，说明Vearch配置或权限有问题

        # 初始化文档处理器
        doc_processor = DocumentProcessor()
        print(f"✅ 文档处理器初始化成功（支持格式：{doc_processor.supported_formats}）")

        # 初始化嵌入模型（Qwen3-Embedding-0.6B）
        embedding_model = ModelScopeEmbedding(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            device="cpu"  # 根据实际环境选择"cuda"或"cpu"
        )
        # 验证嵌入模型维度（确保模型加载成功）
        if hasattr(embedding_model, "dimension") and embedding_model.dimension > 0:
            print(f"✅ 嵌入模型初始化成功（模型：Qwen3-Embedding-0.6B，维度：{embedding_model.dimension}）")
        else:
            raise RuntimeError("嵌入模型维度无效，可能模型加载失败")

        # 初始化知识库核心实例
        knowledge_base = KnowledgeBase(
            vector_db=vearch_db,
            embedding_model=embedding_model,
            doc_processor=doc_processor,
            collection_name="jd_knowledge"  # 目标Vearch集合名称
        )
        print(f"✅ 知识库实例初始化成功（集合：{knowledge_base.collection_name}）")

    except Exception as e:
        print(f"❌ 知识库组件初始化失败：{str(e)}")
        return  # 初始化失败则退出

    # --------------------------
    # 2. 初始化知识库（创建集合）
    # --------------------------
    print("\n" + "=" * 50)
    print("开始初始化知识库（确保集合存在）...")
    try:
        init_success = await knowledge_base.initialize()
        if init_success:
            print(f"✅ 知识库初始化完成（集合已就绪：{knowledge_base.collection_name}）")
        else:
            print(f"⚠️  知识库初始化未完成（可能集合已存在或创建失败）")
    except Exception as e:
        print(f"❌ 知识库初始化失败：{str(e)}")
        return

    # --------------------------
    # 3. 准备TXT文件添加参数
    # --------------------------
    print("\n" + "=" * 50)
    # 待添加的TXT文件路径（根据实际路径调整）
    file_path = "/Users/gechunfa1/Documents/jd-opensource-code/raw_ai/OxyGent/examples/knowledgebase/data.txt"
    # 自定义文档ID（可选，不提供则自动生成）
    custom_document_id = f"doc_{file_path.split('/')[-1].replace('.txt', '')}_{int(time.time())}"
    # 文档拆分大小（默认500字符/段，可根据需求调整）
    chunk_size = 500

    print(f"准备添加TXT文件：{file_path}")
    print(f"自定义文档ID：{custom_document_id}")
    print(f"拆分大小：{chunk_size}字符/段")

    # --------------------------
    # 4. 构建OxyRequest请求
    # --------------------------
    # 按KnowledgeBase.add_document要求构建参数（需包含file_path）
    request_args = {
        "file_path": file_path,
        "document_id": custom_document_id,  # 可选参数
        "chunk_size": chunk_size,  # 可选参数
        "owner": "test_user"  # 可选参数（文档所有者）
    }
    oxy_request = OxyRequest(arguments=request_args)
    print(f"✅ OxyRequest请求构建完成（参数：{list(request_args.keys())}）")

    # --------------------------
    # 5. 执行TXT文件添加
    # --------------------------
    print("\n开始添加TXT文件到知识库...")
    try:
        add_response = await knowledge_base.add_document(oxy_request)
        # 根据OxyState判断添加结果
        if add_response.state == OxyState.COMPLETED:
            print(f"✅ TXT文件添加成功！{add_response.output}")
        else:
            print(f"❌ TXT文件添加失败：{add_response.output}")
            # 若添加失败，可尝试清理已生成的空文档（可选）
            print("尝试清理无效文档...")
            await knowledge_base.delete_document(custom_document_id)
            print("清理完成")
            return
    except Exception as e:
        print(f"❌ TXT文件添加过程异常：{str(e)}")
        return

    # --------------------------
    # 6. 验证添加结果（可选）
    # --------------------------
    print("\n" + "=" * 50)
    print("开始验证添加结果...")

    # 验证1：列出所有文档ID，确认目标文档存在
    print("\n1. 验证文档是否存在于知识库中...")
    doc_ids = await knowledge_base.list_documents()
    if custom_document_id in doc_ids:
        print(f"✅ 验证通过：文档ID {custom_document_id} 已存在于知识库")
    else:
        print(f"❌ 验证失败：文档ID {custom_document_id} 未找到")
        return

    # 验证2：基于文档内容关键词检索，确认可查询到
    print("\n2. 基于文档关键词检索验证...")
    # 读取TXT文件前10个字符作为关键词（或自定义关键词）
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample_text = f.read(10).strip()  # 取前10个字符作为检索关键词
        if sample_text:
            print(f"检索关键词：'{sample_text}'")
            retrieve_result = await knowledge_base.retrieve(query=sample_text, top_k=3)
            if retrieve_result["state"] == "completed" and retrieve_result["output"]:
                print(f"✅ 检索成功！找到{len(retrieve_result['output'])}条相关知识单元")
                # 打印第一条检索结果的内容片段
                first_result = retrieve_result["output"][0]
                print(f"第一条结果片段：{first_result['content'][:50]}...")
            else:
                print(f"⚠️  检索无结果：{retrieve_result['output']}")
        else:
            print("⚠️  TXT文件内容为空，跳过检索验证")
    except Exception as e:
        print(f"⚠️  检索验证失败：{str(e)}")

    # --------------------------
    # 7. 后续操作（可选）
    # --------------------------
    print("\n" + "=" * 50)
    print("TXT文件添加流程全部完成！")
    print(f"最终状态：文档 {custom_document_id} 已成功存入知识库")
    print(f"后续可通过 knowledge_base.retrieve(query='关键词') 查询相关内容")


# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":
    import time  # 用于生成时间戳文档ID

    # 运行异步主函数
    asyncio.run(main())