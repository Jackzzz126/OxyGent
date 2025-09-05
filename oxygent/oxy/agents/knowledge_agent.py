
'''
实现了两类智能体：
KnowledgeBaseAgent：底层知识库代理，负责文档的添加、检索、列表管理，直接对接向量数据库和嵌入模型。
KnowledgeAugmentedReActAgent：上层增强型问答代理，基于ReAct框架，通过调用知识库检索知识，再结合LLM生成精准回答。

两者协同工作 存储->检索->生成 的RAG闭环。
'''

from typing import Optional, Dict, Any, List
from pydantic import Field, ConfigDict

from oxygent.oxy import ReActAgent
from oxygent.oxy.base_oxy import Oxy
from oxygent.schemas import OxyRequest, OxyResponse


class KnowledgeBaseAgent(Oxy):
    """知识库代理：修复空值和异常处理"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    category: str = "knowledge_agent"
    knowledge_base: Any = Field(..., description="知识库实例")

    def __init__(
            self,
            name: str,
            vector_db: Any,
            embedding_model: Any,
            doc_processor: Optional[Any] = None,
            collection_name: str = "default_knowledge_collection",
            **kwargs
    ):
        from oxygent.knowledge.knowledge_base import KnowledgeBase
        knowledge_base = KnowledgeBase(
            vector_db=vector_db,
            embedding_model=embedding_model,
            doc_processor=doc_processor,
            collection_name=collection_name  # 传递集合名称
        )
        super().__init__(
            name=name,
            knowledge_base=knowledge_base,
            **kwargs
        )

    async def _add_document(self, request: OxyRequest) -> OxyResponse:
        try:
            if "file_path" not in request.arguments:
                return OxyResponse(state=3, output="缺少必传参数：file_path")

            await self.knowledge_base.add_document(
                file_path=request.arguments["file_path"],
                document_id=request.arguments.get("document_id"),
                chunk_size=request.arguments.get("chunk_size", 500)
            )
            return OxyResponse(state=1, output="文档添加成功")
        except FileNotFoundError:
            return OxyResponse(state=4, output="文件不存在")
        except Exception as e:
            return OxyResponse(state=2, output=f"添加失败：{str(e)}")

    async def _retrieve(self, request: OxyRequest) -> OxyResponse:
        try:
            if "query" not in request.arguments:
                return OxyResponse(state=3, output="缺少必传参数：query")

            results = await self.knowledge_base.retrieve(
                query=request.arguments["query"],
                top_k=request.arguments.get("top_k", 5)
            )

            # 修复：增加结果空值判断
            content = []
            if results and isinstance(results, dict) and "hits" in results:
                for hit in results["hits"]:
                    if hit and isinstance(hit, dict):
                        content.append(hit.get("content", ""))

            return OxyResponse(state=1, output=content)
        except Exception as e:
            return OxyResponse(state=2, output=f"检索失败：{str(e)}")

    async def _list_documents(self, request: OxyRequest) -> OxyResponse:
        """修复：直接调用知识库的list_documents方法"""
        try:
            # 优先使用知识库的list_documents（已修复）
            doc_ids = await self.knowledge_base.list_documents()
            return OxyResponse(state=1, output=doc_ids)
        except Exception as e:
            return OxyResponse(state=2, output=f"列表失败：{str(e)}")

    async def _execute(self, request: OxyRequest) -> OxyResponse:
        """确保始终返回OxyResponse"""
        try:
            operation = request.arguments.get("operation")
            handlers = {
                "add_document": self._add_document,
                "retrieve": self._retrieve,
                "list_documents": self._list_documents
            }

            if operation not in handlers:
                return OxyResponse(state=3, output=f"不支持的操作：{operation}")

            # 调用处理器并确保返回OxyResponse
            result = await handlers[operation](request)
            if isinstance(result, OxyResponse):
                return result
            else:
                return OxyResponse(state=1, output=str(result))
        except Exception as e:
            return OxyResponse(state=2, output=f"执行失败：{str(e)}")


class KnowledgeAugmentedReActAgent(ReActAgent):
    """带知识库的ReAct智能体：增强回答生成逻辑"""
    knowledge_agent_name: str = Field(..., description="知识库代理名称")

    async def _reasoning_step(self, query: str) -> str:
        # 1. 调用知识库检索
        retrieve_resp = await self.mas.call(
            callee=self.knowledge_agent_name,
            caller=self.name,
            arguments={"operation": "retrieve", "query": query}
        )

        # 2. 处理检索结果
        if isinstance(retrieve_resp, OxyResponse) and retrieve_resp.state == 1:
            knowledge_text = "\n".join([f"- {c}" for c in retrieve_resp.output if c]) or "未检索到相关知识"
        else:
            err_msg = retrieve_resp.output if isinstance(retrieve_resp, OxyResponse) else str(retrieve_resp)
            knowledge_text = f"知识检索失败：{err_msg}"

        # 3. 生成回答（增强提示词逻辑）
        prompt = f"""基于以下参考知识回答用户问题：{knowledge_text}用户问题：{query} 回答要求：1. 仅使用参考知识中的信息，不要编造内容;2. 如果没有相关知识，直接说明"没有找到相关信息";3. 保持回答简洁明了"""
        llm_resp = await self.llm.generate(prompt=prompt)
        return llm_resp.output if hasattr(llm_resp, "output") else str(llm_resp)
