
from oxygent.oxy.base_oxy import Oxy
from oxygent.schemas import OxyRequest, OxyResponse
from .knowledge_unit import KnowledgeUnit
from typing import List, Dict, Optional

# 支持多模态 -> 知识库处理核心单元
# 操作向量数据库: 单独定义一套抽象的方法 (es/vearch/mivlus/....)

class KnowledgeBase(Oxy):
    """知识库核心组件，融合RAGFlow的检索与存储能力"""
    category: str = "knowledge_base"
    vector_db: str  # 向量数据库Oxy名称
    doc_processor: str  # 文档处理器Oxy名称
    embedding_model: str  # 嵌入模型Oxy名称
    vector_db_proxy: str # 增删查改

    # 生成词向量
    async def _generate_embeddings(self, oxy_request: OxyRequest, texts: List[str]) -> List[List[float]]:
        """调用嵌入模型生成向量"""
        # 真实可以调用的invoker
        # embedding_model.invoke -> 抽象通用的方法: embedding(content, ..) -> 封装一套modelscope & huggingface api加载/调用embedding_model api
        response = await oxy_request.call(
            callee=self.embedding_model,
            arguments={"texts": texts}
        )
        return response.output

    async def _add_knowledge_units(self, oxy_request: OxyRequest, units: List[KnowledgeUnit]):
        """添加知识单元到向量数据库"""
        # 生成嵌入向量
        embeddings = await self._generate_embeddings(
            oxy_request,
            [unit.content for unit in units]
        )

        # 批量插入向量库
        for unit, embedding in zip(units, embeddings):
            unit.embedding = embedding
            # no.2 -> 调用向量数据库
            await oxy_request.call(
                callee=self.vector_db,
                arguments={
                    "operation": "insert",
                    "id": unit.id,
                    "vector": embedding,
                    "metadata": unit.dict()
                }
            )

    async def _retrieve(self, oxy_request: OxyRequest, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> \
    List[KnowledgeUnit]:
        """混合检索实现"""
        # 生成查询向量
        query_embedding = await self._generate_embeddings(oxy_request, [query])
        if not query_embedding:
            return []

        # 向量检索
        response = await oxy_request.call(
            callee=self.vector_db,
            arguments={
                "operation": "search",
                "vector": query_embedding[0],
                "top_k": top_k,
                "filters": filters or {}
            }
        )

        # 转换为KnowledgeUnit
        return [KnowledgeUnit(**item["metadata"]) for item in response.output]

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        operation = oxy_request.arguments.get("operation")

        if operation == "add_document":
            # 调用文档处理器处理文件
            process_response = await oxy_request.call(
                callee=self.doc_processor,
                arguments=oxy_request.arguments
            )

            if process_response.state != "completed":
                return OxyResponse(state="error", output=process_response.output)

            # 转换为KnowledgeUnit对象
            units = [KnowledgeUnit(**data) for data in process_response.output]

            # 添加到知识库
            await self._add_knowledge_units(oxy_request, units)
            return OxyResponse(
                state="completed",
                output=f"Added {len(units)} knowledge units from document"
            )

        elif operation == "retrieve":
            results = await self._retrieve(
                oxy_request,
                query=oxy_request.arguments.get("query"),
                top_k=oxy_request.arguments.get("top_k", 5),
                filters=oxy_request.arguments.get("filters")
            )
            return OxyResponse(
                state="completed",
                output=[unit.dict() for unit in results]
            )

        elif operation == "delete_document":
            document_id = oxy_request.arguments.get("document_id")
            #
            # vector_db_proxy(...)
            # await oxy_request.call(
            #     callee=self.vector_db,
            #     arguments={
            #         "operation": "delete",
            #         "filters": {"document_id": document_id}
            #     }
            # )
            return OxyResponse(state="completed", output=f"Deleted document {document_id}")

        else:
            return OxyResponse(state="error", output=f"Unknown operation: {operation}")