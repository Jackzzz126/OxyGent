import asyncio
import time
import uuid
import numpy as np  # 确保导入numpy
from typing import List, Dict, Optional, Any

from .knowledge_unit import KnowledgeUnit
from .document_processor import DocumentProcessor
from oxygent.databases.db_vector_v2.base_vector_db import BaseVectorDB
from oxygent.knowledge.embeddings.abc_embedding import AbstractEmbedding
from .. import OxyRequest, OxyResponse, OxyState  # 导入OxyState枚举


class KnowledgeBase:
    """知识库核心组件，修复与BaseVectorDB和DocumentProcessor的接口适配问题"""

    def __init__(
            self,
            vector_db: BaseVectorDB,
            embedding_model: AbstractEmbedding,
            doc_processor: Optional[DocumentProcessor] = None,
            collection_name: str = "default_knowledge_collection"
    ):
        # 新增：参数类型校验，确保依赖组件符合抽象基类规范
        if not isinstance(vector_db, BaseVectorDB):
            raise TypeError(f"vector_db must be BaseVectorDB instance, got {type(vector_db)}")
        if not isinstance(embedding_model, AbstractEmbedding):
            raise TypeError(f"embedding_model must be AbstractEmbedding instance, got {type(embedding_model)}")

        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.doc_processor = doc_processor or DocumentProcessor()
        self.collection_name = collection_name
        self._initialized = False  # 初始化状态标记

    async def initialize(self):
        """显式初始化方法，需在事件循环中调用"""
        if not self._initialized:
            await self._ensure_collection_exists()
            self._initialized = True
        return self._initialized

    async def _ensure_collection_exists(self):
        """确保集合存在，处理list_collections返回None的情况"""
        try:
            collections = await self.vector_db.list_collections()
            collections = collections or []  # 处理None情况

            if self.collection_name not in collections:
                # 验证嵌入模型维度
                if not hasattr(self.embedding_model, "dimension") or self.embedding_model.dimension <= 0:
                    print(f"警告：嵌入模型维度无效（{getattr(self.embedding_model, 'dimension', '未定义')}），无法创建集合")
                    return

                # 创建集合（严格遵循BaseVectorDB的参数规范）
                create_success = await self.vector_db.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.embedding_model.dimension,
                    # metric_type="COSINE"
                    metric_type = "L2"  # 替换为L2
                )
                if create_success:
                    print(f"成功创建集合：{self.collection_name}")
                else:
                    print(f"创建集合{self.collection_name}失败（vector_db返回False）")
        except Exception as e:
            print(f"确保集合存在失败：{str(e)}")

    async def _generate_embeddings(self, texts: List[str]) -> list[Any] | np.ndarray:
        """生成文本嵌入向量"""
        if not texts:
            return []
        try:
            return await self.embedding_model.embed(texts)
        except Exception as e:
            print(f"嵌入向量生成失败：{str(e)}")
            return []

    async def add_knowledge_units(self, units: List[KnowledgeUnit]) -> Dict[str, Any]:
        """添加知识单元到向量数据库"""
        if not units:
            return {"state": "completed", "output": "No knowledge units to add"}

        # 生成嵌入向量
        embeddings = await self._generate_embeddings([unit.content for unit in units])
        if len(embeddings) != len(units):
            return {
                "state": "error",
                "output": f"嵌入向量数量不匹配：{len(embeddings)}个向量对应{len(units)}个知识单元"
            }

        # 准备插入数据（确保vectors为np.ndarray或List[np.ndarray]，符合BaseVectorDB要求）
        ids = []
        vectors = []
        metadata_list = []
        for unit, embedding in zip(units, embeddings):
            unit.embedding = embedding
            unit_id = unit.id or f"ku_{uuid.uuid4()}"
            unit.created_at = unit.created_at or time.strftime("%Y-%m-%d %H:%M:%S")
            ids.append(unit_id)
            # 确保每个向量都是np.ndarray（避免原生list）
            vectors.append(np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding)
            metadata_list.append(unit.dict())

        # 执行插入（严格遵循BaseVectorDB的返回值规范：插入成功的ID列表）
        try:
            inserted_ids = await self.vector_db.insert(
                collection_name=self.collection_name,
                vectors=vectors,
                ids=ids,
                metadata=metadata_list
            )
            if not inserted_ids or len(inserted_ids) != len(ids):
                return {
                    "state": "error",
                    "output": f"插入失败：仅成功插入{len(inserted_ids)}/{len(ids)}个向量"
                }

            return {
                "state": "completed",
                "output": f"成功添加{len(inserted_ids)}个知识单元"
            }
        except Exception as e:
            return {"state": "error", "output": f"向量数据库插入失败：{str(e)}"}

    async def retrieve(
            self,
            query: str,
            top_k: int = 5,
            filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """检索与查询相关的知识单元"""
        # 生成查询嵌入（确保为np.ndarray）
        query_embedding = await self._generate_embeddings([query])
        if not query_embedding or not isinstance(query_embedding[0], (np.ndarray, list)):
            return {"state": "error", "output": "查询嵌入向量生成失败或格式无效"}

        # 执行向量搜索（严格遵循BaseVectorDB参数规范）
        try:
            search_result = await self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=np.array(query_embedding[0]),  # 转换为np.ndarray
                limit=top_k,
                filter=filters or {}
            )
        except Exception as e:
            return {"state": "error", "output": f"向量搜索失败：{str(e)}"}

        # 处理搜索结果（确保符合BaseVectorDB返回格式：List[Dict[str, Any]]）
        try:
            units = []
            for item in search_result:
                # 验证搜索结果是否包含必要的"metadata"字段
                if "metadata" not in item:
                    raise KeyError("搜索结果缺少'metadata'字段")
                units.append(KnowledgeUnit(**item["metadata"]))

            return {
                "state": "completed",
                "output": [{"content": unit.content, "metadata": unit.dict()} for unit in units]
            }
        except Exception as e:
            return {"state": "error", "output": f"搜索结果格式错误：{str(e)}"}

    async def list_documents(self) -> List[str]:
        """列出所有文档ID（修复random_vector类型问题）"""
        try:
            # 生成随机向量（转换为np.ndarray，符合BaseVectorDB要求）
            random_vector = np.random.rand(self.embedding_model.dimension).astype(np.float32)

            # 执行全量搜索（limit设为10000确保覆盖所有数据）
            search_result = await self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=random_vector,
                limit=10000,
                filter={}
            )

            # 提取并去重document_id
            doc_ids = set()
            for item in search_result:
                metadata = item.get("metadata", {})
                doc_id = metadata.get("document_id")
                if doc_id:
                    doc_ids.add(doc_id)

            print(f"找到文档ID：{list(doc_ids)}")
            return list(doc_ids)
        except Exception as e:
            print(f"列出文档失败：{str(e)}")
            return []

    async def add_document(self, request: OxyRequest) -> OxyResponse:
        """添加文档并拆分存入知识库（复用DocumentProcessor.process，修复状态码）"""
        try:
            # 1. 验证必传参数
            if "file_path" not in request.arguments:
                return OxyResponse(
                    state=OxyState.FAILED,  # 使用OxyState枚举
                    output="缺少必传参数：file_path"
                )

            # 2. 提取参数
            file_path = request.arguments["file_path"]
            document_id = request.arguments.get("document_id", f"doc_{uuid.uuid4()}")
            chunk_size = request.arguments.get("chunk_size", 500)
            owner = request.arguments.get("owner", "system")

            # 3. 复用DocumentProcessor.process处理文档（支持多格式、自动拆分、元数据提取）
            process_result = await self.doc_processor.process(
                file_path=file_path,
                owner=owner,
                document_id=document_id,
                chunk_size=chunk_size
            )

            # 4. 处理文档处理结果
            if process_result["state"] != "completed":
                return OxyResponse(
                    state=OxyState.FAILED,
                    output=f"文档处理失败：{process_result['output']}"
                )

            # 5. 转换为KnowledgeUnit实例
            knowledge_units = [KnowledgeUnit(**ku_dict) for ku_dict in process_result["output"]]
            if not knowledge_units:
                return OxyResponse(
                    state=OxyState.FAILED,
                    output="文档处理后未生成任何知识单元"
                )

            # 6. 添加到向量数据库
            add_result = await self.add_knowledge_units(knowledge_units)
            if add_result["state"] == "completed":
                return OxyResponse(
                    state=OxyState.COMPLETED,  # 使用OxyState枚举
                    output=f"文档添加成功（ID: {document_id}），共拆分{len(knowledge_units)}个片段"
                )
            else:
                return OxyResponse(
                    state=OxyState.FAILED,
                    output=f"文档添加失败：{add_result['output']}"
                )

        except Exception as e:
            return OxyResponse(
                state=OxyState.FAILED,
                output=f"处理文档时出错：{str(e)}"
            )

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """删除指定文档的所有知识单元（严格遵循BaseVectorDB的delete参数规范）"""
        try:
            # BaseVectorDB.delete要求ids和filter二选一，这里仅使用filter
            deleted_count = await self.vector_db.delete(
                collection_name=self.collection_name,
                ids=[],  # 传入空列表满足参数要求
                filter={"document_id": document_id}
            )
            return {
                "state": "completed",
                "output": f"成功删除文档{document_id}的{deleted_count}个知识单元"
            }
        except Exception as e:
            return {"state": "error", "output": f"删除文档失败：{str(e)}"}