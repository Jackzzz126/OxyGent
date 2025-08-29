
from oxygent.oxy.base_tool import BaseTool
from typing import List, Dict, Optional
from pydantic import Field


class KnowledgeBaseTool(BaseTool):
    """知识库管理工具"""
    name: str = "knowledge_base_tool"
    desc: str = "Tool for managing knowledge base, including adding documents and querying knowledge"
    kb_name: str = Field(..., description="Name of the knowledge base to operate")

    async def add_documents(self, file_paths: List[str], owner: str = "system") -> str:
        """添加文档到知识库"""
        results = []
        for path in file_paths:
            resp = await self.mas.call_oxy(
                callee=self.kb_name,
                arguments={
                    "operation": "add_document",
                    "file_path": path,
                    "owner": owner
                }
            )
            results.append(f"File {path}: {resp.output}")
        return "\n".join(results)

    async def query_knowledge(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """查询知识库内容"""
        resp = await self.mas.call_oxy(
            callee=self.kb_name,
            arguments={
                "operation": "retrieve",
                "query": query,
                "top_k": top_k,
                "filters": filters
            }
        )
        return resp.output

    async def delete_document(self, document_id: str) -> str:
        """删除知识库中的文档"""
        resp = await self.mas.call_oxy(
            callee=self.kb_name,
            arguments={
                "operation": "delete_document",
                "document_id": document_id
            }
        )
        return resp.output