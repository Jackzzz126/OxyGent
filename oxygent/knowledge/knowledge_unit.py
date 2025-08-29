
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid

class KnowledgeUnit(BaseModel):
    """知识单元模型，对应RAGFlow中的文档片段与元数据组合"""
    id: str = str(uuid.uuid4())
    content: str
    chunk_id: str
    document_id: str
    metadata: Dict = {}
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""
    owner: str = ""