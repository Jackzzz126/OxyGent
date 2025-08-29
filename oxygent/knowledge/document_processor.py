
from oxygent.oxy.base_oxy import Oxy
from oxygent.schemas import OxyRequest, OxyResponse
from .knowledge_unit import KnowledgeUnit
from typing import List, Dict
import fitz
import docx
import time
import uuid
import markdown
from bs4 import BeautifulSoup

# todo 支持多模态 -> 抽象
# 继承对象Oxy的利弊
class DocumentProcessor(Oxy):
    """文档处理组件，实现智能分段与元数据提取"""
    category: str = "kb_processor"
    supported_formats: List[str] = ["pdf", "docx", "txt", "md"]

    async def _parse_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    async def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    async def _parse_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    async def _parse_md(self, file_path: str) -> str:
        """解析Markdown并提取纯文本内容（去除格式标签）"""
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        # 转换为HTML后提取纯文本
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    async def _split_into_chunks(self, content: str, chunk_size: int = 500) -> Dict[str, str]:
        """基于语义的分段逻辑，参考RAGFlow的分段算法"""
        chunks = {}
        paragraphs = content.split("\n\n")
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)
            if current_length + para_length > chunk_size and current_chunk:
                chunk_id = f"chunk_{uuid.uuid4()}"
                chunks[chunk_id] = "\n\n".join(current_chunk)
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length

        if current_chunk:
            chunk_id = f"chunk_{uuid.uuid4()}"
            chunks[chunk_id] = "\n\n".join(current_chunk)

        return chunks

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        file_path = oxy_request.arguments.get("file_path")
        owner = oxy_request.arguments.get("owner", "system")
        document_id = oxy_request.arguments.get("document_id", f"doc_{uuid.uuid4()}")

        # 1. 解析文档
        ext = file_path.split(".")[-1].lower()
        if ext not in self.supported_formats:
            return OxyResponse(
                state="error",
                output=f"Unsupported format: {ext}, supported: {self.supported_formats}"
            )

        parse_method = getattr(self, f"_parse_{ext}", None)
        if not parse_method:
            return OxyResponse(state="error", output=f"No parser for {ext}")

        content = await parse_method(file_path)

        # 2. 分段处理
        chunks = await self._split_into_chunks(content)

        # 3. 生成知识单元
        knowledge_units = [
            KnowledgeUnit(
                content=chunk,
                chunk_id=chunk_id,
                document_id=document_id,
                metadata={
                    "source": file_path,
                    "format": ext,
                    "length": len(chunk)
                },
                owner=owner,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S")
            ) for chunk_id, chunk in chunks.items()
        ]

        return OxyResponse(
            state="completed",
            output=[ku.dict() for ku in knowledge_units]
        )