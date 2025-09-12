import json
import csv
from pathlib import Path
from typing import Union,Dict,Any,List
import asyncio

from .base import BaseLoader
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.logging import LogContext


class DocumentLoader(BaseLoader):
    """Loader for text-base document formats.
    
    Supports: TXT,MD,HTML,JSON,CSV,DOCX"""
    def __init__(self):
        super().init__()
        self.supported_formats=[
            DocumentType.TXT,
            DocumentType.MD,
            DocumentType.HTML,
            DocumentType.JSON,
            DocumentType.CSV,
            DocumentType.DOCX,
        ]

    async def load_single(
        self,
        source: Union[str,Path],
        encoding: str="utf-8",
        **kwargs
    )->Document:
        """
        Load a single text-based document.

        Args:
            source: File path
            encoding: text encoding
            **kwargs: Additional options

        Return:
            Loaded document
        """
        source=Path(source)

        if not source.exists():
            raise DocumentLoadError(f"File not found: {source}")
        doc_type=self.get_document_type(source)

        with LogContext("load_document", file=str(source),type=doc_type):
            try:
                #Route to appropriate loader method
                if doc_type==DocumentType.TXT:
                    content=await self._load_text(source,encoding)
                elif doc_type==DocumentType.MD:
                    content=await self._load_markdown(source,encoding)
                elif doc_type==DocumentType.HTML:
                    content=await self._load_html(source,encoding)
                elif doc_type==DocumentType.JSON:
                    content=await self._load_json(source,encoding)
                elif doc_type==DocumentType.CSV:
                    content=await self._load_csv(source,encoding)
                elif doc_type==DocumentType.DOCX:
                    content=await self._load_docx(source,encoding)
                else:
                    raise DocumentLoadError(f"Unsupported format: {doc_type}")

                #create document
                title = source.stem
                document=self.create_document(
                    title=title,
                    content=content,
                    source=source,
                    doc_type=doc_type,
                    encoding=encoding,
                    extraction_method=f"DocumentLoader.{doc_type.value}"
                )
                return document
            
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load {doc_type.value} file: {source}"
                    file_path=str(source),
                    cause=e
                )
            
    async def _load_text(self,path: Path,encoding:str)->str:
        """Load plain text file."""
        return await asyncio.to_thread(path.read_text,encoding=encoding)
    
    async def _load_markdown(self,path:Path, encoding:str)->str:
        """Load Markdown file."""
        # For now , treat as plain text
        #In a full implementaion , you might convert to HTML or extract metadata
        return await asyncio.to_thread(path.read_text, encoding=encoding)
    
    async def _load_html(self,path:Path , encoding:str)->str:
        """Load HTML file and extract text content."""
        def _extract_html_text():
            try:
                from bs4 import BeautifulSoup
                with open(path,'r',encoding=encoding) as f

