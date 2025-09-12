import asyncio
from pathlib import Path
from typing import Union

from .base import BaseLoader
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.logging import LogContext


class PDFLoader(BaseLoader):
    """Loader for PDF documents."""
    def __init__(self):
        super().__init__()
        self.supported_formats=[DocumentType.PDF]

    async def load_single(
        self,
        source:Union[str,Path],
        **kwargs
    )->Document:
        """Load a PDF document.
        
        Args:
            source:PDF file path
            ***kwargs: Additional options
            
        Returns:
            Loaded document
        """
        source=Path(source)

        if not source.exists():
            raise DocumentLoadError(f"File not found: {source}")
        with LogContext("load_pdf", file=str(source)):
            try:
                content=await self._extract_pdf_text(source)

                document=self.create_document(
                    title=source.stem,
                    content=content,
                    source=source,
                    doc_type=DocumentType.PDF,
                    extraction_mathod="PDFLoader.pymnpdf",
                )
                return document
            
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load PDF file:{source}",
                    file_path=str(source),
                    cause=e
                )
            
    async def _extract_pdf_text(self,path:Path)-> str:
        """Extract text from PDF usinh PyMuPDF."""
        def _extract_text():
            try:
                import fitz  #PyMuPDF

                doc=fitz.open(path)
                text_parts=[]

                for page_num in range(doc.page_count):
                    page=doc[page_num]
                    text=page.get_text()
                    if text.strip():text_parts.append(f"page {page_num+1}:\n{text}")
                
                doc.close()
                return "\n\n".join(text_parts)
            except ImportError:
                raise DocumentLoadError(
                    """"""
                )