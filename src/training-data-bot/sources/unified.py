"""
This module provides a unified interface that automatically chooses
the appropriate loader based on source type and format.
"""
import asyncio
from pathlib import Path
from typing import List , Union , Optional , Dict , Any

from .base import BaseLoader
from .documents import DocumentLoader
from .pdf import PDFLoader
from.web import WebLoader
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError , UnsupportedFormatError
from ..core.logging import get_logger, LogContent

class UnifiedLoader(BaseLoader):
    """
    Unified loader that automatically selects appropriate sub-loaders'

    this loader acts as a dispatcher , routing different source types
    to specialized loaders while providing a consistant interface.
    """

    def __init__(self,decodo_client=None):
        super().__init__()
        self.logger=get_logger("loader.UnifiedLoader")

        #initialise sub-loaders(share DecodoClient with WebLoader if provided)
        self.decument_loader=DocumentLoader()
        self.pdf_loader=PDFLoader()
        self.web_loader()

        if decodo_client:
            #Use shared DecodoClient instance for better resource management
            self.web_loader=WebLoader(use_decodo=True)
            self.web_loader.decodo_client=decodo_client
            self.web_loader.use_decodo=True
            self.logger.info("UnifiedLoader using share Decodo client")
        else:
            self.web_loader=WebLoader()

        #All supported formats from sub-loaders
        self.supported_formats=list(DocumentType)

    async def load_single(
            slef,
            source:Union[str,Path],
            **kwargs
    )-> Document:
        """
        Load a single document using the appropriate loader

        Args:
            source: Source path URL or identifier
            **kwargs: Additional loading options

        Returns:
            Loaded document
        """
        with LogContext("unified_load_single",source=str()):
            try:
                #Determine source type and select appropriate loader
                loader=self._select_loader(source)

                if loader is None:
                    raise UnsupportedFormatError(
                        file_format=str(source),
                        supported_formats=[fmt.value for fmt in self.supported_formats]
                    )
                
                #Load using selected loader
                document=await loader.load_single(source, **kwargs)
                self.logger.debug(f"Sucessfully loaded {source} using {loader.__c}")

                return document
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load document from {source}",
                    file_path=str(source),
                    cause=e;
                )
            
    async def load_directory(
        self,
        directory: Union[str,Path],
        recursive: bool=True,
        file_patterns: Optional[List[str]]=None,
        **kwargs
    )-> List[Document]:
        """
        Load all supported documents froma directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            file_patterns: Optional file patterns to match
            **kwargs:Additional loading options

        Returns:
            List of loaded documents
        """
        directory=Path(directory)

        if not directory.exists() or not directory.is_dir():
            raise DocumentLoadError(f"Directory not found:{directory}")
        
        #Find all supported files
        sources=self._find_supported_files(
            directory,
            recursive=recursive,
            patterns=file_patterns
        )

        if not sources:
            self.logger.warning(f"No supported files found in Directory")
            return []
        
        self.logger.info(f"Found {len(sources)} supported files")

        #Load all files
        return await self.load_multiple(sources, **kwargs)
    
def _select_loader(self, sources:Union [str,Path])-> Optional[BaseLoader]:
    """
    Select the appropriate loader for the given source.

    Args:
        source: Source to load

    Returns:
        Appropriate loader or None if unsupported
    """
    try:
        #handle URLs
        if isinstance(source,str) and source.startswith(('http://','https://')):
            return self.web_loader
    
        #Handle file paths
        source=Path(source) if isinstance(source,str) else source

        if not source.exists():
            return None
        
        #Get file extension
        suffix=source.suffix.lower().lstrip('.')

        try:
            doc_type=DocumentType(suffix)
        except ValueError:
            return None
        
        #Route to appropriate loader
        if doc_type == DocumentType.PDF:
            return self.pdf_loader
        elif doc_type in [DocumentType.TXT, DocumentType.MD, DocumentType.HTML, Document.JSON, DocumentType.CSV, DocumentType.DOCX]:
            return self.document_loader
        else:
            return None
    except Exception as e:
        self.logger.error(f"Error selecting loader for {source}: {e}")
        return None
    

def _find_supported_files(
        self,
        directory: Path,
        recursive: bool =True,
        patterns: Optional[List[str]]=None
)-> List[Path]:
    """
    Find all supported files in a directory.

    Args:
        directory:directory to search
        recorsive: whether to search recursively
        patterns: Optional glob patterns to match
    
        Returns:
        List of file paths

    """
    files=[]

    #Default patterns for  supported formats
    if patterns is None:
        patterns=[
            "*.pdf","*.txt","*.md","*.html","*.htm","*.docx","*.json","*.csv"
        ]

    try:
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.rglob(pattern))
    
    except Exception as e:
        self.logger.error(f"Error finding files in {directory}:{e}")

    #Filter to only existing files and remove diplicates
    uniqie_files=list(set(f for f in files if f.is_file()))

    return sorted(uniqie_files)

async def close(self):
    """Clenan up resources from all sub-leaders."""
    try:
        #close web loader(which has Decodo client)
        if hasattr(self.web_loader , 'close'):
            await self.web_loader.close()
            self.logger.debug("WebLoader closed")

        #Other loaders may not need cleanup , but check anyway
        for loader_name in ['document_loader','pdf_loader']:
            loader=getattr(self,loader_name,None)
            if loader and hasattr(loader, 'close'):
                await loader.close()
                self.logger.debug(f"{loader_name} closed")

        self.logger.debug("UnifiedLoader cleanup completed")

    except Exception as e:
        self.logger.error(f"Error during UnifiedLoader cleanup: {e}")  

async def __aenter__(self):
    """Async context manager entry."""
    return self

async def __exit__(self,exc_type,exc_val,exc_tb):
    """Async context manager exit."""
    await self.close()

async def get_source_info(self, source:Union[str,Path])-> Dict[str]:
    """
    Get information about a source without loading it.

    Args:
        source :source to analyse

    Returns:
        Source information dictionary
    """
    info={
        "source":str(source),
        "supported":False,
        "loader":None,
        "estimated_size":0,
        "doc_type":None,
        "error":None,
    }