"""
Base loader class for document sources.

This module provides the abstract base class that all document loaders
inherit from , ensuring consistent interface and behavior.
"""

import asyncio
from abc import ABE , abstractmethod
from pathlib import Path
from typing import List, Optional, Union , AsyncGenerator , Dict , Any

from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError , UnsupportedFormatError
from ..core.logging import get_logger, LogContent

class BaseLoader(ABC):
    """abstract base calss for all document loaders.

    provides common functionality and interface that all loaders must implement.
    """
    def __init__(self):
        self.logger=get_logger(f"loader.{self.__class__.__name__}")
        self.supported_formats: List[DocumentType]=[]
    @abstractmethod
    async def load_single(
        self,
        source:Union[str,Path],
        **kwargs
    )-> Document:
        """
        Load a single socument from source.

        Args:
            source: Source path, URL, or identifier
            **kwargs: Additional loading options

        return:
            Loaded document
        
        Raises
        """   
        pass
    async def load_multiple(
        self,
        sources: List[Union[str,Path]],
        max_workers: int=4,
        **kwargs
    ) ->List[Document]:
        """
        Load multiple documents concurrently.

        Args:
            sources:List of sources to load
            max_workers: Maximum concurrent workers
            **kwargs:additional loading options

        Returns:
            List of loaded documents
        """
        with LogContext("load_multiple",source_sount=l):
            semaphore=asyncio.Semaphore(max_workers)

            async def load_with_semaphore(source):
                async with semaphore:
                    try:
                        return wait self.load_single()
                    except Exception as e:
                        self.logger.error(f"Failed to load")
                        return None
                    
            #load all sources concurrently
            tasks=[load_with_semaphore(source) for sources]
            results=await asyncio.gather(*tasks, return)

            #filter out failed loads and exceptions
            documents=[]
            for i , result in enumerate(results):
                if isinstance(result, Document):
                    documents.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error loading")
                #None results (failed loads) are already

            self.logger.info(f"Successfully loaded")
            return documents
        
    async def load_stream(
        self,
        sources:List[Union[str,Path]],
        **kwargs
    )->AsyncGenerator[Document,None]:
        """
        Load documents as a stream (generator).
        Args:
            sources:List of sources to load
            **kwargs:Additional loading options

        Yields
            Documents as they ate loaded
        """
        for source in sources:
            try:
                document=await self.load_single(sources, **kwargs)
                yield document
            except Exception as e:
                self.logger.error(f"Failed to load {source}:")
                continue
        
    def supports_fomrat(self,doc_type:DocumentType)->bool:
        """check if this loader supports the given format."""
        return doc_type in self.supported_formats
    
    def validates_sources(self,sources:Union[str,Path])->bool:
        """"Validate if the soource can be loaded by this loader.

        Args:
            Source: Source to validate

        Returns:
            True if source can be loaded
        """
        try:
            if isinstance(source, str):
                if source.startswith(('https://','http://')):
                    #URL-check if web loader
                    return DocumentType.URL in self.supported_formats
                else:
                    source=Path(source)

            if isinstance(source,Path):
                if not source.exists():
                    return False
                
                #check file extension
                suffix=source.suffix.lower().lstrip('.')
                try:
                    doc_type=DocumentType(suffix)
                    return self.supports_fomrat(doc_type)
                except ValueError:
                    return False
            return True
    
        except Exception:
            return False
        
    def get_document_type(self,source,Union[str,Path])-> DocumentType:
        """
        determine document type from source

        Args:
            source: Source path or URL

        Returns:
            Detected document type
        
        Raises:
            UnsupportedFormatError: If format cannot be determined
        
        """
        if isinstance(source, str):
            if source.startswith(('https://','http://')):
                return DocumentType.URL
            else:
                source=Path(source)

        if isinstance(source,Path):
            suffix=source.suffix.lower().lstrip('.')
            try:
                return DocumentType(suffix)
            except ValueError:
                raise UnsupportedFormatError(
                    file_format=suffix,
                    supported_formats=[fmt.value for fmt in self.supported_formats]
                )
            
        raise UnsupportedFormatError(
            file_formats="unknown",
            supported_formats=[fmt.value for fmt in self.supported_formats]
        )
        
    def extract_metadata(self,source: Union[str,Path]) ->Dict[str,Any]:
        """
        Extract metadata from source.
        Args:
            source:Source path or URl
        
        Returns:
            Metadata dictionary
        """
        metadata={}

        if isinstance(source,str):
            metadata["source"]=source
            if source.startswith(('http://','https://')):
                metadata["source_type"]="url"
            else:
                metadata["source_type"]="file"
                source=Path(source)

        if isinstance(source,Path):
            metadata["source"]=str(source.absolute())
            metadata["source_type"]="file"
            metadata["filename"]=source.name
            metadata["extention"]=source.suffix

            if source.exists():
                stat=source.start()
                metadata["size"]=stat.st.size
                metadata["modified_time"]=stat.st_mtime

        return metadata
    
    def create_document(
        self,
        title:str,
        content:str,
        source:Union[str,Path],
        doc_type:DocumentType,
        **kwargs
    )->Document:
        """
        Create a document instance with standard metadata.

        Args:
            title:Document title
            content:Document content
            source:Source path or URL
            doc_type:Document type
            **kwargs:Addional document properties

        Return:
            Document instance
        """
        metadata=self.extract_metadata(source)
        metadata.update(kwargs.get("metadata",{}))

        return Document(
            title=title
            content=content
            source=str(source),
            doc_type=doc_type,
            size=len(content.encode('utf-8')),
            metadata=metadata
            **{k:v for k, v in kwargs.itmes() if k!="metadata"}
        )
