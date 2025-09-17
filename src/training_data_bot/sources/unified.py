"""
Unified document loader.

This module provides a unified interface that automatically selects
the appropriate loader based on source type and format.
"""

import asyncio
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

from .base import BaseLoader
from .documents import DocumentLoader
from .pdf import PDFLoader
from .web import WebLoader
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError, UnsupportedFormatError
from ..core.logging import get_logger, LogContext


class UnifiedLoader(BaseLoader):
    """
    Unified loader that automatically selects appropriate sub-loaders.
    
    This loader acts as a dispatcher, routing different source types
    to specialized loaders while providing a consistent interface.
    """

    def __init__(self, decodo_client=None):
        super().__init__()
        self.logger = get_logger("loader.UnifiedLoader")
        
        # Initialize sub-loaders (share DecodoClient with WebLoader if provided)
        self.document_loader = DocumentLoader()
        self.pdf_loader = PDFLoader()
        
        if decodo_client:
            # Use shared DecodoClient instance for better resource management
            self.web_loader = WebLoader(use_decodo=True)
            self.web_loader.decodo_client = decodo_client
            self.web_loader.use_decodo = True
            self.logger.info("🌐 UnifiedLoader using shared Decodo client")
        else:
            self.web_loader = WebLoader()
        
        # All supported formats from sub-loaders
        self.supported_formats = list(DocumentType)

    async def load_single(
        self,
        source: Union[str, Path],
        **kwargs
    ) -> Document:
        """
        Load a single document using the appropriate loader.
        
        Args:
            source: Source path, URL, or identifier
            **kwargs: Additional loading options
            
        Returns:
            Loaded document
        """
        with LogContext("unified_load_single", source=str(source)):
            try:
                # Determine source type and select appropriate loader
                loader = self._select_loader(source)
                
                if loader is None:
                    raise UnsupportedFormatError(
                        file_format=str(source),
                        supported_formats=[fmt.value for fmt in self.supported_formats]
                    )
                
                # Load using selected loader
                document = await loader.load_single(source, **kwargs)
                self.logger.debug(f"Successfully loaded {source} using {loader.__class__.__name__}")
                
                return document
                
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to load document from {source}",
                    file_path=str(source),
                    cause=e
                )

    async def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            file_patterns: Optional file patterns to match
            **kwargs: Additional loading options
            
        Returns:
            List of loaded documents
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise DocumentLoadError(f"Directory not found: {directory}")
        
        # Find all supported files
        sources = self._find_supported_files(
            directory,
            recursive=recursive,
            patterns=file_patterns
        )
        
        if not sources:
            self.logger.warning(f"No supported files found in {directory}")
            return []
        
        self.logger.info(f"Found {len(sources)} supported files in {directory}")
        
        # Load all files
        return await self.load_multiple(sources, **kwargs)

    def _select_loader(self, source: Union[str, Path]) -> Optional[BaseLoader]:
        """
        Select the appropriate loader for the given source.
        
        Args:
            source: Source to load
            
        Returns:
            Appropriate loader or None if unsupported
        """
        try:
            # Handle URLs
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                return self.web_loader
            
            # Handle file paths
            source = Path(source) if isinstance(source, str) else source
            
            if not source.exists():
                return None
            
            # Get file extension
            suffix = source.suffix.lower().lstrip('.')
            
            try:
                doc_type = DocumentType(suffix)
            except ValueError:
                return None
            
            # Route to appropriate loader
            if doc_type == DocumentType.PDF:
                return self.pdf_loader
            elif doc_type in [DocumentType.TXT, DocumentType.MD, DocumentType.HTML, 
                             DocumentType.JSON, DocumentType.CSV, DocumentType.DOCX]:
                return self.document_loader
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error selecting loader for {source}: {e}")
            return None

    def _find_supported_files(
        self,
        directory: Path,
        recursive: bool = True,
        patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Find all supported files in directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            patterns: Optional glob patterns to match
            
        Returns:
            List of file paths
        """
        files = []
        
        # Default patterns for supported formats
        if patterns is None:
            patterns = [
                "*.pdf", "*.txt", "*.md", "*.html", "*.htm",
                "*.docx", "*.json", "*.csv"
            ]
        
        try:
            for pattern in patterns:
                if recursive:
                    files.extend(directory.rglob(pattern))
                else:
                    files.extend(directory.glob(pattern))
        except Exception as e:
            self.logger.error(f"Error finding files in {directory}: {e}")
        
        # Filter to only existing files and remove duplicates
        unique_files = list(set(f for f in files if f.is_file()))
        
        return sorted(unique_files)

    async def close(self):
        """Clean up resources from all sub-loaders."""
        try:
            # Close web loader (which has Decodo client)
            if hasattr(self.web_loader, 'close'):
                await self.web_loader.close()
                self.logger.debug("🔧 WebLoader closed")
            
            # Other loaders may not need cleanup, but check anyway
            for loader_name in ['document_loader', 'pdf_loader']:
                loader = getattr(self, loader_name, None)
                if loader and hasattr(loader, 'close'):
                    await loader.close()
                    self.logger.debug(f"🔧 {loader_name} closed")
            
            self.logger.debug("🔧 UnifiedLoader cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during UnifiedLoader cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def get_source_info(self, source: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a source without loading it.
        
        Args:
            source: Source to analyze
            
        Returns:
            Source information dictionary
        """
        info = {
            "source": str(source),
            "supported": False,
            "loader": None,
            "estimated_size": 0,
            "doc_type": None,
            "error": None,
        }
        
        try:
            # Check if source is supported
            loader = self._select_loader(source)
            if loader:
                info["supported"] = True
                info["loader"] = loader.__class__.__name__
                info["doc_type"] = self.get_document_type(source).value
                
                # Get size estimate
                if isinstance(source, (str, Path)):
                    path = Path(source)
                    if path.exists() and path.is_file():
                        info["estimated_size"] = path.stat().st_size
                        
        except Exception as e:
            info["error"] = str(e)
        
        return info

    async def batch_info(self, sources: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Get batch information about multiple sources.
        
        Args:
            sources: List of sources to analyze
            
        Returns:
            Batch information summary
        """
        info_tasks = [self.get_source_info(source) for source in sources]
        source_infos = await asyncio.gather(*info_tasks, return_exceptions=True)
        
        # Aggregate statistics
        stats = {
            "total_sources": len(sources),
            "supported_sources": 0,
            "unsupported_sources": 0,
            "total_size": 0,
            "by_type": {},
            "by_loader": {},
            "errors": 0,
        }
        
        for info in source_infos:
            if isinstance(info, dict):
                if info["supported"]:
                    stats["supported_sources"] += 1
                    stats["total_size"] += info["estimated_size"]
                    
                    # Count by type
                    doc_type = info["doc_type"]
                    if doc_type:
                        stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
                    
                    # Count by loader
                    loader = info["loader"]
                    if loader:
                        stats["by_loader"][loader] = stats["by_loader"].get(loader, 0) + 1
                else:
                    stats["unsupported_sources"] += 1
                
                if info["error"]:
                    stats["errors"] += 1
            else:
                stats["errors"] += 1
        
        return stats 