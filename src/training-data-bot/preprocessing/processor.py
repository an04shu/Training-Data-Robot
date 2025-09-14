"""Text preprocessing and chunking."""

import re
from typing import List
from uuid import uuid4

from ..core.models import Document, TextChunk
from ..core.config import settings
from ..core.logging import get_logger


class TextPreprocessor:
    """Text preprocessing and chunking."""

    def __init__(self):
        self.logger = get_logger("preprocessor")
        self.chunk_size = settings.processing.chunk_size
        self.chunk_overlap = settings.processing.chunk_overlap

    async def process_document(self, document: Document) -> List[TextChunk]:
        """Process document into chunks."""
        # Clean text
        cleaned_text = self._clean_text(document.content)
        
        # Create chunks
        chunks = self._create_chunks(cleaned_text, document.id)
        
        self.logger.debug(f"Created {len(chunks)} chunks from document {document.id}")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        return '\n'.join(lines)

    def _create_chunks(self, text: str, document_id) -> List[TextChunk]:
        """Create text chunks with overlap."""
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Single chunk
            chunk = TextChunk(
                document_id=document_id,
                content=text,
                start_index=0,
                end_index=len(text),
                chunk_index=0,
                token_count=len(words)
            )
            chunks.append(chunk)
        else:
            # Multiple chunks
            chunk_index = 0
            start_word = 0
            
            while start_word < len(words):
                end_word = min(start_word + self.chunk_size, len(words))
                chunk_words = words[start_word:end_word]
                chunk_text = ' '.join(chunk_words)
                
                chunk = TextChunk(
                    document_id=document_id,
                    content=chunk_text,
                    start_index=start_word,
                    end_index=end_word,
                    chunk_index=chunk_index,
                    token_count=len(chunk_words)
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start_word = end_word - self.chunk_overlap
                chunk_index += 1
                
                if end_word >= len(words):
                    break
        
        return chunks 