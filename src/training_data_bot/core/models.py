"""
Core data models for Training Data Bot.

This module defines Pydantic models for all data structures used throughout
the application, ensuring type safety and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


class BaseEntity(BaseModel):
    """Base class for all entities with common fields."""
    
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


# Enums
class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    URL = "url"


class TaskType(str, Enum):
    """Available task types."""
    QA_GENERATION = "qa_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    NER = "named_entity_recognition"
    RED_TEAMING = "red_teaming"
    INSTRUCTION_RESPONSE = "instruction_response"


class QualityMetric(str, Enum):
    """Quality assessment metrics."""
    TOXICITY = "toxicity"
    BIAS = "bias"
    DIVERSITY = "diversity"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"


class ProcessingStatus(str, Enum):
    """Processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Export format options."""
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


# Document Models
class Document(BaseEntity):
    """Represents a source document."""
    
    title: str
    content: str
    source: str  # file path, URL, etc.
    doc_type: DocumentType
    language: Optional[str] = "en"
    encoding: Optional[str] = "utf-8"
    size: int = 0  # bytes
    word_count: int = 0
    char_count: int = 0
    
    # Processing info
    extraction_method: Optional[str] = None
    processing_time: Optional[float] = None
    
    @validator("word_count", pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if v == 0 and "content" in values:
            return len(values["content"].split())
        return v
    
    @validator("char_count", pre=True, always=True)
    def calculate_char_count(cls, v, values):
        if v == 0 and "content" in values:
            return len(values["content"])
        return v


class TextChunk(BaseEntity):
    """Represents a chunk of text from a document."""
    
    document_id: UUID
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = 0
    
    # Context preservation
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    
    # Semantic info
    embeddings: Optional[List[float]] = None
    topics: List[str] = Field(default_factory=list)
    
    @validator("token_count", pre=True, always=True)
    def estimate_token_count(cls, v, values):
        if v == 0 and "content" in values:
            # Rough estimation: 1 token ≈ 4 characters
            return len(values["content"]) // 4
        return v


# Task Models
class TaskTemplate(BaseEntity):
    """Represents a task template with prompt and configuration."""
    
    name: str
    task_type: TaskType
    description: str
    prompt_template: str
    
    # Task-specific configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality requirements
    min_output_length: int = 10
    max_output_length: int = 2000
    quality_thresholds: Dict[QualityMetric, float] = Field(default_factory=dict)
    
    # Performance settings
    timeout: int = 60
    max_retries: int = 3


class TaskResult(BaseEntity):
    """Result of a task execution."""
    
    task_id: UUID
    template_id: UUID
    input_chunk_id: UUID
    
    # Output
    output: str
    confidence: Optional[float] = None
    
    # Quality scores
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    
    # Processing info
    processing_time: float
    token_usage: int = 0
    cost: Optional[float] = None
    
    # Status
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None


# Training Data Models
class TrainingExample(BaseEntity):
    """A single training example."""
    
    input_text: str
    output_text: str
    task_type: TaskType
    
    # Source tracking
    source_document_id: UUID
    source_chunk_id: Optional[UUID] = None
    template_id: Optional[UUID] = None
    
    # Quality assessment
    quality_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    quality_approved: Optional[bool] = None
    
    # Additional fields for different formats
    instruction: Optional[str] = None  # For instruction-following datasets
    context: Optional[str] = None      # For context-based tasks
    category: Optional[str] = None     # For classification tasks


class Dataset(BaseEntity):
    """A collection of training examples."""
    
    name: str
    description: str
    version: str = "1.0.0"
    
    # Content
    examples: List[TrainingExample] = Field(default_factory=list)
    
    # Statistics
    total_examples: int = 0
    task_type_counts: Dict[TaskType, int] = Field(default_factory=dict)
    quality_stats: Dict[QualityMetric, Dict[str, float]] = Field(default_factory=dict)
    
    # Splits
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Export info
    export_format: ExportFormat = ExportFormat.JSONL
    exported_at: Optional[datetime] = None
    export_path: Optional[Path] = None
    
    @validator("total_examples", pre=True, always=True)
    def calculate_total_examples(cls, v, values):
        if "examples" in values:
            return len(values["examples"])
        return v


# API Models
class APIRequest(BaseModel):
    """Base API request model."""
    
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel):
    """Base API response model."""
    
    request_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class DecodoRequest(APIRequest):
    """Request to Decodo API."""
    
    prompt: str
    input_text: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    task_type: Optional[TaskType] = None


class DecodoResponse(APIResponse):
    """Response from Decodo API."""
    
    output: Optional[str] = None
    confidence: Optional[float] = None
    token_usage: int = 0
    cost: Optional[float] = None
    processing_time: Optional[float] = None


# Quality Assessment Models
class QualityReport(BaseEntity):
    """Quality assessment report for a dataset or example."""
    
    target_id: UUID  # ID of dataset or example being assessed
    target_type: str  # "dataset" or "example"
    
    # Overall quality score
    overall_score: float
    passed: bool
    
    # Individual metric scores
    metric_scores: Dict[QualityMetric, float] = Field(default_factory=dict)
    metric_details: Dict[QualityMetric, Dict[str, Any]] = Field(default_factory=dict)
    
    # Issues found
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Assessment metadata
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    assessor: str = "system"  # or user ID
    assessment_time: float = 0.0


# Processing Job Models
class ProcessingJob(BaseEntity):
    """Represents a long-running processing job."""
    
    name: str
    job_type: str  # "document_processing", "task_execution", "quality_assessment"
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Input/Output
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress tracking
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100


# Configuration Models
class ProjectConfig(BaseModel):
    """Project-level configuration."""
    
    name: str
    description: str
    version: str = "1.0.0"
    
    # Task configuration
    default_task_types: List[TaskType] = Field(default_factory=list)
    quality_requirements: Dict[QualityMetric, float] = Field(default_factory=dict)
    
    # Processing settings
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 300
    
    # Export settings
    default_export_format: ExportFormat = ExportFormat.JSONL
    output_directory: Path = Path("./outputs")
    
    # Data source settings
    supported_formats: List[DocumentType] = Field(default_factory=lambda: list(DocumentType))


# Utility Models
class FileInfo(BaseModel):
    """Information about a file."""
    
    path: Path
    name: str
    size: int
    modified_at: datetime
    file_type: DocumentType
    encoding: Optional[str] = None
    
    @validator("name", pre=True, always=True)
    def extract_name(cls, v, values):
        if not v and "path" in values:
            return values["path"].name
        return v


class ProgressInfo(BaseModel):
    """Progress information for operations."""
    
    current: int = 0
    total: int = 0
    message: str = ""
    percentage: float = 0.0
    eta: Optional[datetime] = None
    
    @validator("percentage", pre=True, always=True)
    def calculate_percentage(cls, v, values):
        if values.get("total", 0) > 0:
            return (values.get("current", 0) / values["total"]) * 100
        return 0.0 