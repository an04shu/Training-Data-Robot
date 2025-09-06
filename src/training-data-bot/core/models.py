"""
Core data models for Training Data Bot.

This module defines Pydantic models for all data structures used throughout
the application, ensuring type safety and validation.
"""

from datetime import datetime                          #stores timestamps like created_at , updated_at
from enum import Enum                                  #instaed of plain strings ( eg PROCESSING)
from pathlib import Path
from typing import Any, Dict, List,Optional, Union     #Dict,List<Optional,Union,Any
from uuid import UUID,uuid4                            #globally unique ids for objects(documents,jobs,datasets)

from pydantic import BaseModel, Field, validator , root_validator
#base class for model , ? , validate single field , validate the whole model at once



"""

It’s a base template for all the important data objects (Document, Dataset, Job, etc.).
Instead of writing id, created_at, metadata again in every model, you define them once here, then inherit.

"""
class BaseEntity(BaseModel):
    """Base class for all entities with common fields."""
    id: UUID =Field(default_factory=uuid4)                             #generates a random UUID for every object(looks like 550e8400-e29b-41d4-a716-446655440000)
    created_at:datetime=Field(default_factory=datetime.utcnow)
    updated_at:Optional[datetime]=None                                 #Starts empty (None) until the first update happend.
    metadata: Dict[str,any]= Field(default_factory=dict)               #Starts empty (None)

    class Config:
        use_enum_values=True                                           #If you use an Enum field, Pydantic will store the actual value (like "pdf") instead of the Enum object (DocumentType.PDF).This makes JSON exports cleaner.
        allow_population_by_field_name=True                            #Lets you create objects using either the field’s Python name or alias
        arbitary_types_allowed=True                                    #Allows fields with types Pydantic doesn’t know natively(like Path)

#enums
class DocumentType(str,Enum):
    """Supported document types."""
    PDF="pdf"
    DOCX="docx"
    TXT="txt"
    MD="md"              
    HTML="html"
    JSON="json"          #like data tables
    csv="csv"            
    URL="url"


class TaskType(str,Enum):
    """Available task types."""                  
    QA_GENERATION="qageneration"                 #generate q&a
    CLASSIFICATION ="classification"             #sort things into groups 
    SUMMARIZATION="summarization"                #summaries
    NER="named entity recognition"               #find names and places
    RED_TEAMING="red_teaming"                    #test for problems
    INSTRUCTION_RESPONSE="instruction_response"  #follow instructions

class QualityMetric(str,Enum):
    """quality assessment metrics."""
    TOXICITY="toxicity"                     #is it mean or harful?
    BIAS="bias"                             #is it fare to everyone
    DIVERSITY="diversity"                   #is it varied and intresting
    COHERENCE="coherence"                   #does it make sense
    RELEVANCE="relevance"                   #is it related to the topics

class ProcessingStatus(str,Enum):
    """Processing status values."""
    PENDING="pending"
    PROCESSING="processing"
    COMPLETED="completed"
    FAILED="failed"
    CANCELLED="cancelled"

class ExportFormat(str,Enum):
    """Export format options."""
    JSONL="jsonl"
    CSV="csv"
    PARQUET="parquet"
    HUGGINGFACE="huggingface"
    OPENAI="openai"

class Document(BaseEntity):
    "Represents a source document."

    title:str
    content:str
    source:str
    doc_type: DocumentType
    language: Optional[str]="en"
    encoding: Optional[str]="utf-8"
    size:int=0 #bytes
    word_count:int=0
    char_count:int=0

    #Processing info
    extraction_method: Optional[str]=None
    processing_time: Optional[float]=None
    @validator("word_count",pre=True, always=True)
    def calculate_word_count(cls,v,values):
        if v==0 and "content" in values:
            return len(values["content"].split())
        return v
    @validator("char_count", pre=True,always=True)
    def calculate_char_count(cls,v,values):
        if v==0 and "content" in values:
            return len(values["content"])
        return v
    
class TextChunk(BaseEntity):
    """Represents a chunk of text from a document."""
    document_id: UUID
    content=str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = 0

    #Context preservation
    preceding_context: Optional[str]=None
    following_context: Optional[str]=None

    #Semantic info
    embeddings: Optional[List[float]]=None
    topics: List[str]=Field(default_factory=list)

    @validator("token_count", pre=True, always=True)
    def estimate_token_count(cls, v, values):
        if v==0 and "content" in values:
            #Rough estimation : 1 token ~4 characters
            return len(values["content"]) //4
        return v
class TaskTemplate(BaseEntity):
    """Represents a task template with prompt and configuration."""

    name:str
    task_type: TaskType
    description:str
    prompt_template=str

    #Task_specific configuration
    parameters: Dict[str,any] =Field(default_factory=dict)

    #quality requirements
    min_output_length: int=10
    max_output_length: int=2000
    quality_thresholds: Dict[QualityMetric , float]=Field(default_factor=dict)

    #Performance settings
    timeout: int=60
    max_retries: int=3

class TaskResult(BaseEntity):
    """Result of a task execution."""

    task_id:UUID
    template_id:UUID
    input_chunk_id:UUID

    #Output
    output: str
    confidence :Optional[float]=None

    #Quality scores
    quality_scores: Dict[QualityMetric,float]=Field(default_factory=dict)

    #Processing info
    processing_time: float
    token_usage: int=0
    cost:Optional[float]=None

    #Status
    statuc : ProcessingStatus=ProcessingStatus.PENDING
    error_message: Optional[str]=None

#Training Data Models
class TrainingExampl(BaseEntity):
    """A single training example."""

    input_text:str
    out_text:str
    task_type:TaskType

    #Source tracking
    source_document_id: UUID 
    source_chunk_id: Optional[UUID]
    template_id: Optional[UUID]

    #Quality assessment
    quality_scores: Dict[QualityMetric,float]=Field(default_factory=dict)
    quality_approved: Optional[bool]=None
    #Additional fields for different formats
    instruction: Optional[str]=None
    context: Optional[str]=None
    catergory: Optional[str]= None

class Dataset(BaseEntity):
    """A collection of training examples."""

    name=str
    description:str
    version: str="1.0.0"

    #context
    examples: List[TrainingExample]=Field(default_factory=list)

    #Statistics
    total_examples:int =0
    task_type_counts: Dict[TaskType,int]=Field(default_factory=dict)
    quality_stats: Dict[QualityMetric,Dict[str,float]]=Field(default_factory)

    #Splits
    train_split: float=0.8
    validation_split: float=0.1
    test_split: float=0.1

    #Export info
    export_format: ExportFormat=ExportFormat.JSONL
    exported_at: Optional[datetime]=None
    exporrted_path: Optional[Path]=None

    @validator("total examples",pre=True, always=True)
    def calculate_total_examples(cls,v,values):
        if "examples" in values:
            return len(values["examples"])
        return v

#API models
class APIRequest(BaseEntity):
    """Base API request model."""   
    request_id: UUID =Field(default_factory=uuid4)
    timestamp:datetime=Field(default_factory=datetime.utc)

class APIResponse(BaseEntity):
    """Base API response_model."""

    request_id: UUID
    timestamp: datetime=Field(default_factory=datetime.utc)
    success: bool
    message :Optional[str]=None
    data: Optional[Any]=None
    error : Optional[Dict[str,Any]]=None

class DecodoRequest(APIRequest):
    """Request to Decodo API."""
    promt: str
    input_text: str
    parameters: Dict[str,Any]=Field(default_factory=dict)
    task_type: Optional[TaskType]=None

class DecodoResponse(APIResponse):
    """Response to Decodo API.""" 
    output: Optional[str]=None
    confidence: Optional[float]=None
    token_usage : int=0
    cost:Optional[float]=None
    processing_time: Optional[float]=None

#Quality Assessment Models
class QualityReport(BaseEntity):
    """Quality assessment report for a dataset or example."""

    target_id:UUID  #ID of dataset or example being assessed
    target_type:str   #"dataset" or "example"

    #Overall quality score
    overall_score:float
    passed:bool

    #Individual metric scores
    metric_scores: Dict[QualityMetric, float]




