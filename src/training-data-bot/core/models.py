"""
Core data models for Training Data Bot.

This module defines Pydantic models for all data structures used throughout
the application, ensuring type safety and validation.
"""

from datetime import datetime , timezone                        #stores timestamps like created_at , updated_at
from enum import Enum                                  #instaed of plain strings ( eg PROCESSING) , Less risk of typos, easier refactoring
from pathlib import Path
from typing import Any, Dict, List,Optional, Union     #Dict,List,Optional,Union,Any
from uuid import UUID,uuid4                            #globally unique ids for objects(documents,jobs,datasets)

from pydantic import BaseModel, Field, validator , root_validator             #validate single field , validate the whole model at once
#Pydantic is a powerful data validation and settings management library for Python that leverages type hints to validate and serialize data schemas.
"""Pydantic is like a security guard + organizer for your data.

    It makes sure the data you create or pass into your system is valid.
    It automatically converts (serializes) things into the right type.
    It works using Python type hints."""


"""

It’s a base template for all the important data objects (Document, Dataset, Job, etc.).
Instead of writing id, created_at, metadata again in every model, you define them once here, then inherit.

"""

class BaseEntity(BaseModel):
    """Base class for all entities with common fields."""
    id: UUID =Field(default_factory=uuid4)                             #generates a random UUID for every object(looks like 550e8400-e29b-41d4-a716-446655440000)
    created_at:datetime=Field(default_factory=lambda: datetime.now(timezone.utc))
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
    NER="named entity recognition"               #find names and places?
    RED_TEAMING="red_teaming"                    #test for problems?
    INSTRUCTION_RESPONSE="instruction_response"  #follow instructions?


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
    JSONL="jsonl"                  #JSON lines
    CSV="csv"
    PARQUET="parquet"              #Apache Parquet is a data file format that supports efficient data storage and retrieval for complex data in bulk
    HUGGINGFACE="huggingface"
    OPENAI="openai"


class Document(BaseEntity):
    "Represents a source document."

    title:str
    content:str
    source:str                           #a file path or url(as string)
    doc_type: DocumentType
    language: Optional[str]="en"         #ISO-ish code for the language-english
    encoding: Optional[str]="utf-8"      #text encoding
    size:int=0                           #bytes
    word_count:int=0
    char_count:int=0
    scraped_metadata:Optional[Dict]=None       # Extra info from web scraping

    #Processing info
    extraction_method: Optional[str]=None
    processing_time: Optional[float]=None

    
    
    @validator("word_count",pre=True, always=True)                  # ??
    def calculate_word_count(cls,v,values):                         # cls-> The class itself (in this case, Document).Similar to self in instance methods, but here it’s the class because validators are class methods.
        if v==0 and "content" in values:                            # v-> The current value of the field being validated(word_count), default is defind above as 0 that means we have not passed it.
            return len(values["content"].split())                   #values->A dictionary of all other fields that Pydantic has already validated before this one. Key = field name, Value = parsed value.
        return v
    @validator("char_count", pre=True,always=True)
    def calculate_char_count(cls,v,values):                #document, char_count, dict
        if v==0 and "content" in values:                   #"content" in values checks if "content" is a key in that dict.
            return len(values["content"])
        return v
    
    
class TextChunk(BaseEntity):
    """Represents a chunk of text from a document."""

    document_id: UUID                    #Links the chunk back to its parent Document
    content:str
    start_index: int
    end_index: int
    chunk_index: int
    token_count: int = 0                  #?
    source_url:Optional[str]=None         # Original URL if from web

    #Context preservation                     #?
    preceding_context: Optional[str]=None
    following_context: Optional[str]=None

    #Semantic info
    embeddings: Optional[List[float]]=None               #?
    topics: List[str]=Field(default_factory=list)        #Tags or topics detected in this chunk.

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
    prompt_template:str

    #Task_specific configuration
    parameters: Dict[str,any] = Field(default_factory=dict)
    quality_filters:List[QualityMetric]=Field(default_factory=list) 

    #quality requirements
    min_output_length: int=10
    max_output_length: int=2000
    quality_thresholds: Dict[QualityMetric , float]=Field(default_factory=dict)

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
    confidence : Optional[float]=None

    #Quality scores
    quality_scores: Dict[QualityMetric,float]=Field(default_factory=dict)

    #Processing info
    processing_time: float
    token_usage: int=0
    cost:Optional[float]=None

    #Status
    status : ProcessingStatus=ProcessingStatus.PENDING
    error_message: Optional[str]=None

    ai_provider:str        #which AI brain did the work


#Training Data Models
class TrainingExample(BaseEntity):
    """A single training example."""

    input_text:str
    output_text:str
    task_type:TaskType

    #Source tracking
    source_document_id: UUID                     #link back to the parent Document

    source_url:Optional[str]=None              #Original web source

    source_chunk_id: Optional[UUID]=None       #which chunk it came from
    template_id: Optional[UUID]=None           #TaskTemplate was used

    #Quality assessment
    quality_scores: Dict[QualityMetric,float]=Field(default_factory=dict)            #Example: {QualityMetric.RELEVANCE: 0.9, QualityMetric.COHERENCE: 0.85}.
    quality_approved: Optional[bool]=None

    #Additional fields for different formats
    instruction: Optional[str]=None             #for instruction-following datasets??
    context: Optional[str]=None                 #for context-based tasks
    catergory: Optional[str]= None              #for classification tasks

    difficulty_level:str="medium"       # Easy, Medium, Hard
    tokens_used:int=0                   #how many AI tokens used
    generation_cost: float=0.0          #how much it cost


class Dataset(BaseEntity):
    """A collection of training examples."""

    name:str
    description:str
    version: str="1.0.0"

    #context
    examples: List[TrainingExample]=Field(default_factory=list)

    task_types: List[TaskType]=Field(default_factory=list)            #what kinds of tasks
    source_urls: List[str]=Field(default_factory=list)                #all websites we scraped
    total_cost:float=0.0 
    export_formats:List[str]=Field(default_factory=list)             # Available export formats
    
    #Statistics
    total_examples:int =0
    task_type_counts: Dict[TaskType,int]=Field(default_factory=dict)
    quality_stats: Dict[QualityMetric,Dict[str,float]]=Field(default_factory=dict)

    #Splits
    train_split: float=0.8
    validation_split: float=0.1
    test_split: float=0.1

    #Export info
    export_format: ExportFormat=ExportFormat.JSONL
    exported_at: Optional[datetime]=None
    exported_path: Optional[Path]=None

    @validator("total_examples",pre=True, always=True)
    def calculate_total_examples(cls,v,values):
        if "examples" in values:
            return len(values["examples"])
        return v


#API models
class APIRequest(BaseEntity):
    """Base API request model."""   
    request_id: UUID = Field(default_factory=uuid4)
    timestamp : datetime = Field(default_factory=datetime.now(timezone.utc))

class APIResponse(BaseEntity):
    """Base API response_model."""

    request_id: UUID
    timestamp: datetime=Field(default_factory=datetime.now(timezone.utc))
    success: bool
    message :Optional[str]=None
    data: Optional[Any]=None
    error : Optional[Dict[str,Any]]=None

class DecodoRequest(APIRequest):
    """Request to Decodo API."""
    prompt: str
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
    overall_score:float=0.0
    passed:bool

    #Individual metric scores
    metric_scores: Dict[QualityMetric, float]=Field(default_factory=dict)
    metric_details: Dict[QualityMetric, Dict[str,Any]]=Field(default_factory=dict)

    #Issues found
    issues: List[str]=Field(default_factory=list)
    warnings: List[str]=Field(default_factory=list)

    #Assessment metadata
    assessed_at: datetime=Field(default_factory=datetime.now(timezone.utc))
    assessor: str="system"  # or user id (Who did the assessment. "system" if automatic, or a user ID if human.)
    assessment_time: float=0.0


class ProcessingJob(BaseEntity):
    """Represents a long-runnning processing job."""

    name:str
    job_type:str
    status:ProcessingStatus=ProcessingStatus.PENDING

    #Input/Output
    input_data: Dict[str,Any]=Field(default_factory=dict)
    output_data: Dict[str,Any]=Field(default_factory=dict)

    #Progress tracking
    total_items: int=0
    processed_items: int=0
    failed_items: int=0

    #Timing
    started_at: Optional[datetime]=None
    completed_at:Optional[datetime]=None
    estimated_completion: Optional[datetime]=None

    #error handling
    error_message: Optional[str]=None
    retry_count: int=0
    max_retries: int=3

    @property
    def progress_percentage(self)->float:
        """calculate progress percentage."""
        if self.total_items==0:
            return 0.0
        return (self.processed_items / self.total_items)*100
    
class ProjectConfig(BaseModel):
    """Project-level configuration."""

    name:str
    description:str
    version:str="1.0.0"

    #Task configuration
    default_task_types: List[TaskType]=Field(default_factory=list)
    quality_requirements: Dict[QualityMetric, float]=Field(default_factory=dict)

    #Processing settings
    batch_size: int=10
    max_workers: int=4
    timeout: int=300

    #Export settings
    default_export_format:ExportFormat=ExportFormat.JSONL
    output_directory: Path=Path("./outputs")

    #Data source settings
    supported_formats: List[DocumentType]=Field(default_factory=list)


class FileInfo(BaseModel):
    """Information about a file."""

    path:Path
    name:str
    size:int
    modified_at: datetime
    file_type: DocumentType
    encoding : Optional[str]=None                   #?

    @validator("name", pre=True , always=True)
    def extract_name(cls, v , values):
        if not v and "path" in values:
            return values["path"].name                 #If name isn't given (not v) but path is present, it auto-fills name from the path’s filename
        return v
    

class ProgressInfo(BaseModel):
    """progress info for operations."""

    current: int=0
    total: int=0
    message: str
    percentage: float=0.0
    eta: Optional[datetime]=None                    #Estimated Time of Arrival

    @validator("percentage", pre=True, always=True)
    def calculate_percentage(cls, v, values):
        if values.get("total",0)>0:                               #checks if "total" exists in the dictionary.If "total" exists → returns its value.If not → returns 0 as a default
            return (values.get("current",0)/values["total"])
        return 0.0





