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
    


