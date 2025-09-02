"""
Main Training Data Bot class.

This module contains the core TrainingDataBot class that orchestrates all functionality including document loading, processing, quality assessment, and dataset export.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional,Union, Any
from uuid import UUID

from .core.config import settings
from .core.logging import get_logger, LogContext
from .core.exceptions import TrainingDataBotError, ConfigurationError
from .core.models import(
    Document,
    Dataset,
    TrainingExample,
    TaskType,
    DocumentType,
    ProcessingJob,
    ProcessingStatus,
    QualityReport,
    ExportFormat,
)

#import modules
from .sources import UnifiedLoader
from .decodo import DecodoClient
from .ai import AIClient
from .tasks import TaskManager
from .preprocessing import TextPreprocessor
from .evaluation import QualityEvaluator
from .storage import DatasetExporter, DatabaseManager

class TrainingDataBot:
    """
    Main Training Data Bot class.
    this class provides a high-level interface for:
    -loading documents from various sources
    -Processing text with task templates
    -Quality assessment and filtering
    -Dataset creation and export
    """
    def __init__(self,config:Optional[Dict[str,Any]]=None):
        """
        Initialize the Training Data Bot.
        Args:
            config : Optional configuration overrides
        """

        self.logger=get_logger("training_data_bot")
        self.config=config or {}
        self._init_components()
        self.logger.info("Training Data Bot initialized successfully")

    def _init_components(self):
        """Initialize all bot components."""
        try:
            self.loader=UnifiedLoader()
            self.decodo_client=DecodoClient()
            self.ai_client=AIClient()
            self.task_manager=TaskManager()
            self.preprocessor=TextPreprocessor()
            self.evaluator=QualityEvaluator()
            self.exporter=DatasetExporter()
            self.db_manager=DatabaseManager()

            #state(memory boxes)
            self.documents:Dict[UUID,Document]={}
            self.datasets:Dict[UUID,Dataset]={}
            self.jobs:Dict[UUID,ProcessingJob]={}

        except Exception as e:
                raise ConfigurationError("Failed to initialize bot components", 
                context={"error":str(e)},
                cause = e
            )
        
    async def load_document(
        self,
        sources:Union[str,Path,List[Union[str,Path]]],
        doc_types:Optional[List[DocumentType]]=None,
        **kwargs
    ) ->List[Document]:
        """
        Load documents from various sources.

        Args:
            sources:Single source or list of sources (files , URLs , directories)
            doc_types :  Optional filter for document types
            **kwargs : additional loading options

        Returns:
            List of loaded documents
        """
        with LogContext("document_loading",sources=str(sources)):
            try:
                #Ensure sources is a list
                if isinstance(sources,(str,Path)):
                    sources=[sources]

                #check if any source is a directory
                documents=[]
                for source in sources:
                    source_path=Path(source)
                    if source_path.is_dir():
                        dir_docs=await self.loader.load.load_directory(source_path)
                        documents.extend(dir_docs)
                    else:
                        doc=await self.loader.load_single(source)
                        documents.append(doc)
                for doc in documents:
                    self.documents[doc.id]=doc

                self.logger.info(f"Loaded {len(documents)} documents")
                return documents
            
            except Exception as e:
                self.logger.error(f"Failed to load documents:{e}")
                raise

    async def process_documents(
            self,
            documents:Optional[List[Document]]=None,
            task_types:Optional[List[TaskType]]=None,
            quality_filter:bool=True,
            **kwargs
    ) -> Dataset:
        """
        Process documents to create training data.

        Args:
            documents : Documents to process(default: all loaded documents)
            task_types : Task types to execute (default: from config)
            quality_filter : Whether to apply quality filtering
            ** kwargs: additional processing options

        returns:
            Generated dataset
        """
        with LogContext("document_processing"):
            try:
                #Use all documents if none specified
                if documents is None:                          
                    documents=list(self.documents.values())

                if not documents:                                 #If still empty
                   raise TrainingDataBotError("No documents to parse ")
                
                #Use default task types if none specified
                if task_types is None:
                   task_types=[TaskType.QA_GENERATION , TaskType.CLASSIFICATION]

                #Create processing job
                job=ProcessingJob(
                    name=f"Process{len(documents)} documents",
                    job_type ="document_processing",
                    total_items=len(documents)*len(task_types),
                    input_data={
                        "document_count":len(documents),
                        "task_types":[t.value for t in task_types],
                        "quality_filter": quality_filter,
                    }
                )
                self.jobs[job.id]=job
                job.status=ProcessingStatus.PROCESSING

                #process documents
                all_examples=[]

                for doc in documents:
                    #Process document (chunking,cleaning)
                    chunks=await self.preprocessor.process_document(doc)

                    #Process each chunk with each task types
                    for task_type in task_types:
                        for chunk in chunks:
                            try:
                                result=await self.task_manager.execute_task(
                                    task_type=task_type,
                                    input_chunk=chunk,
                                    client=self.ai_client
                                )

                                #Create Training example
                                example=TrainingExample(
                                    input_text=chunk.content,
                                    output_text=result.output,
                                    task_type=task_type,
                                    source_document_id=chunk.id,
                                    template_id=result.template_id,
                                    quality_scores=result.quality_scores,
                                )

                                #apply quality filtering if enabled
                                if quality_filter:
                                    quality_report=await self.evaluator.evaluate(example)
                                    if quality_report.passed:
                                        all_examples.append(example)
                                        example.quality_approved=True
                                    else:
                                        example.quality_approved=False
                                        self.logger.debug(f"Example filtered due to porr quality")
                                else:
                                    all_examples.append(example)

                                job.processed_items+=1

                            except Exception as e:
                                self.logger.error(f"Failed to process chunk: {e}")
                                job.failed_items+=1
                                continue

                #Create dataset
                dataset=Dataset(
                    name=f"Generated Dataset {len(self.datasets)+1}",
                    description=f"Dataset generated from {len(documents)} document",
                    examples=all_examples,
                )

                # Store dataset
                self.datasets[dataset.id]=dataset
                
                #Update job status
                job.status=ProcessingStatus.COMPLETED
                job.output_data={
                    "dataset_id":str(dataset.id),
                    "examples_generated":len(all_examples),
                    "quality_filtered":quality_filter,
                }
                self.logger.info(f"Processing completed, Generated{len(all_examples)}")
                return dataset
                        
            except Exception as e:
                if 'job' in locals():
                    job.status=ProcessingStatus.FAILED
                    job.error_message=str(e)
                self.logger.error(f"Document processing failed: {e}")
                raise

    async def evaluate_dataset(
            self,
            dataset: Dataset,
            detailed_report: bool=True
    ) -> QualityReport:
        """
        Evaluate the quality of a dataset.

        Args:
            dataset: Dataset to evaluate
            detailed_report:Whether to include detailed metrices
        Returns:
            Quality report
        """  
        with LogContext("dataset_evaluation",dataset_id=str(dataset.id)):
            try:
                report=await self.evaluator.evaluate_dataset(
                    dataset=dataset,
                    detailed=detailed_report
                )

                self.logger.info(
                    f"Dataset evaluation completed. Overall score:{report.overall_score:.2f}"
                )
                return report
            except Exception as e:
                self.logger.error(f"Dataset evaluation failed: {e}")
                raise
    
    # async def



         


