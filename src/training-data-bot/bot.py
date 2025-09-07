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
from .sources import UnifiedLoader                 # the task worker chooser
from .decodo import DecodoClient                   #The internet detective
from .ai import AIClient                           #The AI brain
from .tasks import TaskManager
from .preprocessing import TextPreprocessor
from .evaluation import QualityEvaluator
from .storage import DatasetExporter, DatabaseManager

class TrainingDataBot:                   # smartest, most organized manager
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
            self.documents:Dict[UUID,Document]={}       #Filing cabinet for all homework
            self.datasets:Dict[UUID,Dataset]={}         #Trophy case for completed projects
            self.jobs:Dict[UUID,ProcessingJob]={}       #To-do list for all work

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
                        dir_docs=await self.loader.load_directory(source_path)
                        documents.extend(dir_docs)                      #add to new indexes
                    else:
                        doc=await self.loader.load_single(source)
                        documents.append(doc)                           #add many to same index
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
                                        self.logger.debug(f"Example filtered due to poor quality")
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
    
    async def export_dataset(
        self,
        dataset:Dataset,
        output_path=Union[str,Path],                 
        format: ExportFormat=ExportFormat.JSONL,     
        split_data: bool =True,                      
        **kwargs
    ) -> Path :
        """
        Export datset to file.

        Args:
            dataset: Dataset to export
            output_path: Output file path
            format: Export format
            split_data: weather to create train/val/test splits
            **kwargs: Additional export options

        Returns:
            Path to export file(s)
        """
        with LogContext("dataset_export",dataset_id=str(dataset.id),format=format.value):
            try:
                exported_path=await self.exporter.export_dataset(
                    dataset=dataset,
                    output_path=Path(output_path),
                    format=format,
                    split_data=split_data,
                    **kwargs
                )
            
                # Update dataset metadata
                dataset.export_format=format
                dataset.export_path=exported_path

                self.logger.info(f"Dataset exported to {exported_path}")
                return exported_path

            except Exception as e:
                self.logger.error(f"Dataset export failed: {e}")
                raise

        def get_statistics(self) -> Dict[str,Any]:
            """Get bot statictics and status."""
            return{
                "documents":{
                    "total":len(self.documents),
                    "by_type": self._count_by_type(self.documents.values(),"doc_type"),
                    "total_size":sum(doc.size for doc in self.documents.values()),
                },
                "datasets":{
                    "total":len(self.datasets),
                    "by_task_type": self._count_examples_by_task_type(),
                    "total_examples":sum(len(ds.examples) for ds in self.datasets.values())
                },
                "jobs":{
                    "total":len(self.jobs),
                    "by_status": self._count_by_type(self.jobs.values(),"status"),
                    "active": len([j for j in self.jobs.values() if j.status ==ProcessingStatus.PROCESSING]),
                },
                "quality":{
                    "approved_examples": sum(
                        len([ex for ex in ds.examples if ex.quality_approved])
                        for ds in self.datasets.values()
                    ),
                    "total_examples":sum(len(ds.examples) for ds in self.datasets.values()),
                }
            }

        def _count_by_type(self,items, attr_name:str) -> Dict[str,int]:
            """Count items by attribute value."""
            counts={}
            for item in items:
                value=getattr(item, attr_name)
                if hasattr(value,'value'):  #Handle enums
                    value=value.value
                counts[str(value)]=counts.get(str(value),0) +1
            return counts

        def _count_examples_by_task_type(self) -> Dict[str,int]:
            """Count examples by task type accross all datasets."""
            counts={}
            for dataset in self.datasets.values():
                for example in dataset.examples:
                    task_type=example.task_type.value
                    counts[task_type]=counts.get(task_type,0)+1
            return counts

        async def cleanup(self):
            """Cleanup resources and close connections."""
            try:
                #Close database connections
                await self.db_manager.close()

                #Close loader (which will close its WebLoader and Decodo client)
                if hasattr(self.loader,'close'):
                    await self.loader.close()

                #Close remaining HTTP clients
                if hasattr(self.decodo_client,'close'):
                    await self.decodo_client.close()

                if hasattr(self.ai_client, 'close'):
                    await self.ai_client.close()

                self.logger.info("Bot cleanup completed")

            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")

        async def __aenter__(self):
            """"async context manager entry."""
            return self
        async def __aexit__(self,exc_type,exc_val,exc_tb):
            """Async context manager exit."""
            await self.cleanup()

        # Convenience methods
        async def quick_process(
            self,
            source:Union[str,Path],
            output_path: Union[str,Path],
            task_types:Optional[List[TaskType]]=None,
            export_format: ExportFormat=ExportFormat.JSONL
        ) -> Dataset:
            """
            Quick end to end processing

            Args:
                source: Single source to process
                output_path: Where to save the dataset
                task_types:Task types to execute
                export_format:export format

            Returns:
                Generated and exported dataset
            """
            #Load documents
            documents=await self.load_document([source])

            #Process documents
            dataset=await self.process_documents(
                documents=documents,
                task_types=task_types
            )

            #Export dataset
            await self.export_dataset(
                dataset=dataset,
                output_path=output_path,
                format=export_format
            )

            return dataset


    """
    One-Click Example

    bot = TrainingDataBot()
    dataset = await bot.quick_process("my_essay.pdf", "
    training_data.jsonl")
    Done!
    
    """


