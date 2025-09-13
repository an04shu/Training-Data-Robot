import asyncio
from typing import Dict, List, Optional
from uuid import UUID , uuid4

from ..core.models import TaskType , TaskTemplate , TaskResult, TextChunk
from ..core.exceptions import TaskError , TemplateError 
from ..core.logging import get_logger , LogContext
from .qa_generation import QAGenerator
from .classification import ClassificationGenerator
from .summarization import SummarizationGenerator

class TaskManager:
    """
    Manages task templates and execution.
    """
    def __init__(self):
        self.logger=get_logger("task_manager")
        self.templates:Dict[UUID, TaskTemplate]={}

        #Initialise task generations
        self.generators={
            TaskType.QA_GENERATION: QAGenerator(),
            TaskType.CLASSIFICATION: ClassificationGenerator(),
            TaskType.SUMMARIZATION: SummarizationGenerator()
        }

        #Load default templates
        self._load_default_templates()

    async def execute_task(
        self,
        task_type: TaskType,
        input_chunk:TextChunk,
        client,
        template_id: Optional[UUID]=None
    )-> TaskResult:
        """
        Execute a task on a text chunk.

        Args:
            task_type:Type of task to execute
            input_chunk:Text chunk to process
            client: Decodo client for API calls
            template_id:Optional specific template to use

        Returns:
            Task execution result
        """
        with LogContext("execute_task", task_type=task_type.value,chunk_id=str()):
            try:
                #Get template
                if template_id:
                    Template=self.templates.get(template_id)
                    if not template:
                        raise TemplateError(f"Template not found: {template_id}")
                else:
                    template=self._get_default_template(task_type)
                
                #Get appropriate generator
                generator=self.generators.get(task_type)
                if not generator:
                    raise TaskError(f"No generator available for task type :{task_type}")
                

                #Execute task
                result=await generator.execute(
                    template=template,
                    input_chunk=input_chunk,
                    client=client
                )

                self.logger.debug(f"Task {task_type.value}")
                return result
            
            except Exception as e:
                self.logger.error(f"Task execution failed : {e}")

                #Return failed result
                return TaskResult(
                    task_id=uuid4(),
                    template_id=template_id or uuid4(),
                    input_chunk_id=input_chunk.id,
                    output="",
                    status=processingStatus.FAILED,
                    error_message=str(e),
                    processing_time=0.0
                )
