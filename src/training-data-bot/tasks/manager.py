"""
Task manager for coordinating task execution.

This module provides the main TaskManager class that coordinates
task template execution and result processing.
"""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from ..core.models import TaskType, TaskTemplate, TaskResult, TextChunk, ProcessingStatus
from ..core.exceptions import TaskError, TemplateError
from ..core.logging import get_logger, LogContext
from .qa_generation import QAGenerator
from .classification import ClassificationGenerator
from .summarization import SummarizationGenerator


class TaskManager:
    """
    Manages task templates and execution.
    """

    def __init__(self):
        self.logger = get_logger("task_manager")
        self.templates: Dict[UUID, TaskTemplate] = {}
        
        # Initialize task generators
        self.generators = {
            TaskType.QA_GENERATION: QAGenerator(),
            TaskType.CLASSIFICATION: ClassificationGenerator(),
            TaskType.SUMMARIZATION: SummarizationGenerator(),
        }
        
        # Load default templates
        self._load_default_templates()

    async def execute_task(
        self,
        task_type: TaskType,
        input_chunk: TextChunk,
        client,
        template_id: Optional[UUID] = None
    ) -> TaskResult:
        """
        Execute a task on a text chunk.
        
        Args:
            task_type: Type of task to execute
            input_chunk: Text chunk to process
            client: Decodo client for API calls
            template_id: Optional specific template to use
            
        Returns:
            Task execution result
        """
        with LogContext("execute_task", task_type=task_type.value, chunk_id=str(input_chunk.id)):
            try:
                # Get template
                if template_id:
                    template = self.templates.get(template_id)
                    if not template:
                        raise TemplateError(f"Template not found: {template_id}")
                else:
                    template = self._get_default_template(task_type)
                
                # Get appropriate generator
                generator = self.generators.get(task_type)
                if not generator:
                    raise TaskError(f"No generator available for task type: {task_type}")
                
                # Execute task
                result = await generator.execute(
                    template=template,
                    input_chunk=input_chunk,
                    client=client
                )
                
                self.logger.debug(f"Task {task_type.value} completed for chunk {input_chunk.id}")
                return result
                
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                
                # Return failed result
                return TaskResult(
                    task_id=uuid4(),
                    template_id=template_id or uuid4(),
                    input_chunk_id=input_chunk.id,
                    output="",
                    status=ProcessingStatus.FAILED,
                    error_message=str(e),
                    processing_time=0.0
                )

    async def create_template(
        self,
        name: str,
        task_type: TaskType,
        prompt_template: str,
        description: str = "",
        **parameters
    ) -> UUID:
        """
        Create a new task template.
        
        Args:
            name: Template name
            task_type: Task type
            prompt_template: Jinja2 template string
            description: Template description
            **parameters: Additional parameters
            
        Returns:
            Template ID
        """
        template = TaskTemplate(
            name=name,
            task_type=task_type,
            description=description,
            prompt_template=prompt_template,
            parameters=parameters
        )
        
        self.templates[template.id] = template
        self.logger.info(f"Created template '{name}' with ID: {template.id}")
        
        return template.id

    def get_template(self, template_id: UUID) -> Optional[TaskTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def list_templates(self, task_type: Optional[TaskType] = None) -> List[TaskTemplate]:
        """List all templates, optionally filtered by task type."""
        templates = list(self.templates.values())
        
        if task_type:
            templates = [t for t in templates if t.task_type == task_type]
        
        return templates

    def _get_default_template(self, task_type: TaskType) -> TaskTemplate:
        """Get default template for task type."""
        # Find first template of the given type
        for template in self.templates.values():
            if template.task_type == task_type:
                return template
        
        # If no template found, create a basic one
        return self._create_basic_template(task_type)

    def _create_basic_template(self, task_type: TaskType) -> TaskTemplate:
        """Create a basic template for the task type."""
        templates = {
            TaskType.QA_GENERATION: {
                "name": "Basic QA Generation",
                "prompt": "Generate question-answer pairs from the following text:\n\n{{ text }}\n\nQ&A:",
                "description": "Basic question-answer generation template"
            },
            TaskType.CLASSIFICATION: {
                "name": "Basic Classification", 
                "prompt": "Classify the following text:\n\n{{ text }}\n\nClassification:",
                "description": "Basic text classification template"
            },
            TaskType.SUMMARIZATION: {
                "name": "Basic Summarization",
                "prompt": "Summarize the following text:\n\n{{ text }}\n\nSummary:",
                "description": "Basic text summarization template"
            }
        }
        
        template_config = templates.get(task_type)
        if not template_config:
            raise TaskError(f"No default template available for {task_type}")
        
        template = TaskTemplate(
            name=template_config["name"],
            task_type=task_type,
            description=template_config["description"],
            prompt_template=template_config["prompt"]
        )
        
        self.templates[template.id] = template
        return template

    def _load_default_templates(self):
        """Load default templates for all task types."""
        # Create default templates for each task type
        for task_type in [TaskType.QA_GENERATION, TaskType.CLASSIFICATION, TaskType.SUMMARIZATION]:
            try:
                self._create_basic_template(task_type)
            except Exception as e:
                self.logger.error(f"Failed to create default template for {task_type}: {e}")

    async def bulk_execute(
        self,
        tasks: List[Dict],
        client,
        max_concurrent: int = 5
    ) -> List[TaskResult]:
        """
        Execute multiple tasks concurrently.
        
        Args:
            tasks: List of task specifications
            client: Decodo client
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of task results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(task_spec):
            async with semaphore:
                return await self.execute_task(
                    task_type=task_spec["task_type"],
                    input_chunk=task_spec["input_chunk"],
                    client=client,
                    template_id=task_spec.get("template_id")
                )
        
        tasks_to_execute = [execute_single(task) for task in tasks]
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, TaskResult):
                valid_results.append(result)
            else:
                self.logger.error(f"Task {i} failed: {result}")
        
        return valid_results 