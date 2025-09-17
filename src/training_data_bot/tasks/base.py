"""
Base task generator.

This module provides the abstract base class for all task generators.
"""

import time
from abc import ABC, abstractmethod
from uuid import uuid4

from ..core.models import TaskTemplate, TaskResult, TextChunk, ProcessingStatus
from ..core.logging import get_logger


class BaseTaskGenerator(ABC):
    """Base class for task generators."""

    def __init__(self):
        self.logger = get_logger(f"task.{self.__class__.__name__}")

    @abstractmethod
    async def execute(
        self,
        template: TaskTemplate,
        input_chunk: TextChunk,
        client
    ) -> TaskResult:
        """Execute the task."""
        pass

    def _render_prompt(self, template: TaskTemplate, input_chunk: TextChunk) -> str:
        """Render prompt template with input data."""
        try:
            from jinja2 import Template
            
            jinja_template = Template(template.prompt_template)
            return jinja_template.render(
                text=input_chunk.content,
                chunk=input_chunk,
                **template.parameters
            )
        except ImportError:
            # Fallback: simple string replacement
            prompt = template.prompt_template.replace("{{ text }}", input_chunk.content)
            return prompt


# For backwards compatibility
TaskTemplate = TaskTemplate 