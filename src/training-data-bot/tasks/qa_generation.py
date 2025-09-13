import time
from uuid import uuid4

from .base import BaseTaskGenerator
from ..core.models import TaskTemplate, TaskResult, TextChunk

class QAGenerator(BaseTaskGenerator):
    """Generate questions-answer pairs from text."""

    async def execute(
        self,
        template:TaskTemplate,
        input_chunk:TextChunk,
        client
    )-> TaskResult:
        """Execute QA generation task."""
        start_time=time.time()

        try:
            #Render prompt
            prompt=self._render_prompt(template, input_chunk)

            #Call Decodo API
            response=await client.process_text(
                prompt=prompt,
                input_text=input_chunk.content,
                task_type=template.task_type
            )

            processsing_time=time.time()-start_time

            return TaskResult(
                task_id==uuid4(),
                template_id=template.id,
                input_chunk_id=input_chunk.id,
                output=response.output or "",
                confidence=response.confidence,
                processing_time=processing_time
            )