"""
Task management module.

This module provides task templates and execution capabilities for
generating different types of training data.
"""

from .manager import TaskManager
from .qa_generation import QAGenerator
from .classification import ClassificationGenerator
from .summarization import SummarizationGenerator
from .base import TaskTemplate

__all__ = [
    "TaskManager",
    "QAGenerator", 
    "ClassificationGenerator",
    "SummarizationGenerator",
    "TaskTemplate",
] 