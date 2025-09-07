"""
Training Data Curation Bot

Enterprise-grade training data curation bot for 
LLM fine-tuning using Decado + Python automation.

"""

__version__="0.1.0"
__author__="Training Data Bot Team"
__email__="tan04sh29210@gmail.com"
__description__="Enterprise-grade training data curation bot for LLM fine-tuning"


from .core.config import settings
from .core.logging import get_logger
from .core.exceptions import TrainingDataBotError

from .bot import TrainingDataBot

from .sources import(
    PDFLoader,
    WebLoader,
    DocumentLoader,
    UnifiedLoader,              # Boss who decides which worker to use
)
from .tasks import(
    QAGenerator,
    ClassificationGenerator,
    SummarizationGenerator,
    TaskTemplate,              # The instruction sheets for workers
)

from .decodo import DecodoClient
from .preprocessing import TextPreprocessor
from .evaluation import QualityEvaluator
from .storage import DatasetExporter

__all_=[
    #Core
    "TrainingDataBot",
    "settings",
    "get_logger",
    "TrainingDataBotError",

    #Sources
    "PDFLoader",
    "WebLoader",
    "DocumentLoader",
    "UnifiedLoader",

    #Tasks
    "QAGenerator",
    "ClassificationGenerator",
    "SummarizationGenerator",
    "TaskTemplate",

    #Services
    "DecodoClient",
    "TextPreprocessor",
    "QualityEvaluator",
    "DatasetExporter",
]
