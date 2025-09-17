"""
Custom exceptions for Training Data Bot.

This module defines all custom exceptions used throughout the application
with proper error codes, messages, and context information.
"""

from typing import Any, Dict, Optional
import traceback


class TrainingDataBotError(Exception):
    """Base exception for all Training Data Bot errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        
        # Create detailed error message
        error_msg = f"[{error_code}] {message}"
        if context:
            error_msg += f" | Context: {context}"
        if cause:
            error_msg += f" | Caused by: {str(cause)}"
            
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc(),
        }


# Configuration Errors
class ConfigurationError(TrainingDataBotError):
    """Raised when there are configuration issues."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Required configuration '{config_key}' is missing",
            config_key=config_key,
            error_code="MISSING_CONFIG",
            **kwargs
        )


# API Errors  
class APIError(TrainingDataBotError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, **kwargs):
        context = kwargs.get("context", {})
        if status_code:
            context["status_code"] = status_code
        super().__init__(
            message,
            error_code="API_ERROR",
            context=context,
            **kwargs
        )


class DecodoAPIError(APIError):
    """Raised when Decodo API calls fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DECODO_API_ERROR",
            **kwargs
        )


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str = "API rate limit exceeded", **kwargs):
        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            **kwargs
        )


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str = "API authentication failed", **kwargs):
        super().__init__(
            message,
            error_code="AUTH_ERROR",
            **kwargs
        )


# Data Processing Errors
class ProcessingError(TrainingDataBotError):
    """Base class for data processing errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="PROCESSING_ERROR",
            **kwargs
        )


class DocumentLoadError(ProcessingError):
    """Raised when document loading fails."""
    
    def __init__(self, message: str, file_path: str = None, **kwargs):
        context = kwargs.get("context", {})
        if file_path:
            context["file_path"] = file_path
        super().__init__(
            message,
            error_code="DOCUMENT_LOAD_ERROR",
            context=context,
            **kwargs
        )


class UnsupportedFormatError(ProcessingError):
    """Raised when trying to process unsupported file formats."""
    
    def __init__(self, file_format: str, supported_formats: list = None, **kwargs):
        context = kwargs.get("context", {})
        context["file_format"] = file_format
        if supported_formats:
            context["supported_formats"] = supported_formats
            
        message = f"Unsupported file format: {file_format}"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
            
        super().__init__(
            message,
            error_code="UNSUPPORTED_FORMAT",
            context=context,
            **kwargs
        )


class TextProcessingError(ProcessingError):
    """Raised when text processing operations fail."""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        super().__init__(
            message,
            error_code="TEXT_PROCESSING_ERROR",
            context=context,
            **kwargs
        )


class ChunkingError(TextProcessingError):
    """Raised when text chunking fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            operation="chunking",
            error_code="CHUNKING_ERROR",
            **kwargs
        )


# Quality Assessment Errors
class QualityError(TrainingDataBotError):
    """Base class for quality assessment errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="QUALITY_ERROR",
            **kwargs
        )


class ToxicityError(QualityError):
    """Raised when toxicity detection fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="TOXICITY_ERROR",
            **kwargs
        )


class BiasError(QualityError):
    """Raised when bias detection fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="BIAS_ERROR",
            **kwargs
        )


class QualityThresholdError(QualityError):
    """Raised when content doesn't meet quality thresholds."""
    
    def __init__(self, message: str, metric: str = None, score: float = None, 
                 threshold: float = None, **kwargs):
        context = kwargs.get("context", {})
        if metric:
            context["metric"] = metric
        if score is not None:
            context["score"] = score
        if threshold is not None:
            context["threshold"] = threshold
            
        super().__init__(
            message,
            error_code="QUALITY_THRESHOLD_ERROR",
            context=context,
            **kwargs
        )


# Storage Errors
class StorageError(TrainingDataBotError):
    """Base class for storage-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="STORAGE_ERROR",
            **kwargs
        )


class DatabaseError(StorageError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        super().__init__(
            message,
            error_code="DATABASE_ERROR",
            context=context,
            **kwargs
        )


class ExportError(StorageError):
    """Raised when data export fails."""
    
    def __init__(self, message: str, format: str = None, **kwargs):
        context = kwargs.get("context", {})
        if format:
            context["format"] = format
        super().__init__(
            message,
            error_code="EXPORT_ERROR",
            context=context,
            **kwargs
        )


# Task Execution Errors
class TaskError(TrainingDataBotError):
    """Base class for task execution errors."""
    
    def __init__(self, message: str, task_type: str = None, **kwargs):
        context = kwargs.get("context", {})
        if task_type:
            context["task_type"] = task_type
        super().__init__(
            message,
            error_code="TASK_ERROR",
            context=context,
            **kwargs
        )


class TemplateError(TaskError):
    """Raised when task template processing fails."""
    
    def __init__(self, message: str, template_name: str = None, **kwargs):
        context = kwargs.get("context", {})
        if template_name:
            context["template_name"] = template_name
        super().__init__(
            message,
            error_code="TEMPLATE_ERROR",
            context=context,
            **kwargs
        )


class TaskTimeoutError(TaskError):
    """Raised when task execution times out."""
    
    def __init__(self, message: str = "Task execution timed out", timeout: int = None, **kwargs):
        context = kwargs.get("context", {})
        if timeout:
            context["timeout"] = timeout
        super().__init__(
            message,
            error_code="TASK_TIMEOUT",
            context=context,
            **kwargs
        )


# Validation Errors
class ValidationError(TrainingDataBotError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs
        )


# Utility functions
def handle_exception(func):
    """Decorator to handle exceptions and convert them to TrainingDataBotError."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TrainingDataBotError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Convert unknown exceptions to our base exception
            raise TrainingDataBotError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__},
                cause=e,
            )
    return wrapper


def reraise_as(exception_class: type, message: str = None, **kwargs):
    """Context manager to catch exceptions and reraise as custom exception."""
    class _RerraiseContext:
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                error_message = message or f"Operation failed: {str(exc_val)}"
                raise exception_class(
                    message=error_message,
                    cause=exc_val,
                    **kwargs
                )
    
    return _RerraiseContext() 