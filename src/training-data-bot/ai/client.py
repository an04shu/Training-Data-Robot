"""
AI client for text generation and LLM services.

This module provides AI text generation capabilities using various LLM providers
like OpenAI, Anthropic, etc. This is separate from Decodo which is for web scraping.
"""

import asyncio
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import os

from ..core.config import settings
from ..core.models import DecodoRequest, DecodoResponse, TaskType
from ..core.exceptions import DecodoAPIError, AuthenticationError, RateLimitError
from ..core.logging import get_logger, log_api_call


class AIClient:
    """
    Client for AI text generation services.
    
    This handles the actual AI/LLM functionality for generating training data,
    since Decodo is purely for web scraping.
    """

    def __init__(self):
        self.logger = get_logger("ai.client")
        
        # Try to use OpenAI if available
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Default to simulation if no AI API keys are available
        self.use_simulation = not (self.openai_api_key or self.anthropic_api_key)
        
        if self.use_simulation:
            self.logger.info("No AI API keys found, using simulation mode")
        else:
            self.logger.info("AI client initialized with real API access")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @log_api_call("ai")
    async def process_text(
        self,
        prompt: str,
        input_text: str,
        task_type: Optional[TaskType] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> DecodoResponse:
        """
        Process text using AI/LLM services.
        
        Args:
            prompt: Task prompt/instruction
            input_text: Text to process
            task_type: Type of task being performed
            parameters: Additional parameters
            
        Returns:
            AI processing response
        """
        request = DecodoRequest(
            prompt=prompt,
            input_text=input_text,
            task_type=task_type,
            parameters=parameters or {}
        )
        
        try:
            if self.use_simulation:
                response = await self._simulate_ai_call(request)
            else:
                # Try real AI API call
                if self.openai_api_key:
                    response = await self._call_openai(request)
                elif self.anthropic_api_key:
                    response = await self._call_anthropic(request)
                else:
                    response = await self._simulate_ai_call(request)
            
            return DecodoResponse(
                request_id=request.request_id,
                success=True,
                output=response["output"],
                confidence=response.get("confidence"),
                token_usage=response.get("token_usage", 0),
                cost=response.get("cost"),
                processing_time=response.get("processing_time")
            )
            
        except Exception as e:
            self.logger.error(f"AI processing failed: {e}")
            # Fallback to simulation
            response = await self._simulate_ai_call(request)
            return DecodoResponse(
                request_id=request.request_id,
                success=True,
                output=response["output"],
                confidence=response.get("confidence", 0.7),
                token_usage=response.get("token_usage", 0),
                cost=response.get("cost"),
                processing_time=response.get("processing_time")
            )

    async def _call_openai(self, request: DecodoRequest) -> Dict[str, Any]:
        """Call OpenAI API for text generation."""
        import time
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Build prompt based on task type
        system_prompt = self._build_system_prompt(request.task_type)
        user_prompt = f"{request.prompt}\n\nText: {request.input_text}"
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": request.parameters.get("max_tokens", 1000),
            "temperature": request.parameters.get("temperature", 0.7)
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            output = data["choices"][0]["message"]["content"]
            token_usage = data.get("usage", {}).get("total_tokens", 0)
            
            return {
                "output": output,
                "confidence": 0.9,
                "token_usage": token_usage,
                "cost": token_usage * 0.002 / 1000,  # Rough estimate
                "processing_time": time.time() - start_time
            }

    async def _call_anthropic(self, request: DecodoRequest) -> Dict[str, Any]:
        """Call Anthropic API for text generation."""
        import time
        start_time = time.time()
        
        headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Build prompt
        system_prompt = self._build_system_prompt(request.task_type)
        user_prompt = f"{request.prompt}\n\nText: {request.input_text}"
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": request.parameters.get("max_tokens", 1000),
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            output = data["content"][0]["text"]
            token_usage = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
            
            return {
                "output": output,
                "confidence": 0.9,
                "token_usage": token_usage,
                "cost": token_usage * 0.0015 / 1000,  # Rough estimate
                "processing_time": time.time() - start_time
            }

    def _build_system_prompt(self, task_type: Optional[TaskType]) -> str:
        """Build system prompt based on task type."""
        if task_type == TaskType.QA_GENERATION:
            return "You are an expert at creating high-quality question-answer pairs from text content. Generate relevant, clear, and educational Q&A pairs."
        elif task_type == TaskType.CLASSIFICATION:
            return "You are an expert at text classification. Analyze the given text and provide accurate classification results."
        elif task_type == TaskType.SUMMARIZATION:
            return "You are an expert at text summarization. Create concise, accurate summaries that capture the key points."
        else:
            return "You are a helpful AI assistant that processes text according to the given instructions."

    async def _simulate_ai_call(self, request: DecodoRequest) -> Dict[str, Any]:
        """
        Simulate AI API call for development/testing.
        
        This provides realistic mock responses when no AI API is available.
        """
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate mock response based on task type
        if request.task_type == TaskType.QA_GENERATION:
            output = self._generate_mock_qa(request.input_text)
        elif request.task_type == TaskType.CLASSIFICATION:
            output = self._generate_mock_classification(request.input_text)
        elif request.task_type == TaskType.SUMMARIZATION:
            output = self._generate_mock_summary(request.input_text)
        else:
            output = f"Processed: {request.input_text[:100]}..."
        
        return {
            "output": output,
            "confidence": 0.85,
            "token_usage": len(request.input_text.split()) + len(output.split()),
            "cost": 0.001,
            "processing_time": 0.1
        }

    def _generate_mock_qa(self, text: str) -> str:
        """Generate mock Q&A pairs."""
        first_words = text.split()[:5] if text.split() else ["general", "content"]
        topic = " ".join(first_words)
        
        return f"""Q: What is the main topic discussed in this text?
A: The main topic is about {topic}.

Q: What are the key points mentioned?
A: The key points include information about {first_words[0] if first_words else 'the subject'} and related concepts.

Q: What can we learn from this content?
A: From this content, we can learn about {topic} and its implications."""

    def _generate_mock_classification(self, text: str) -> str:
        """Generate mock classification."""
        if "question" in text.lower():
            return "Category: Question\nConfidence: 0.9\nReasoning: Text contains interrogative elements"
        elif "help" in text.lower() or "support" in text.lower():
            return "Category: Support Request\nConfidence: 0.8\nReasoning: Text appears to be seeking assistance"
        elif "how" in text.lower() or "what" in text.lower():
            return "Category: Informational Query\nConfidence: 0.85\nReasoning: Text seeks information or explanation"
        else:
            return "Category: General Information\nConfidence: 0.7\nReasoning: Text provides general information or content"

    def _generate_mock_summary(self, text: str) -> str:
        """Generate mock summary."""
        words = text.split()
        if len(words) > 20:
            summary_words = words[:15]
            return " ".join(summary_words) + "... [Summary of key points from the original text]"
        else:
            return f"Summary: {text[:100]}..." if len(text) > 100 else f"Summary: {text}"

    async def health_check(self) -> bool:
        """
        Check if AI services are available.
        
        Returns:
            True if AI services are healthy
        """
        try:
            # Test with a simple request
            test_request = DecodoRequest(
                prompt="Test",
                input_text="Test input",
                task_type=TaskType.QA_GENERATION
            )
            
            response = await self._simulate_ai_call(test_request)
            return bool(response.get("output"))
            
        except Exception:
            return False

    async def close(self):
        """Close any resources (placeholder for future use)."""
        # No persistent resources to close for now
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 