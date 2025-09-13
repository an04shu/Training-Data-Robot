import asyncio
from typing import Dict , Any , Optional
import httpx
from tenacity import retry , stop_after_attempt, wait_exp
import json 
import os


from ..core.config import settings
from ..core.models import DecodoRequest DecodoRequest,
from ..core.exceptions import DecodoAPIError , Authentication
from ..core.logging import get_logger , log_api_call


class AIClient:
    """Client for AI text generation services.
    
    this handles the actual AI/LLM functionality for generating training data 
    since Decodo is purely for web scraping."""

    def __init__(self):
        self.logger=get_logger("ai.client")

        #Try to use OpenAI if available
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key=os.getenv("ANTHROPHIC_API_KEY")

        #Default to simulaion if no API keys are avaialable
        self.use_simulation = not(self.openai_api_key or self.anthropic_api_key)

        if self.use_simulation:
            self.logger.info("No AI API keys found, using simulation mode")
        else:
            self.logger.info("AI client initialized with real API acess")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1,min=4,max=10)
        )
        @log_api_call("ai")
        async def process_text(
            self,
            prompt:str,
            input_text:str,
            task_type: Optional[TaskType]=None,
            parameters: Optional[Dict[str,Any]]=None
        )-> DecodoResponse:
            """
            Process text using AI/LLM services.

            Args:
                prompt: Task prompt/instruction
                input_text:Text to process
                task_types: Type of task being performed
                parameters: Additional parameters

            Returns:
                AI processing response
            """
            request=DecodoRequest(
                prompt=prompt,
                input_text=input_text,
                task_type=task_type
                parameters=parameters or {}
            )
            try:
                if self.use_simulation:
                    response=await self._simulate_ai_call(request)
                else:
                    #Try real AI API call
                    if self.openai_api_key:
                        response=await self._call_openai(request)
                    elif self.anthopic_api_key:
                        response=await self._call_anthropic(request)
                
                return DecodoResponse(
                    request_id=request.request_id,
                    sucess=True,
                    output=response["output"],
                    confidence=response.get("confidence"),
                    token_usage=response.get("token_usage",0),
                    cost=response.get("cost"),
                    processing_time=response.get("processing_time")
                )
            except Exception as e:
                self.logger.error(f"AI processing failed: {e}")
                #Fallback to simulation
                response=await self._simulate_ai_call(request)
                returnnDecodoResponse(
                    request_id=request.request_id,
                    sucess=True,
                    output_response["output"],
                    confidence=response.get("condidence",0.7),
                    token_usage=response.get("token_usage",0),
                    cost=response.get("cost"),
                    processing_time=response.get("processing_time")
                )
        
        async def _call_openai(self,request.DecodoRequest)->Dict[str,Any]:
            """Call OpenAi API for text generation."""
            import time
            start_time=time.time()
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }

            #Build prompt based on task type
            system_prompt=self._build_system_prompt(request.task_type)
            user_prompt=f"{request.prompt}\n\nText:{request.input_text}"

            payload={
                "model":"gpt-3.5-turbo",
                "messages":[
                    {"role":"system","content":System_prompt},
                    {"role": "user","content":user_prompt}
                ],
                "max_tokens":request.parameters.get("max_token",1000)           
                "temperature":request.parameteres.get("temperature",0.7)
            }
            async with httpx.SyncClient() as client:
                response=await clinet.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()

                data=response.json()
                output=data["choices"][0]["message"]["content"]
                token_usage=data.get("usafe",{}).get("total_tokens",0)

                return{
                    "output":output,
                    "confidence":0.9,
                    "token_usage":token_usage,
                    "cost":token_usage*0.002/100, #Rough estimate
                    "processing_time":time.time()-start_time
                }
            
        async def _call_anthropic(self, request:DecodoRequest)->Dict[str,Any]:
            """Call anthropic API for text generation."""
            import time
            start_time=time.time()

            headers={
                "x-api-key":self.antropic_api_key,
                "Content-Type":"application/json",
                "anthropic-version":"2023-06-01"
            }

            #Build prompt
            system_prompt=self.build_system_prompt(request.task_type)
            user_prompt=f"{request.prompt}\n\nText: {request.input_text}"

            payload={
                "model":"claude-3-haiku-20240307",
                "max_tokens":request.paramerters.get("max_token",1000),
                "system":system_prompt,
                "messages"=[
                    {"role":"user","content":user_prompt}
                ]
            }
            async with httpx.AysncClient() as client:
                response=await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()

                data=response.json()
                output=data["content"][0]["text"]
                token_usage=data.get("usage",{}).get("input_tokens",0)+data.get

                return{
                    "output":output,
                    "confidence":0.9,
                    "token_usage":token_usage,
                    "cost":token_usage* 0.0015/1000, #Rough estimate
                    "processing_time":time.time()-start_time
                }
        
        def _build_system_prompt(self,task_type:Optional[TaskType])->str:
            """Build system prompt bases on the type."""
            if task_type==TaskType.QA_GENERATION:
                return "you have an expert at creating high quality question-answer pairs  from text content. Generate relevent , "
            elif task_type==TaskType.CLASSIFICATION:
                return "you have an expert at text classification. Analayze the given text and provide accurate classification"
            elif task_type==TaskType.SUMMARIZATION:
                return "you have an expert at etxt summarization. Create concise , accurate summaries that capture the key point"
            else:
                return "You are a helpful AI assistant that process text according to the given instructions."

        async def _simulate_ai_call(self, request:DecodoRequest)->Dict[str,any]:
            """
            Simulate AI API call for development/testing/
            
            This provides realistic mock response when no AI API is available.
            """
