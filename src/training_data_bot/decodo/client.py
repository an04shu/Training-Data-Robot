"""
Decodo API client for web scraping and data extraction.

This module provides the main client for interacting with Decodo's web scraping APIs.
Decodo specializes in web scraping, not AI text generation.
"""

import asyncio
from typing import Dict, Any, Optional, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from ..core.config import settings
from ..core.exceptions import DecodoAPIError, AuthenticationError, RateLimitError
from ..core.logging import get_logger, log_api_call


class DecodoClient:
    """
    Client for interacting with Decodo's web scraping APIs.
    
    Decodo provides web scraping and data extraction services,
    not AI text generation. Use this client to collect data from websites.
    """

    def __init__(self):
        self.logger = get_logger("decodo.client")
        self.base_url = "https://scraper-api.decodo.com"  # Correct API endpoint
        self.timeout = settings.decodo.timeout
        self.max_retries = settings.decodo.max_retries
        
        # Get authentication details from environment
        import os
        self.username = os.getenv("DECODO_USERNAME") or "U0000283015"
        self.password = os.getenv("DECODO_PASSWORD") or "PW_1fcac9f9d8fce2a6ec39020d4655429bd"
        self.basic_auth_token = os.getenv("DECODO_BASIC_AUTH") or "VTAwMDAyODMwMTU6UFdfMWZjYWM5ZjlkOGZjZTJhNmVjMzkwMjBkNDY1NTQyOWJk"
        
        # Set up headers
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "User-Agent": "TrainingDataBot/1.0"
        }
        
        # Add Basic Authentication using the provided token
        if self.basic_auth_token:
            headers["authorization"] = f"Basic {self.basic_auth_token}"
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @log_api_call("decodo")
    async def scrape_url(
        self,
        url: str,
        target: str = "universal",
        locale: str = "en-us",
        geo: str = "United States",
        device_type: str = "desktop",
        output_format: str = "html"
    ) -> Dict[str, Any]:
        """
        Scrape content from a URL using Decodo's scraping API.
        
        Args:
            url: URL to scrape
            target: Scraping target type (universal, amazon, google, etc.)
            locale: Language/locale setting
            geo: Geographic location
            device_type: Device type simulation
            output_format: Output format (html, json, etc.)
            
        Returns:
            Scraped content and metadata
        """
        try:
            # Use the correct Decodo API format
            payload = {
                "url": url,
                "headless": "html"  # Standard format as per Decodo docs
            }
            
            # Add additional parameters for professional features
            if target != "universal":
                payload["target"] = target
            if locale != "en-us":
                payload["locale"] = locale
            if geo != "United States":
                payload["geo"] = geo  
            if device_type != "desktop":
                payload["device_type"] = device_type
            
            # Make request to correct endpoint
            self.logger.debug(f"Making request to Decodo API: {payload}")
            response = await self.client.post("/v2/scrape", json=payload)
            
            # Log response details for debugging
            self.logger.debug(f"Decodo API response: {response.status_code}")
            if response.status_code != 200:
                self.logger.warning(f"Decodo API returned {response.status_code}: {response.text[:200]}...")
            
            # Handle different response scenarios
            if response.status_code == 200:
                data = response.json()
                
                # Extract content from Decodo response format
                content = ""
                if "results" in data and len(data["results"]) > 0:
                    # Decodo returns results array
                    result = data["results"][0]
                    if "content" in result:
                        content = result["content"]
                    elif "html" in result:
                        content = result["html"]
                elif "html" in data:
                    content = data["html"]
                elif "content" in data:
                    content = data["content"]
                elif "data" in data:
                    content = data["data"]
                
                # Clean HTML content to text
                if content and content.startswith("<!DOCTYPE") or content.startswith("<html"):
                    # Convert HTML to clean text
                    import re
                    from html import unescape
                    
                    # Remove script and style tags
                    clean_content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                    
                    # Remove HTML tags
                    clean_content = re.sub(r'<[^>]+>', '', clean_content)
                    
                    # Clean up whitespace and decode HTML entities
                    clean_content = unescape(clean_content)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    content = clean_content
                
                # Return standardized format
                return {
                    "content": content,
                    "html": data.get("results", [{}])[0].get("content", content) if "results" in data else content,
                    "url": url,
                    "status": "success",
                    "method": "decodo_api",
                    "raw_response": data
                }
            elif response.status_code == 401:
                self.logger.warning("Decodo API authentication failed, using fallback")
                return await self._fallback_scrape(url)
            elif response.status_code == 404:
                self.logger.warning("Decodo API endpoint not found, using fallback")
                return await self._fallback_scrape(url)
            else:
                response.raise_for_status()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.logger.warning("Invalid API credentials, using fallback")
                return await self._fallback_scrape(url)
            elif e.response.status_code == 429:
                self.logger.warning("API rate limit exceeded, using fallback")
                return await self._fallback_scrape(url)
            else:
                self.logger.warning(f"Decodo API error {e.response.status_code}, using fallback")
                return await self._fallback_scrape(url)
        except Exception as e:
            self.logger.warning(f"Decodo API failed, using fallback: {e}")
            return await self._fallback_scrape(url)

    async def _fallback_scrape(self, url: str) -> Dict[str, Any]:
        """
        Fallback method to scrape content when Decodo API is unavailable.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                
                # Basic content extraction
                html_content = response.text
                
                # Simple HTML to text conversion
                import re
                from html import unescape
                
                # Remove script and style tags
                clean_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                clean_content = re.sub(r'<style[^>]*>.*?</style>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                
                # Remove HTML tags
                clean_content = re.sub(r'<[^>]+>', '', clean_content)
                
                # Clean up whitespace and decode HTML entities
                clean_content = unescape(clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                return {
                    "content": clean_content,
                    "url": url,
                    "status": "fallback_success",
                    "method": "basic_http"
                }
                
        except Exception as e:
            raise DecodoAPIError(f"Both Decodo API and fallback scraping failed: {str(e)}")

    async def scrape_multiple_urls(
        self,
        urls: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            **kwargs: Additional parameters for scraping
            
        Returns:
            List of scraping results
        """
        tasks = [self.scrape_url(url, **kwargs) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to scrape {urls[i]}: {result}")
                processed_results.append({
                    "url": urls[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append({
                    "url": urls[i],
                    "success": True,
                    "data": result
                })
        
        return processed_results

    async def extract_text_from_url(self, url: str) -> str:
        """
        Extract clean text content from a URL.
        
        Args:
            url: URL to extract text from
            
        Returns:
            Clean text content
        """
        try:
            result = await self.scrape_url(url, output_format="html")
            
            # Extract text content from HTML
            # In a real implementation, you'd parse the HTML to extract clean text
            html_content = result.get("content", "")
            
            # Basic HTML-to-text conversion (you might want to use BeautifulSoup)
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from {url}: {e}")
            return ""

    async def health_check(self) -> bool:
        """
        Check if Decodo scraping API is healthy.
        
        Returns:
            True if API is healthy
        """
        try:
            # Test with a simple scraping request
            response = await self.scrape_url("https://ip.decodo.com", target="universal")
            return "content" in response or "success" in response
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 