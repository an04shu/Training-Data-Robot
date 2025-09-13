"""Decodo API client for web scrapping and data extraction.

This module provides  the main client for interacting with Decodo's web scraping 
decodo specializes in web scraping , not AI text generation.
"""

import asyncio
from typing import Dict , Any , Optional, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from ..core.config import settings
from ..core.exceptions import get_logger , log_api_call

class DecodoClient:
    """
    Clinet for interacting with Decodo's web scrapping API's.

    Decodo provides web scrapping and data extraction services,
    not AI text generation . Use this client to colect data from websites.
    """
    def __init__(self):
        self.logger=get_logger("decodo.client")
        self.base_url="https://scraper-api.decodo.com"   #corect API
        self.timeout=settings.decodo.timeout
        self.max_retries=settings.decodo,max_retries

        #Get authentication details from environment
        import os
        self.username=os.getenv("DECODO_USERNAME") or "U0000283015"
        self.password=os.getenv("DECODO_PASSWORD") or "PW_1fcac9f9d8fce2a6ec39"
        self.basic_auth_token=os.getenv("DECODO_BASIC_AUTH") or "VTAwMDAyODMqM"

        #set up headers
        headers={
            "accept":"application/json",
            "content-type":"application/json",
            "User-Agent":"TrainingDataBot/1.0"
        }

        #Add Basic authentication using the provided token
        if self.basic_auth_token:
            headers["authentication"]=f"Basic {self.basic_auth_token}"

        #Create HTTP client
        self.client=httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers
        )
    
    @retry(
        stop=stop_after_attempt(3)
        wait=wait_exponential(multiplier=1,min=4,max=10)
    )
    @log_api_call("decodo")
    async def scrape_url(
        self,
        url:str,
        target:str="universal",
        locale:str="en-us",
        geo:str="United States",
        device_type:str="desktop",
        output_format:str="html"
    )->Dict[str,Any]:
        """
        Scrape content froma URL using Decodo's scraping API.

        Args:
            url:URL to scrape
            target:Scraping target type(universal , amazon , google , etc...)
            locale: Language/locale setting 
            geo:Geographic location
            device_type:Device type simulation
            output_format:output format(html,json,etc.)

        Return:
            Scraped content and metadata
        """
        try:
            #Use the correct Decodo API format
            payload={
                "url":url,
                "headless":"html"   #Standard format as per Decodo docs
            }
            # add Additional parameters for professional features
            if target !="universal":
                payload["target"]=target
            if locale !="en-us":
                payload["locale"]=locale
            if geo !="United states":
                payload["geo"]=geo
            if device_type !="desktop":
                payload["device_type"]=device_type

            #Make request to correct endpoint
            self.logger.debug(f"Making request to Decodo")
