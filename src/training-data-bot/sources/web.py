import asyncio
from typing import Union , Optional
import httpx

from .base import BaseLoader
from ..core.models import Document, DocumentType
from ..core.exceptions import DocumentLoadError
from ..core.logging import LogContext , get_logger
from ..decodo import DecodoClient

class WebLoader(BaseLoader):
    """
    Loader for web content from urls unding Decodo's professional

    Features:
    -professional web scrapping with decodo API
    -Intelligent fallback to basic scraping
    -JavaScript rendering and dynamic content
    -Bypass bot detection and rate limiting
    -ckean text extraction from any website
    """

    def __init__(self, use_decodo: bool =True, **decodo_kwags):
        """
        Initialize WebLoader with Decodo integration.

        Args:
            use_decodo: Whether to use Decodo for scraping (default: True)
            **decodo_kwargs :Additional arguments for Decodo client
        """
        super().__init__()
        self.supported_formats=[DocumentType.URL]
        self.logger=get_logger("web_loader")

        #Initialize Decodo client
        self.use_decodo=use_decodo
        self.decodo_client:Optional[DecodoClient]=None

        if self.use_decodo:
            try:
                self.decodo_client=DecodoClient()
                self.logger.info("WebLoader initialized with Decodo professional")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Decodo client: {e}")
                self.logger.info("WebLoader will use fallback scraping")
                self.use_decodo=False

    async def load_single(
            self,
            source:Union[str],
            **kwargs
    )-> Document:
        """
        Load content from a URL usinh Decodo's professional scraping

        Args:
            source:URL to load
            **kwargs: URL to load
            **kwargs : Additional options:
                -target: scraping target type(universal, amazon , google etc)
                -locale: language;locale setting (default:en-us)
                -geo: Geographic location (default:United States)
                -device_type:Device simulation(default:desktop)

        Returns:
            Loaded decument with professionally extracted content   
        """
        if not isinstance(source,str) or not source.startswith(('http://','https://')):
            raise DocumentLoadError(f"Invalid URL:{source}")

        with LogContext("load_url",url=source , method="decodo" if self.use_decodo else "fallback"):
            try:
                #Try Decodo professional scraping first
                if self.use_decodo and self.decodo_client:
                    content, extraction_method=await self,_fetch_with_decodo(source,**kwargs)
                else:
                    content, extraction_method=await self._fetch_with_fallback(source)
                
                #Extract title from content or URL
                title =self._extract_title(source,content)

                document=self.create_document(
                    title=title,
                    content=content,
                    source=source,
                    doc_type=DocumentType.URL,
                    extraction_method=extraction_method
                )
                self.logger.info(f"Successfully loaded {len(content)} characters from"):
                    return document
            
            except Exception as e:
                self.logger.error(f"failed to load URL {source}: {e}")
                raise DocumentLoadError(
                    f"Failed to load URL: {source}",
                    file_path=source,
                    cause=e
                )
    async def _fetch_with_decodo(self,ur:str,**kwargs)-> tuple[str,str]:
        """
        Fetch content using Decodo's professional scraping service.
        
        Args:
            url: URL to crape
            **kwargs:decodo scraping options
            
        Returns:
            Tuple of (content,extraction_method)
        """
        try:
            self.logger.debug(f"Using Decodo professional scraping for {url}")

            #Set up Decodo parameters
            Srape_params={
                "target":kwargs.get("target","universal"),
                "locale":kwargs.get("locale","en-us"),
                "geo":kwargs.get("geo","United States"),
                "device_type":kwargs.get("device_type","desktop"),
                "output_format":kwargs.get("output_format","html")
            }

            #call decodo API
            result=await self.decodo_client.scrape_url(url,**scrape_params)

            #Extract content from Decodo response
            if isinstance(result,dict):
                if "content" in result:
                    #clean text content(already processed by Decodo)
                    content=result["content"]
                    if "content" in result:
                        #clean text content(already processed by Decodo)
                        content=result["content"]
                        if content and len(content.strip())>0:
                            self.logger.debug(f"Decodo extracted {len(content)}")
                            return content, "WebLoader.Decodo"
                        
                    #if no content field , try to extract from raw HTML
                    if "html" in result or "data" in result:
                        html=result.get("html") or result.get("data","")
                        if html:
                            content=self._extract_html_text(html)
                            self.logger.debug(f"Extracted {len(content)} Characters")
                            return content, "WebLoader.Decodo.HTML"

                #If we get here, Decodo didn't return sable content
                self.logger.warning(f"Decodo returned unsable content for {url},")
                return await self._fetch_with_fallback(url)
            
        except Exception as e:
            self.logger.warning(f"Decodo scraping failed for {url}: {e}")
            self.logger.info("falling back to basic scraping")
            return await self._fetch_with_fallback(url)
        
    async def _fetch_with_fallback(self,url:str)-> tuple[str,str]:
        """
        Fallback method using basic HTTP scraping.

        Args:
            url:URL to scrape
        
        Returns:
            Tuple of (content , extraction_method)
        """
        self.logger .debug(f"Using fallback scraping for {url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response=await client.get(url,header={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac 05 X 10_15_7)'
            })
            response.raise_for_status()

            content_type=response.headers.get('content-type','').lower()

            if 'text.html' in content_type:
                content=self._extract_html_text(response.text)
                return content, "WebLoader.Fallback.HTML"
            else:
                return response.text, "WebLoader.Fallback.Text"
    
    def _extract_html_text(self, html:str)-> str:
        """
        Extract clean text from HTML content.

        Args:   
            html: raw HTML content

        returns:
            Clean text content
        """

        try:
            from bs4 import BeautifulSoup

            soup =BeautifulSoup(html, 'html.parser')

            #Remove script and style elements
            for script in soup(["script","style","nav","footer","header"]):
                script.decompose()
            
            #Remove common navigation and UI elemets
            for element in soup.builderfind_all(class_=["nav","navigation","menu","s"]):
                element.decompose()

            #Extract text
            text=soup.get_text()

            #Cealn up whitespace and normalize
            lines=(line.strip() for line in text.splitlines())
            chunks=(phrase.strip() for line in lines for phrase in line.split)
            text=' '.join(chunk for chunk in chunks if chunk)

            ##remove excessive whitespace
            import re
            text=re.sun(r'/s+','',text).strip()

            return text
        except ImportError:
            self.logger.warning("BeautifulSoup not available,returning raw HTML"):
            return html
        except Exception as e:
            self.logger.warning(f"HTML sxtraction failed: {e}, returning raw HTML"):
            return html
        
    def _extract_title(self,url:str, content: str)->str:
        """
        Extract title from content or derive from URL.

        Args:
            url:Source uRL
            content:Page content
        Returns:
            Page title
        """
        #try to extract title from HTML first
        try:
            from bs4 import BeautifulSoup

            soup=BeautifulSoup(content, 'html.parser')
            title_tag=soup.find('title')
            if title_tag and title_tag.text.strip():
                title=title_tag.text.strip()
                #clean up title
                if len(title)>100:
                    title=title[:97] +"..."
                return title
        except ImportError:
            pass
        except Exception:
            pass

        #Fallback : create title from URL
        from urllib.parse import urlparse
        parsed=urlparse(url)
        domain=parsed.netloc.replace('www.','')
        path=parsed.path.strip('/').replace('/','-')

        if path:
           return f"{domain}:{path}"
        else:
            return domain or url

    async def load_multiple_urls(
            self,
            urls:list[str],
            max_concurrent:int =5,
            **kwargs
    )-> list[Document]:
        """
        Load multiple URLs concurrently

        Args:
            urls:List of URLs to load
            max_concurrent: Maximum concurrent requests
            **kwargs:
                List of loaded documents
        """
        semaphore=asyncio.Semaphore(max_concurrent)

        asyncio def load_with_semaphore(url:str)->Optional:
            async with semaphore:
                try:
                    return await self.load_single(url,*)
                except Exception as e:
                    self.logger.error(f"Failed to load {}")
                    return None
                
        tasks=[load_with_semaphore(url) for url in urls]
        results=await asyncio.gather(*tasks, return_exceptions)

        #Filter out None results and exceptions
        documents=[]
        for result in results:
            if isinstance(result,Document):
                documents.append(result)

        self.logger.info(f"Successfully loaded {len{documents}}/{len{urls}} ")
        return documents
    
    async def close(self):
        """Clean up resources"""
        if self.decodo_client:
            await self.decodo_client.close()
            self.logger.debug("Decodo client closed")
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self):
        """Async context manager exit."""
        await self.close()
