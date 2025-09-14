#!/usr/bin/env python3
"""
Web Scraper + AI Training Data Generator

A Streamlit application that demonstrates the full pipeline:
1. Scrape content from any website using Decodo
2. Generate training data using OpenAI
3. Display and export results
"""

import streamlit as st
import asyncio
import json
from pathlib import Path
import time
from datetime import datetime
import os
import sys

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from training_data_bot.decodo import DecodoClient
from training_data_bot.ai import AIClient
from training_data_bot.core.models import TaskType
from training_data_bot.preprocessing import TextPreprocessor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="Web Scraper + AI Training Data Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Web Scraper + AI Training Data Generator</h1>
    <p>Scrape any website with Decodo and generate training data with OpenAI</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'scraped_content' not in st.session_state:
    st.session_state.scraped_content = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# API Status Check
st.sidebar.subheader("🔌 API Status")
openai_key = os.getenv("OPENAI_API_KEY")
decodo_auth = os.getenv("DECODO_BASIC_AUTH")

if openai_key:
    st.sidebar.success("✅ OpenAI API Key Found")
else:
    st.sidebar.error("❌ OpenAI API Key Missing")

if decodo_auth:
    st.sidebar.success("✅ Decodo Auth Found")
else:
    st.sidebar.error("❌ Decodo Auth Missing")

# Task Configuration
st.sidebar.subheader("🎯 Task Settings")
task_type = st.sidebar.selectbox(
    "Select Task Type:",
    ["QA Generation", "Summarization", "Classification"],
    help="Choose what type of training data to generate"
)

num_examples = st.sidebar.slider(
    "Number of Examples:",
    min_value=1,
    max_value=10,
    value=3,
    help="How many training examples to generate"
)

chunk_size = st.sidebar.slider(
    "Text Chunk Size:",
    min_value=200,
    max_value=2000,
    value=800,
    help="Size of text chunks for processing"
)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🌐 Step 1: Web Scraping")
    
    # URL input
    url = st.text_input(
        "Enter Website URL:",
        placeholder="https://example.com",
        help="Enter any website URL to scrape content from"
    )
    
    # Sample URLs for testing
    st.write("**Sample URLs to try:**")
    sample_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://blog.openai.com",
        "https://huggingface.co/blog"
    ]
    
    for sample_url in sample_urls:
        if st.button(f"📋 {sample_url}", key=f"sample_{sample_url}"):
            st.session_state.url = sample_url
            st.rerun()
    
    # Scraping options
    st.subheader("🔧 Scraping Options")
    device_type = st.selectbox("Device Type:", ["desktop", "mobile"])
    locale = st.selectbox("Locale:", ["en-us", "en-gb", "es-es", "fr-fr"])
    
    # Scrape button
    scrape_button = st.button("🚀 Start Scraping", type="primary")

with col2:
    st.header("🤖 Step 2: AI Generation")
    
    # Task-specific prompts
    if task_type == "QA Generation":
        prompt_template = st.text_area(
            "Custom Prompt (Optional):",
            value="Generate educational question-answer pairs from the following text. Create clear, specific questions with detailed answers.",
            height=100
        )
    elif task_type == "Summarization":
        prompt_template = st.text_area(
            "Custom Prompt (Optional):",
            value="Create a comprehensive summary of the following text, highlighting key points and main ideas.",
            height=100
        )
    else:  # Classification
        prompt_template = st.text_area(
            "Custom Prompt (Optional):",
            value="Classify the following text into relevant categories and provide reasoning for the classification.",
            height=100
        )
    
    # Generate button
    generate_button = st.button("🎯 Generate Training Data", type="primary")

# Processing section
if scrape_button and url:
    async def scrape_website():
        try:
            st.session_state.processing_status = "🔄 Scraping website..."
            
            # Initialize Decodo client
            decodo_client = DecodoClient()
            
            # Scrape the website
            result = await decodo_client.scrape_url(
                url=url,
                target="universal",
                locale=locale,
                device_type=device_type
            )
            
            # Extract text content (this might need adjustment based on Decodo's actual response format)
            content = result.get("content", result.get("text", str(result)))
            
            # Basic HTML cleaning (you might want to use BeautifulSoup for better cleaning)
            import re
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            st.session_state.scraped_content = clean_content
            st.session_state.processing_status = "✅ Scraping completed!"
            
            await decodo_client.close()
            
        except Exception as e:
            st.session_state.processing_status = f"❌ Scraping failed: {str(e)}"
    
    # Run the async function
    asyncio.run(scrape_website())

# Display scraping results
if st.session_state.processing_status:
    if "✅" in st.session_state.processing_status:
        st.markdown(f'<div class="success-box">{st.session_state.processing_status}</div>', unsafe_allow_html=True)
    elif "❌" in st.session_state.processing_status:
        st.markdown(f'<div class="error-box">{st.session_state.processing_status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">{st.session_state.processing_status}</div>', unsafe_allow_html=True)

if st.session_state.scraped_content:
    st.subheader("📄 Scraped Content Preview")
    content_preview = st.session_state.scraped_content[:1000] + "..." if len(st.session_state.scraped_content) > 1000 else st.session_state.scraped_content
    st.text_area("Scraped Content Preview", content_preview, height=200, disabled=True, label_visibility="collapsed")
    
    st.write(f"**Total content length:** {len(st.session_state.scraped_content)} characters")

# AI Generation section
if generate_button and st.session_state.scraped_content:
    async def generate_training_data():
        try:
            st.session_state.processing_status = "🤖 Generating training data with OpenAI..."
            
            # Initialize AI client
            ai_client = AIClient()
            
            # Initialize preprocessor for chunking
            preprocessor = TextPreprocessor()
            
            # Create a mock document for preprocessing
            from training_data_bot.core.models import Document, DocumentType
            mock_doc = Document(
                content=st.session_state.scraped_content,
                title=f"Scraped from {url}",
                source=url,
                doc_type=DocumentType.URL,
                size=len(st.session_state.scraped_content)
            )
            
            # Process document into chunks
            chunks = await preprocessor.process_document(mock_doc)
            
            # Limit chunks based on num_examples
            chunks_to_process = chunks[:num_examples]
            
            # Convert task type
            task_type_enum = {
                "QA Generation": TaskType.QA_GENERATION,
                "Summarization": TaskType.SUMMARIZATION,
                "Classification": TaskType.CLASSIFICATION
            }[task_type]
            
            # Generate training data
            generated_examples = []
            for i, chunk in enumerate(chunks_to_process):
                st.session_state.processing_status = f"🤖 Processing chunk {i+1}/{len(chunks_to_process)}..."
                
                response = await ai_client.process_text(
                    prompt=prompt_template,
                    input_text=chunk.content,
                    task_type=task_type_enum
                )
                
                example = {
                    "id": str(chunk.id),
                    "input": chunk.content,
                    "output": response.output,
                    "task_type": task_type.lower().replace(" ", "_"),
                    "confidence": response.confidence,
                    "token_usage": response.token_usage,
                    "cost": response.cost,
                    "processing_time": response.processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "source_url": url
                }
                
                generated_examples.append(example)
            
            st.session_state.generated_data = generated_examples
            st.session_state.processing_status = f"✅ Generated {len(generated_examples)} training examples!"
            
            await ai_client.close()
            
        except Exception as e:
            st.session_state.processing_status = f"❌ AI generation failed: {str(e)}"
    
    # Run the async function
    asyncio.run(generate_training_data())

# Display generation results
if st.session_state.generated_data:
    st.header("🎯 Generated Training Data")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Examples Generated", len(st.session_state.generated_data))
    with col2:
        total_tokens = sum(ex.get('token_usage', 0) for ex in st.session_state.generated_data)
        st.metric("Total Tokens", total_tokens)
    with col3:
        total_cost = sum(ex.get('cost', 0) for ex in st.session_state.generated_data)
        st.metric("Estimated Cost", f"${total_cost:.4f}")
    with col4:
        avg_confidence = sum(ex.get('confidence', 0) for ex in st.session_state.generated_data) / len(st.session_state.generated_data)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Display examples
    for i, example in enumerate(st.session_state.generated_data):
        with st.expander(f"📝 Example {i+1} - {example['task_type'].title()}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📖 Input")
                st.text_area("Input Text", example['input'], height=200, key=f"input_{i}", label_visibility="collapsed")
            
            with col2:
                st.subheader("🤖 Generated Output")
                st.text_area("Output Text", example['output'], height=200, key=f"output_{i}", label_visibility="collapsed")
            
            # Metadata
            st.write(f"**Confidence:** {example.get('confidence', 'N/A')}")
            st.write(f"**Tokens:** {example.get('token_usage', 'N/A')}")
            st.write(f"**Cost:** ${example.get('cost', 0):.4f}")
    
    # Export options
    st.header("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download JSONL"):
            jsonl_content = "\n".join(json.dumps(ex) for ex in st.session_state.generated_data)
            st.download_button(
                label="💾 Download JSONL File",
                data=jsonl_content,
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                mime="application/jsonlines"
            )
    
    with col2:
        if st.button("📊 Download JSON"):
            json_content = json.dumps(st.session_state.generated_data, indent=2)
            st.download_button(
                label="💾 Download JSON File",
                data=json_content,
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Clear Results"):
            st.session_state.generated_data = []
            st.session_state.scraped_content = None
            st.session_state.processing_status = ""
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 Powered by <strong>Decodo</strong> (Web Scraping) + <strong>OpenAI</strong> (AI Generation)</p>
    <p>Built with ❤️ using <strong>Streamlit</strong> and <strong>Training Data Bot</strong></p>
</div>
""", unsafe_allow_html=True) 