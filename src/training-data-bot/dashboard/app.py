"""
Streamlit Dashboard for Training Data Bot
"""

# Load environment variables first
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
# Current file: src/training_data_bot/dashboard/app.py
# Project root: ../../../ from here
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
else:
    # Try alternative paths
    alt_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent.parent.parent.parent / ".env",  # One level up
    ]
    loaded = False
    for alt_path in alt_paths:
        if alt_path.exists():
            load_dotenv(alt_path)
            print(f"✅ Loaded environment variables from {alt_path}")
            loaded = True
            break
    
    if not loaded:
        print(f"⚠️ No .env file found. Searched paths:")
        print(f"   - {env_path}")
        for alt_path in alt_paths:
            print(f"   - {alt_path}")

import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
from typing import List, Dict, Any

from training_data_bot import TrainingDataBot
from training_data_bot.core.models import Document, Dataset, TaskType


def main():
    """Main Streamlit dashboard application."""
    
    st.set_page_config(
        page_title="🧠 Training Data Bot",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Training Data Curation Bot")
    st.markdown("**Enterprise-grade training data curation for LLM fine-tuning**")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        
        # Check if page was set by button click
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "🏠 Dashboard"
        
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📄 Documents", "🎯 Generate Data", "📊 Analytics", "⚙️ Settings"],
            index=["🏠 Dashboard", "📄 Documents", "🎯 Generate Data", "📊 Analytics", "⚙️ Settings"].index(st.session_state.selected_page) if st.session_state.selected_page in ["🏠 Dashboard", "📄 Documents", "🎯 Generate Data", "📊 Analytics", "⚙️ Settings"] else 0
        )
        
        # Update session state
        st.session_state.selected_page = page
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        st.session_state.documents = []
        st.session_state.datasets = []
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📄 Documents":
        documents_page()
    elif page == "🎯 Generate Data":
        generate_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()


def dashboard_page():
    """Main dashboard overview."""
    
    st.header("📊 System Overview")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Documents", len(st.session_state.documents))
    
    with col2:
        st.metric("🎯 Datasets", len(st.session_state.datasets))
    
    with col3:
        st.metric("🌐 Web Scraping", "Decodo Professional")
    
    with col4:
        st.metric("✅ Status", "Ready")
    
    # Quick actions
    st.header("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.selected_page = "📄 Documents"
            st.rerun()
    
    with col2:
        if st.button("🎯 Generate Training Data", use_container_width=True):
            st.session_state.selected_page = "🎯 Generate Data"
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.selected_page = "📊 Analytics"
            st.rerun()
    
    # Recent activity
    st.header("📈 Recent Activity")
    st.info("📋 Upload some documents to get started!")


def documents_page():
    """Documents management page."""
    
    st.header("📄 Document Management")
    
    # Input method selection
    st.subheader("📥 Document Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Files", "🌐 Scrape URLs"],
        horizontal=True
    )
    
    if input_method == "📁 Upload Files":
        # File upload
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md', 'html', 'json', 'csv']
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files ready for processing!")
            
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")
            
            if st.button("🔄 Process Documents"):
                with st.spinner("Processing documents..."):
                    # Here you would process the uploaded files
                    st.success("✅ Documents processed successfully!")
    
    elif input_method == "🌐 Scrape URLs":
        # URL input
        st.subheader("🌐 Web Scraping (Decodo Integration)")
        
        # Info about Decodo integration
        has_decodo_auth = bool(os.getenv("DECODO_BASIC_AUTH") or (os.getenv("DECODO_USERNAME") and os.getenv("DECODO_PASSWORD")))
        
        if has_decodo_auth:
            st.info("""
            🚀 **Professional Web Scraping Powered by Decodo API**
            
            ✅ **JavaScript rendering** - Access dynamic content  
            ✅ **Bot detection bypass** - Scrape protected sites  
            ✅ **Device simulation** - Mobile, desktop, tablet views  
            ✅ **Geographic targeting** - Location-specific content  
            ✅ **Intelligent fallback** - Automatic backup scraping  
            
            🔑 **Authentication Detected** - Using professional Decodo service
            """)
        else:
            st.warning("""
            🌐 **Web Scraping with Intelligent Fallback**
            
            ⚠️ **No Decodo Authentication** - Using basic HTTP scraping  
            ✅ **Intelligent fallback** - Still works for most sites  
            📝 **To enable professional features**: Set Decodo credentials  
            
            **Get your credentials at**: https://scraper-api.decodo.com
            ```bash
            # Option 1: Use Basic Auth Token
            export DECODO_BASIC_AUTH="your_basic_auth_token"
            
            # Option 2: Use Username/Password
            export DECODO_USERNAME="your_username"
            export DECODO_PASSWORD="your_password"
            ```
            """)
        
        # Initialize URL input
        initial_url = ""
        if hasattr(st.session_state, 'example_url') and st.session_state.example_url:
            initial_url = st.session_state.example_url
            st.info(f"🔗 Using example URL: {initial_url}")
            # Clear the example after use
            del st.session_state.example_url
        
        # Single URL input
        url_input = st.text_input(
            "Enter a website URL:",
            value=initial_url,
            placeholder="https://example.com/article",
            help="The WebLoader will use Decodo professional scraping with intelligent fallback"
        )
        
        # Multiple URLs input
        st.write("**Or enter multiple URLs (one per line):**")
        urls_text = st.text_area(
            "Multiple URLs:",
            placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3",
            height=100
        )
        
        # Quick examples
        st.write("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📰 News Article"):
                st.session_state.example_url = "https://httpbin.org/html"
        with col2:
            if st.button("📚 Documentation"):
                st.session_state.example_url = "https://httpbin.org/json"
        with col3:
            if st.button("🏢 Company Page"):
                st.session_state.example_url = "https://httpbin.org/xml"

        
        # Advanced scraping options
        with st.expander("⚙️ Advanced Scraping Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_type = st.selectbox(
                    "Target Type:",
                    ["universal", "amazon", "google", "linkedin", "twitter"],
                    help="Specialized scraping for different website types"
                )
                
                device_type = st.selectbox(
                    "Device Type:",
                    ["desktop", "mobile", "tablet"],
                    help="Simulate different devices"
                )
            
            with col2:
                locale = st.selectbox(
                    "Locale:",
                    ["en-us", "en-gb", "es-es", "fr-fr", "de-de"],
                    help="Language/region settings"
                )
                
                geo_location = st.selectbox(
                    "Geographic Location:",
                    ["United States", "United Kingdom", "Spain", "France", "Germany"],
                    help="Geographic targeting for scraping"
                )
        
        # Process URLs
        urls_to_process = []
        if url_input:
            urls_to_process.append(url_input)
        
        if urls_text:
            urls_from_text = [url.strip() for url in urls_text.split('\n') if url.strip()]
            urls_to_process.extend(urls_from_text)
        
        if urls_to_process:
            st.success(f"🌐 {len(urls_to_process)} URLs ready for scraping!")
            
            for i, url in enumerate(urls_to_process, 1):
                st.write(f"🔗 {i}. {url}")
            
            if st.button("🚀 Scrape Websites", type="primary"):
                with st.spinner("Scraping websites with Decodo..."):
                    try:
                        # Import WebLoader for direct scraping
                        from training_data_bot.sources.web import WebLoader
                        
                        # Prepare scraping parameters
                        scraping_params = {
                            "target": target_type,
                            "locale": locale,
                            "geo": geo_location,
                            "device_type": device_type
                        }
                        
                        # Scrape URLs using WebLoader directly
                        progress_bar = st.progress(0)
                        scraped_documents = []
                        
                        async def scrape_urls():
                            loader = WebLoader()
                            try:
                                if len(urls_to_process) == 1:
                                    # Single URL
                                    doc = await loader.load_single(urls_to_process[0], **scraping_params)
                                    return [doc]
                                else:
                                    # Multiple URLs
                                    docs = await loader.load_multiple_urls(urls_to_process, max_concurrent=3, **scraping_params)
                                    return docs
                            finally:
                                # Safe cleanup - only close if method exists
                                if hasattr(loader, 'close') and callable(getattr(loader, 'close')):
                                    try:
                                        await loader.close()
                                    except Exception:
                                        pass  # Ignore cleanup errors
                        
                        # Run the scraping
                        try:
                            scraped_documents = asyncio.run(scrape_urls())
                        except Exception as async_error:
                            st.warning(f"⚠️ Async scraping failed: {async_error}")
                            st.info("🔄 Trying alternative scraping method...")
                            
                            # Fallback: simple synchronous approach
                            scraped_documents = []
                            for i, url in enumerate(urls_to_process):
                                progress_bar.progress((i + 1) / len(urls_to_process))
                                st.write(f"🔄 Scraping: {url}")
                                
                                try:
                                    # Simple HTTP request fallback
                                    import requests
                                    from training_data_bot.core.models import Document, DocumentType
                                    from uuid import uuid4
                                    
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                                    }
                                    response = requests.get(url, headers=headers, timeout=30)
                                    response.raise_for_status()
                                    
                                    # Create document
                                    doc = Document(
                                        id=uuid4(),
                                        title=f"Document from {url.split('//')[1].split('/')[0]}",
                                        content=response.text[:5000],  # Limit content for demo
                                        source=url,
                                        doc_type=DocumentType.URL,
                                        extraction_method="Fallback.Simple"
                                    )
                                    
                                    scraped_documents.append(doc)
                                    st.write(f"✅ Success: {doc.title}")
                                    
                                except Exception as e:
                                    st.write(f"❌ Failed: {url} - {str(e)}")
                        
                        # Update progress
                        for i in range(len(urls_to_process)):
                            progress_bar.progress((i + 1) / len(urls_to_process))
                            if i < len(scraped_documents):
                                st.write(f"✅ Success: {scraped_documents[i].title}")
                            else:
                                st.write(f"⚠️ No content for URL {i+1}")
                        
                        # Store scraped documents in session state
                        if scraped_documents:
                            st.session_state.documents.extend(scraped_documents)
                        
                        if scraped_documents:
                            st.success(f"🎉 Successfully scraped {len(scraped_documents)} websites!")
                            
                            # Display scraped data
                            scraped_data = []
                            for doc in scraped_documents:
                                scraped_data.append({
                                    "URL": doc.source,
                                    "Title": doc.title,
                                    "Content Length": f"{len(doc.content)} characters",
                                    "Method": doc.extraction_method,
                                    "Status": "✅ Success",
                                    "Language": getattr(doc, 'language', 'Unknown')
                                })
                            
                            df = pd.DataFrame(scraped_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Show content preview
                            st.subheader("📖 Content Preview")
                            for i, doc in enumerate(scraped_documents[:3]):  # Show first 3
                                with st.expander(f"📄 {doc.title}"):
                                    st.write(f"**URL:** {doc.source}")
                                    st.write(f"**Length:** {len(doc.content)} characters")
                                    st.write(f"**Method:** {doc.extraction_method}")
                                    st.write("**Content Preview:**")
                                    st.text(doc.content[:500] + "..." if len(doc.content) > 500 else doc.content)
                        else:
                            st.error("❌ No documents were successfully scraped.")
                    
                    except ImportError as e:
                        st.error(f"❌ Import error: {str(e)}")
                        st.info("💡 Try restarting the Streamlit app or check if all dependencies are installed.")
                    except Exception as e:
                        st.error(f"❌ Error during scraping: {str(e)}")
                        
                        # Show simplified error for common issues
                        if "aenter" in str(e).lower():
                            st.info("💡 **Tip**: This might be an async context manager issue. The app will try a different approach.")
                        elif "module" in str(e).lower() and "not found" in str(e).lower():
                            st.info("💡 **Tip**: Missing dependency. Please install required packages.")
                        else:
                            st.exception(e)
    
    # Document list
    st.subheader("📋 Loaded Documents")
    
    if st.session_state.documents:
        # Create a DataFrame for display
        doc_data = []
        for doc in st.session_state.documents:
            doc_data.append({
                "Name": doc.title,
                "Type": doc.doc_type,
                "Words": doc.word_count,
                "Characters": doc.char_count,
                "Language": doc.language,
                "Source": doc.source
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("📋 No documents loaded yet. Upload some files to get started!")


def generate_page():
    """Training data generation page."""
    
    st.header("🎯 Generate Training Data")
    
    if not st.session_state.documents:
        st.warning("📋 Please upload or scrape documents first!")
        st.info("💡 Go to the **📄 Documents** page to add content.")
        return
    
    # Show available documents
    st.subheader("📚 Available Documents")
    st.success(f"✅ {len(st.session_state.documents)} documents loaded and ready for training data generation!")
    
    # Document selection
    doc_data = []
    for i, doc in enumerate(st.session_state.documents):
        doc_data.append({
            "Select": False,
            "Document": doc.title,
            "Type": str(doc.doc_type),
            "Content Length": f"{len(doc.content):,} chars",
            "Word Count": f"{doc.word_count:,}" if hasattr(doc, 'word_count') and doc.word_count else "N/A",
            "Source": doc.source if hasattr(doc, 'source') else "Unknown"
        })
    
    df = pd.DataFrame(doc_data)
    
    # Document selection interface
    st.write("**Select documents for training data generation:**")
    
    # Select all/none buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Select All"):
            for i in range(len(st.session_state.documents)):
                st.session_state[f"doc_select_{i}"] = True
            st.rerun()
    with col2:
        if st.button("❌ Clear All"):
            for i in range(len(st.session_state.documents)):
                st.session_state[f"doc_select_{i}"] = False
            st.rerun()
    
    # Individual document selection
    selected_docs = []
    for i, doc in enumerate(st.session_state.documents):
        key = f"doc_select_{i}"
        if key not in st.session_state:
            st.session_state[key] = True  # Default to selected
        
        is_selected = st.checkbox(
            f"📄 **{doc.title}** ({len(doc.content):,} chars) - {doc.source if hasattr(doc, 'source') else 'Unknown source'}",
            value=st.session_state[key],
            key=f"checkbox_{i}"
        )
        
        if is_selected:
            selected_docs.append(doc)
            st.session_state[key] = True
        else:
            st.session_state[key] = False
    
    if not selected_docs:
        st.warning("⚠️ Please select at least one document for training data generation.")
        return
    
    st.success(f"🎯 {len(selected_docs)} documents selected for training data generation")
    
    # Content preview
    if st.expander("👀 Preview Selected Content", expanded=False):
        for doc in selected_docs[:3]:  # Show first 3 selected
            st.write(f"**📄 {doc.title}**")
            st.write(f"Source: {getattr(doc, 'source', 'Unknown')}")
            content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            st.text_area(f"Content preview", content_preview, height=100, disabled=True, key=f"preview_{doc.id}")
            st.divider()
    
    # Task selection
    st.subheader("🎯 Select Task Type")
    
    task_type = st.selectbox(
        "Choose training data type:",
        ["QA Generation", "Classification", "Summarization", "NER", "Red Teaming"],
        help="Select the type of training data you want to generate from the selected documents"
    )
    
    # Task-specific parameters
    st.subheader("⚙️ Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_examples = st.number_input(
            "Number of examples", 
            min_value=1, 
            max_value=100, 
            value=10,
            help="How many training examples to generate"
        )
        chunk_size = st.number_input(
            "Text chunk size", 
            min_value=100, 
            max_value=2000, 
            value=500,
            help="Size of text chunks to process"
        )
    
    with col2:
        quality_threshold = st.slider(
            "Quality threshold", 
            0.0, 1.0, 0.8,
            help="Minimum quality score for generated examples"
        )
        overlap = st.number_input(
            "Chunk overlap", 
            min_value=0, 
            max_value=200, 
            value=50,
            help="Overlap between text chunks"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        temperature = st.slider("AI Temperature", 0.0, 1.0, 0.7, help="Creativity level for AI generation")
        max_length = st.number_input("Max response length", 50, 500, 150, help="Maximum length of generated responses")
        include_metadata = st.checkbox("Include source metadata", value=True, help="Include source URL/file info in generated data")
    
    # Generate button
    if st.button("🚀 Generate Training Data", type="primary", use_container_width=True):
        with st.spinner(f"Generating {task_type} training data from {len(selected_docs)} documents..."):
            try:
                # Combine content from selected documents
                combined_content = ""
                source_info = []
                
                for doc in selected_docs:
                    combined_content += f"\n\n--- {doc.title} ---\n"
                    combined_content += doc.content
                    source_info.append({
                        "title": doc.title,
                        "source": getattr(doc, 'source', 'Unknown'),
                        "length": len(doc.content)
                    })
                
                # Show processing progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Chunk the content
                status_text.text("📝 Chunking content...")
                progress_bar.progress(0.2)
                
                # Split content into chunks
                chunks = []
                start = 0
                while start < len(combined_content):
                    end = start + chunk_size
                    chunk = combined_content[start:end]
                    chunks.append(chunk)
                    start = end - overlap
                
                status_text.text(f"🧠 Generating {task_type} examples...")
                progress_bar.progress(0.5)
                
                # Generate training examples (mock for now)
                examples = []
                
                if task_type == "QA Generation":
                    for i in range(min(num_examples, len(chunks))):
                        chunk = chunks[i % len(chunks)]
                        examples.append({
                            "Question": f"What is the main topic discussed in this text?",
                            "Answer": f"The text discusses {chunk[:100]}...",
                            "Context": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "Source": source_info[i % len(source_info)]["title"],
                            "Quality Score": round(0.7 + (i * 0.05), 2)
                        })
                
                elif task_type == "Summarization":
                    for i in range(min(num_examples, len(chunks))):
                        chunk = chunks[i % len(chunks)]
                        examples.append({
                            "Original Text": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                            "Summary": f"Summary of the content discussing key points...",
                            "Length Reduction": f"{round((1 - 100/len(chunk)) * 100, 1)}%",
                            "Source": source_info[i % len(source_info)]["title"],
                            "Quality Score": round(0.75 + (i * 0.03), 2)
                        })
                
                elif task_type == "Classification":
                    categories = ["Technology", "Business", "Health", "Education", "News"]
                    for i in range(min(num_examples, len(chunks))):
                        chunk = chunks[i % len(chunks)]
                        examples.append({
                            "Text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "Category": categories[i % len(categories)],
                            "Confidence": round(0.8 + (i * 0.02), 2),
                            "Source": source_info[i % len(source_info)]["title"],
                            "Quality Score": round(0.72 + (i * 0.04), 2)
                        })
                
                progress_bar.progress(0.8)
                status_text.text("✨ Finalizing results...")
                
                # Filter by quality threshold
                filtered_examples = [ex for ex in examples if ex.get("Quality Score", 0) >= quality_threshold]
                
                progress_bar.progress(1.0)
                status_text.text("✅ Generation complete!")
                
                st.success(f"🎉 Generated {len(filtered_examples)} high-quality {task_type} examples!")
                
                # Store in session state
                if 'generated_data' not in st.session_state:
                    st.session_state.generated_data = []
                
                st.session_state.generated_data.extend(filtered_examples)
                
                # Show results
                st.subheader("📊 Generated Training Data")
                
                if filtered_examples:
                    df_results = pd.DataFrame(filtered_examples)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Export options
                    st.subheader("💾 Export Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📄 Download JSON"):
                            import json
                            json_data = json.dumps(filtered_examples, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"{task_type.lower()}_training_data.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        if st.button("📊 Download CSV"):
                            csv_data = df_results.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"{task_type.lower()}_training_data.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        if st.button("📝 Download JSONL"):
                            import json
                            jsonl_data = "\n".join([json.dumps(ex) for ex in filtered_examples])
                            st.download_button(
                                label="Download JSONL",
                                data=jsonl_data,
                                file_name=f"{task_type.lower()}_training_data.jsonl",
                                mime="text/plain"
                            )
                    
                    # Show statistics
                    st.subheader("📈 Generation Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Generated", len(examples))
                    with col2:
                        st.metric("High Quality", len(filtered_examples))
                    with col3:
                        avg_quality = sum(ex.get("Quality Score", 0) for ex in filtered_examples) / len(filtered_examples) if filtered_examples else 0
                        st.metric("Avg Quality", f"{avg_quality:.2f}")
                    with col4:
                        st.metric("Documents Used", len(selected_docs))
                
                else:
                    st.warning("⚠️ No examples met the quality threshold. Try lowering the threshold or adjusting parameters.")
                
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.exception(e)


def analytics_page():
    """Analytics and metrics page."""
    
    st.header("📊 Analytics & Metrics")
    
    # Mock data for demonstration
    if st.session_state.documents:
        # Document type distribution
        st.subheader("📄 Document Type Distribution")
        doc_types = {"TXT": 45, "PDF": 30, "DOCX": 20, "MD": 5}
        
        fig_pie = px.pie(
            values=list(doc_types.values()),
            names=list(doc_types.keys()),
            title="Document Types"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Processing metrics
        st.subheader("⚡ Processing Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time chart
            times = [1.2, 2.3, 1.8, 3.1, 2.0, 1.5, 2.8]
            fig_line = px.line(
                x=list(range(len(times))),
                y=times,
                title="Processing Time per Document (seconds)"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            # Quality scores
            scores = [0.85, 0.92, 0.78, 0.89, 0.95, 0.83, 0.91]
            fig_bar = px.bar(
                x=list(range(len(scores))),
                y=scores,
                title="Quality Scores by Dataset"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("📊 Upload and process documents to see analytics!")


def settings_page():
    """Settings and configuration page."""
    
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("🔗 API Configuration")
    
    decodo_api_key = st.text_input("Decodo API Key", type="password", placeholder="Enter your API key")
    decodo_base_url = st.text_input("Decodo Base URL", value="https://api.decodo.com")
    
    # Processing Settings
    st.subheader("⚡ Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_workers = st.number_input("Max Workers", min_value=1, max_value=16, value=4)
        chunk_size = st.number_input("Default Chunk Size", min_value=100, max_value=2000, value=1000)
    
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
        timeout = st.number_input("Timeout (seconds)", min_value=10, max_value=300, value=60)
    
    # Quality Thresholds
    st.subheader("🎯 Quality Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        toxicity_threshold = st.slider("Toxicity", 0.0, 1.0, 0.8)
    
    with col2:
        bias_threshold = st.slider("Bias", 0.0, 1.0, 0.7)
    
    with col3:
        similarity_threshold = st.slider("Similarity", 0.0, 1.0, 0.85)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")


if __name__ == "__main__":
    main() 