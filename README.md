# 🧠 Training Data Curation Bot

**Enterprise-grade training data curation for LLM fine-tuning with professional web scraping**

## 🌟 Overview

The Training Data Curation Bot is a comprehensive system that transforms raw content into high-quality training datasets for machine learning models. With professional web scraping capabilities, AI-powered content generation, and enterprise-grade quality control, it provides everything you need to create world-class training data.

## ✨ Key Features

### 🌐 Professional Web Scraping
- **Decodo Integration**: Enterprise-grade web scraping with professional authentication
- **Smart Fallback System**: Automatic fallback to ensure content is always retrieved
- **Multiple Target Types**: Optimized scraping for different content types (news, e-commerce, etc.)
- **Global Support**: Multi-locale and geographic targeting
- **Parallel Processing**: High-performance concurrent scraping

### 🎯 AI-Powered Content Generation
- **Multiple AI Providers**: OpenAI GPT-4, GPT-3.5, Claude integration
- **Diverse Task Types**: Q&A generation, classification, summarization, NER, red teaming
- **Quality Validation**: Automated quality scoring and filtering
- **Cost Tracking**: Real-time cost monitoring and budget management
- **Professional Templates**: Optimized prompts for different use cases

### 📊 Enterprise Dashboard
- **Dual Input Methods**: File upload and URL scraping in one interface
- **Real-time Progress**: Live updates during processing
- **Interactive Analytics**: Professional charts and metrics
- **Quality Metrics**: Comprehensive quality scoring and validation
- **Export Options**: Multiple formats (JSONL, CSV, HuggingFace)

### 🔍 Quality Control
- **Multi-metric Evaluation**: Toxicity, bias, diversity, coherence, relevance
- **Automated Filtering**: Quality threshold enforcement
- **Detailed Reporting**: Comprehensive quality reports
- **Error Handling**: Robust error recovery and logging

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Decodo Professional Web Scraping
DECODO_USERNAME=your_decodo_username
DECODO_PASSWORD=your_decodo_password
DECODO_BASIC_AUTH=your_encoded_basic_auth_token
```

### 3. Launch the Dashboard
```bash
streamlit run src/training_data_bot/dashboard/app.py
```

### 4. Use the Command Line Interface
```bash
# Process documents
tdb process --source-dir ./documents --output-dir ./results

# Generate specific task types
tdb generate qa --input-file document.txt --output-file qa_results.jsonl

# Evaluate quality
tdb evaluate --dataset-file results.jsonl --output-report quality_report.html
```

## 🎯 Supported Content Types

### 📁 File Types
- **PDF**: Textbooks, research papers, documentation
- **DOCX**: Word documents, reports, articles
- **TXT**: Plain text files, notes, stories
- **MD**: Markdown files, documentation
- **HTML**: Web pages, articles
- **CSV**: Spreadsheets, data tables
- **JSON**: Structured data files

### 🌐 Web Content
- **Any Website**: Wikipedia, news sites, blogs, documentation
- **E-commerce**: Product pages, reviews
- **Social Media**: Posts, profiles (where accessible)
- **News Articles**: Breaking news, analysis
- **Technical Documentation**: API docs, tutorials
- **Academic Papers**: Research articles, preprints

## 🎨 Training Data Types

### ❓ Question & Answer Generation
- **Reading Comprehension**: Q&A pairs from any text
- **Factual Questions**: Knowledge-based questions
- **Analytical Questions**: Critical thinking prompts
- **Multi-choice Questions**: Structured assessments

### 🏷️ Classification
- **Topic Classification**: Categorize content by subject
- **Sentiment Analysis**: Positive, negative, neutral
- **Content Type**: Article, blog, news, academic
- **Difficulty Level**: Easy, medium, hard

### 📝 Summarization
- **Extractive Summaries**: Key sentence extraction
- **Abstractive Summaries**: Paraphrased summaries
- **Multi-length**: Short, medium, long summaries
- **Structured Summaries**: Bullet points, key themes

### 🔍 Named Entity Recognition
- **Person Names**: Identify individuals
- **Organizations**: Companies, institutions
- **Locations**: Cities, countries, landmarks
- **Dates**: Time references, events

### 🛡️ Red Team Testing
- **Safety Testing**: Identify potential risks
- **Bias Detection**: Uncover hidden biases
- **Adversarial Prompts**: Test model robustness
- **Edge Case Discovery**: Find failure modes

## 📊 Analytics & Reporting

### 📈 Quality Metrics
- **Toxicity Score**: Harmful content detection
- **Bias Score**: Fairness assessment
- **Diversity Score**: Content variety
- **Coherence Score**: Logical consistency
- **Relevance Score**: Topic alignment

### 💰 Cost Management
- **Real-time Tracking**: Monitor AI usage costs
- **Budget Controls**: Set spending limits
- **Provider Comparison**: Optimize cost/quality
- **Usage Reports**: Detailed spending analysis

### 📊 Performance Analytics
- **Generation Stats**: Success rates, processing time
- **Quality Trends**: Performance over time
- **Task Breakdown**: Results by task type
- **Export Analytics**: Download comprehensive reports

## 🏗️ Architecture Overview

```
Training Data Bot Architecture
├── 🌐 Web Scraping Layer (Decodo + Fallback)
├── 📄 Document Processing Pipeline
├── 🔄 Text Preprocessing & Chunking
├── 🎯 Task Management System
├── 🤖 AI Generation Workshops
├── 🔍 Quality Control Pipeline
├── 📊 Storage & Export System
└── 🖥️ Web Dashboard Interface
```

## 📋 Step-by-Step Learning Guide

1. **[Step 3: Core Data Models](step3_core_data_models.md)** - Understanding the data architecture
2. **[Step 4: Document Loading Pipeline](step4_document_loading_pipeline.md)** - How content enters the system
3. **[Step 5: Specialized Document Loaders](step5_specialized_document_loaders.md)** - File type handlers
4. **[Step 6: Text Preprocessing Pipeline](step6_text_preprocessing_pipeline.md)** - Content preparation
5. **[Step 7: Task Management System](step7_task_management_system.md)** - Job coordination
6. **[Step 8: AI Task Generation](step8_ai_task_generation.md)** - AI-powered content creation
7. **[Step 9: AI Client Brain](step9_ai_client_brain.md)** - Central AI intelligence
8. **[Step 10: Quality Control](step10_quality_control.md)** - Quality assurance
9. **[Step 11: Storage & Export](step11_storage_export.md)** - Data packaging
10. **[Step 12: Web Scraping with Decodo](step12_web_scraping_decodo.md)** - Professional web scraping
11. **[Step 13: Command Line Interface](step13_command_line_interface.md)** - CLI usage
12. **[Step 14: Web Dashboard](step14_web_dashboard.md)** - Visual interface

## 🔧 Configuration

### Environment Variables
```env
# Required: OpenAI API Key
OPENAI_API_KEY=sk-...

# Optional: Decodo Professional Web Scraping
DECODO_USERNAME=your_username
DECODO_PASSWORD=your_password
DECODO_BASIC_AUTH=encoded_auth_token

# Optional: Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# Optional: System Configuration
MAX_WORKERS=8
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
QUALITY_THRESHOLD=0.7
```

### Configuration File
```yaml
# configs/config.yaml
processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_workers: 8

quality:
  threshold: 0.7
  metrics:
    - toxicity
    - bias
    - diversity
    - coherence
    - relevance

ai:
  preferred_provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 1000
```

## 🎯 Use Cases

### 📚 Educational Content
- **Textbook Processing**: Convert textbooks to Q&A datasets
- **Assessment Creation**: Generate quiz questions
- **Study Materials**: Create summaries and flashcards
- **Curriculum Development**: Build comprehensive learning materials

### 🏢 Enterprise Applications
- **Knowledge Base**: Transform documentation into searchable Q&A
- **Training Materials**: Create employee training content
- **Content Analysis**: Classify and categorize large document collections
- **Compliance**: Generate training data for regulatory content

### 🔬 Research & Development
- **Literature Review**: Summarize research papers
- **Data Annotation**: Create labeled datasets
- **Model Training**: Generate synthetic training data
- **Bias Testing**: Create diverse test sets

## 🛠️ Advanced Features

### 🔄 Batch Processing
```python
# Process multiple sources
sources = [
    "https://wikipedia.org/wiki/AI",
    "documents/research.pdf",
    "https://news.example.com/article"
]

# Generate training data
dataset = await bot.process_sources(
    sources=sources,
    task_types=["qa_generation", "classification"],
    quality_threshold=0.8
)
```

### 🎯 Custom Task Types
```python
# Define custom task
custom_task = TaskTemplate(
    name="Custom Question Generator",
    task_type=TaskType.QA_GENERATION,
    prompt_template="""
    Create domain-specific questions from: {text}
    Focus on: {domain}
    Difficulty: {difficulty}
    """,
    parameters={"domain": "machine_learning", "difficulty": "advanced"}
)
```

### 📊 Export Formats
- **JSONL**: Machine learning standard format
- **CSV**: Spreadsheet-compatible format
- **HuggingFace**: Direct integration with HuggingFace datasets
- **Custom**: Define your own export format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Anthropic** for Claude models
- **Decodo** for professional web scraping
- **Streamlit** for the dashboard framework
- **HuggingFace** for ML ecosystem integration

---

**Transform any content into professional training data with the power of AI!** 🚀 