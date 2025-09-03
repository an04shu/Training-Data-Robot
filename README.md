# 📘 TrainingDataBot

🚀 **Enterprise-Grade Data Preparation for LLMs**  
Forget toy datasets and one-off scripts. **TrainingDataBot** is your end-to-end system for:

- 📂 **Loading** documents (PDF, TXT, URLs, etc.)  
- ⚙️ **Processing** them with task templates (Q&A, summarization, classification)  
- ✅ **Evaluating** quality and filtering bad examples  
- 📦 **Exporting** curated datasets in standard formats (JSONL, CSV, etc.)  

Think of it as your **data factory** for LLM training.  

---

## 🌟 Why This Project?

Most ML repos stop at “I trained a model with X% accuracy.”  
This one goes further. It mimics **real enterprise workflows**:

- Clear **entry point** (`__init__.py`) like a factory gate.  
- **Pipeline orchestration** (`TrainingDataBot`) acting as the factory manager.  
- Multiple **workers** (loaders, preprocessors, task managers, evaluators).  
- **Job tracking** with status updates (PROCESSING ✅, FAILED ❌).  
- **Convenience methods** (like `quick_process`) for one-liner dataset creation.  

💡 It’s not just a script — it’s a **system**. The kind you’d showcase in interviews or production.

---

## 🏗️ Architecture Overview

```bash
training_data_bot/
├── core/            # configs, logging, exceptions, data models
├── sources/         # loaders for PDFs, web pages, text files
├── tasks/           # task templates (Q&A, summarization, classification)
├── preprocessing/   # text cleaning & chunking
├── evaluation/      # quality evaluation module
├── storage/         # dataset exporter + database manager
└── bot.py           # TrainingDataBot manager (heart of the system)


🔑 Key abstractions:
- **Documents** → inputs (raw material)  
- **Jobs** → processing runs (task board)  
- **Datasets** → final training data (finished goods)  

---

## ⚡ Quick Start

```bash
# clone repo
git clone https://github.com/yourname/training-data-bot.git
cd training-data-bot

# install dependencies
pip install -r requirements.txt

