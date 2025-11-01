# 🚀 QuickStart Guide - MariaDB RAG System

**Get your RAG (Retrieval-Augmented Generation) system running in 10 minutes!**

This guide will help you set up and run the complete RAG demonstration using MariaDB Vector, HuggingFace embeddings, and Ollama LLM.

## 📋 Prerequisites

Before starting, ensure you have:

- **Docker** installed and running
- **Python 3.8+** installed  
- **Git** for cloning (if needed)
- **4GB+ RAM** available
- **Internet connection** for model downloads

## ⚡ Quick Setup (5 Steps)

### Step 1: Start MariaDB with Vector Support

```bash
# Start MariaDB container with Vector extension
docker run -d \
  --name mariadb-rag \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=demo123 \
  -e MYSQL_DATABASE=rag_demo \
  mariadb:11.8

# Wait for MariaDB to start (30 seconds)
echo "Waiting for MariaDB to start..."
sleep 30

# Verify it's running
docker ps | grep mariadb-rag
```

### Step 2: Set Up Python Environment

```bash
# Navigate to project directory
cd /path/to/mariadb-rag-project

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Load Sample Data

```bash
# Set up database with sample data
python demo/setup_database.py

# Load movie dataset
python demo/verify_sample_data.py
```

Expected output:

```text
✅ Connected to MariaDB successfully
✅ Database 'rag_demo' exists
✅ Sample data loaded: 20 movies
✅ Vector extension is available
```

### Step 4: Install and Start Ollama

**On macOS/Linux:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve &

# Pull a language model (this may take a few minutes)
ollama pull llama2

# Verify Ollama is working
curl http://localhost:11434/api/tags
```

**On Windows:**

1. Download Ollama from <https://ollama.ai/download>
2. Install and run the application
3. Open Command Prompt and run: `ollama pull llama2`

### Step 5: Start Jupyter and Run Demo

```bash
# Start Jupyter notebook
jupyter notebook demo/demo_notebook.ipynb
```

Your browser will open with the demo notebook. Now you can run the cells!

## 🎯 What You'll See

The demo notebook contains:

1. **Magic Command Loading** - Load the custom RAG magic commands
2. **Vector Indexing** - Create embeddings for movie data
3. **Semantic Search** - Find similar content using natural language
4. **RAG Queries** - Ask questions and get AI-powered answers

## 🧪 Quick Test

Once everything is running, try these commands in the notebook:

```python
# Load magic commands
%load_ext mariadb_rag_magics

# Index movie data with embeddings
%vector_index movies description

# Search for similar movies
%semantic_search movies "space adventure"

# Ask AI questions about your data (default Ollama)
%%rag_query movies
What are some good sci-fi movies in the database?

# Try with HuggingFace models
%%rag_query movies --llm huggingface
What are some good sci-fi movies in the database?

# Combine parameters for fine control
%%rag_query movies --top_k 5 --llm ollama
What are the best action movies for a family?
```

## 🔧 Configuration

The system uses these default configurations (in `config.py`):

- **Database**: `127.0.0.1:3306` (MariaDB container)
- **Models**: `all-MiniLM-L6-v2` (384-dim embeddings)
- **Default LLM**: `ollama` via Ollama (localhost:11434)
- **Available LLMs**: `ollama`, `huggingface`

## 📊 Sample Data

The demo includes:

- **20 movie records** with titles, descriptions, and genres
- **30 airport records** (optional dataset)
- **Vector embeddings** for semantic search

## 🚨 Common Issues & Solutions

### Issue: MariaDB Connection Failed

```bash
# Check if container is running
docker ps | grep mariadb-rag

# Restart if needed
docker restart mariadb-rag
```

### Issue: Ollama Not Responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve &
```

### Issue: Magic Commands Not Found

```python
# In Jupyter, reload the extension
%reload_ext mariadb_rag_magics

# Or force reload
import sys
sys.path.append('.')
%load_ext mariadb_rag_magics
```

### Issue: Model Download Fails

```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf ~/.cache/huggingface/
pip install --upgrade sentence-transformers
```

## 🎬 Demo Flow (5 Minutes)

For presentations, follow this flow:

1. **Introduction** (30s): "Local RAG with MariaDB Vector"
2. **Vector Indexing** (1m): Show embedding creation
3. **Semantic Search** (1.5m): Demo similarity search
4. **RAG Queries** (2m): Ask AI questions about data
5. **Benefits** (30s): Local, no external APIs, real-time

## ⚙️ Advanced Configuration

### Different Embedding Models

```python
# Use higher-quality embeddings (slower but more accurate)
%vector_index movies description --model all-mpnet-base-v2
```

### Custom Search Parameters

```python
# Get more results
%semantic_search movies "action movie" --top_k 10

# Custom context and LLM provider for RAG
%%rag_query movies --top_k 5 --llm huggingface
What's the best action movie for a family?

# Compare responses from different providers
%%rag_query movies --llm ollama
What are good romantic comedies?

%%rag_query movies --llm huggingface  
What are good romantic comedies?
```

### Multiple Datasets

```python
# Load airport data too
python demo/setup_openflights.py

# Search airports
%vector_index airports description --model all-MiniLM-L6-v2
%semantic_search airports "international hub"
```

## 🏆 Success Criteria

You'll know everything is working when:

- ✅ MariaDB container is running and accessible
- ✅ Magic commands load without errors
- ✅ Vector indexing completes successfully  
- ✅ Semantic search returns relevant results
- ✅ RAG queries generate AI responses
- ✅ Ollama responds to API calls

## 📞 Getting Help

If you encounter issues:

1. **Check Prerequisites**: Ensure Docker and Python are working
2. **Verify Services**: Both MariaDB and Ollama must be running
3. **Check Logs**: Use `docker logs mariadb-rag` for database issues
4. **Test Components**: Use the quick test commands above
5. **Restart Services**: Sometimes a restart fixes connection issues

## 🎯 Next Steps

Once the basic setup works:

- **Explore Different Models**: Try various embedding models
- **Add Your Data**: Replace sample data with your own content
- **Customize Queries**: Experiment with different RAG prompts
- **Scale Up**: Add more complex datasets and use cases

---

**🎉 Congratulations!** You now have a fully functional local RAG system running with MariaDB Vector!

*This setup gives you semantic search and AI-powered question answering without any external API dependencies.*
