# 🚀 **How to Integrate MariaDB RAG in Jupyter**

**Complete guide to add RAG magic commands to your Jupyter environment**

## 🎯 **Quick Setup (5 minutes)**

### Step 1: Install Dependencies
```bash
pip install jupyter ipython numpy
```

### Step 2: Copy Magic Commands File
```bash
# Copy jupyter_rag_magics.py to your project directory
# Or install it as a package
```

### Step 3: Start Jupyter
```bash
jupyter notebook
# Or use JupyterLab:
jupyter lab
```

### Step 4: Load Magic Commands in Notebook
```python
%load_ext jupyter_rag_magics
```

## 🎬 **Usage Examples**

### 1. Create Vector Index
```python
%vector_index
```

### 2. Semantic Search
```python
%semantic_search "space adventure movies"
%semantic_search "artificial intelligence themes"
%semantic_search "romantic comedies"
```

### 3. RAG Queries
```python
%%rag_query
What are the best sci-fi movies for beginners?
```

```python
%%rag_query
I want something with strong character development. Any recommendations?
```

## 🔧 **Advanced Integration**

### Custom Data Sources
Replace the `DEMO_MOVIES` in `jupyter_rag_magics.py` with your data:

```python
# Your custom data
YOUR_DATA = [
    {"id": 1, "title": "Item 1", "content": "Description...", "category": "Type A"},
    {"id": 2, "title": "Item 2", "content": "Description...", "category": "Type B"},
    # ... more data
]
```

### MariaDB Integration
For production, connect to real MariaDB Vector:

```python
import mariadb

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'your_user',
    'password': 'your_password',
    'database': 'your_database'
}

def get_data_from_db():
    conn = mariadb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM your_table")
    return cursor.fetchall()
```

### Enhanced Embeddings
Use real sentence-transformers for production:

```python
from sentence_transformers import SentenceTransformer

def production_embedding(self, text: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([text])[0]
```

## 🎯 **Magic Command Reference**

| Command | Purpose | Usage |
|---------|---------|-------|
| `%vector_index` | Create embeddings | `%vector_index` |
| `%semantic_search` | Find similar items | `%semantic_search "query"` |
| `%%rag_query` | Ask AI questions | `%%rag_query\nYour question` |

## 🚀 **Production Deployment**

### 1. Package Installation
```bash
# Create setup.py for your magic commands
pip install -e .
```

### 2. Auto-load in Jupyter
Add to `~/.ipython/profile_default/ipython_config.py`:
```python
c.InteractiveShellApp.extensions = ['jupyter_rag_magics']
```

### 3. Team Distribution
```bash
# Share via pip
pip install your-rag-package

# Or via conda
conda install -c your-channel your-rag-package
```

## 🎬 **Demo Files Included**

- `jupyter_rag_magics.py` - Magic commands implementation
- `interactive_rag_demo.ipynb` - Complete demo notebook
- `jupyter_integration_demo.py` - Standalone demo script

## 🔍 **Troubleshooting**

### Magic Commands Not Loading
```python
# Reload if needed
%reload_ext jupyter_rag_magics

# Check if loaded
%lsmagic
```

### Import Errors
```bash
# Install missing dependencies
pip install numpy ipython

# Check Python path
import sys
print(sys.path)
```

### Performance Issues
```python
# Reduce data size for testing
DEMO_MOVIES = DEMO_MOVIES[:5]  # Use fewer items

# Optimize embeddings
def simple_embedding(text):
    # Use simpler embedding for testing
    return [len(text), text.count(' ')]
```

## 🎉 **Success!**

You now have RAG capabilities integrated directly into Jupyter! 

**Key Benefits:**
- ✅ Interactive data exploration with AI
- ✅ Semantic search in notebook cells
- ✅ Real-time RAG query responses
- ✅ Easy integration with existing workflows

**Next Steps:**
- Connect to your actual data sources
- Add MariaDB Vector for production scale
- Integrate with Ollama for advanced LLM responses
- Share with your team for collaborative AI workflows

🚀 **Transform your data science workflows with local, private RAG!**