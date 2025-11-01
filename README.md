# 🚀 MariaDB RAG Magic Commands

**Multi-Domain Retrieval-Augmented Generation with Local Models and MariaDB Vector**

A implementation of Jupyter magic commands that demonstrate complete RAG workflows using MariaDB Vector with local models. Features **dual-dataset support** for entertainment and aviation intelligence!

## 🌟 **INNOVATION HIGHLIGHTS**

### 🎬🛫 **Dual-Dataset RAG Architecture**
- **Movies Dataset**: Traditional entertainment recommendations
- **OpenFlights Dataset**: Real-world aviation intelligence 
- **Seamless Switching**: Use `-of` flag to switch between domains
- **Cross-Domain Intelligence**: Same AI techniques, different applications

### 🔒 **Privacy-First & Local Processing**
- **100% Local**: No external APIs required - complete privacy and control
- **HuggingFace Embeddings**: Local sentence transformers
- **Ollama LLM**: Local language model inference  
- **MariaDB Vector**: Local vector storage and similarity search

## 🚀 Features

- 🔍 **`%vector_index`**: Create vector embeddings using HuggingFace transformers (local)
- 🎯 **`%semantic_search`**: Find similar records using vector similarity search
- 🤖 **`%%rag_query`**: Answer questions using database context via Ollama (local LLM)
- 🔒 **Privacy-First**: All processing happens locally - no data leaves your machine
- ⚡ **Fast Setup**: Docker-based demo ready in under 15 minutes

## 🎯 Quick Demo

```python
# Load the magic commands
%load_ext mariadb_rag_magics

# Create embeddings for movie descriptions
%vector_index demo_content content

# Find similar movies using semantic search
%semantic_search demo_content "space adventure with heroes"

# Ask AI questions about your data
%%rag_query demo_content
What are the best sci-fi movies for someone new to the genre?
```

## 📋 Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| **MariaDB** | 11.8+ | Vector storage & similarity search |
| **Ollama** | Latest | Local LLM inference |
| **Python** | 3.8+ | Magic commands & ML models |
| **Docker** | Optional | Easy MariaDB setup |

## 🛠️ Installation & Setup

### Option 1: Quick Docker Setup (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd mariadb-rag-magics

# 2. Start MariaDB with Vector support
docker run -d --name mariadb-vector \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=demo123 \
  mariadb:11.8

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Setup database and load sample data
python demo/setup_database.py

# 5. Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama2

# 6. Validate setup
python demo/validate_demo_setup.py

# 7. Start the demo
jupyter notebook demo/demo_notebook.ipynb
```

### Option 2: Manual Installation

<details>
<summary>Click to expand manual setup instructions</summary>

#### Step 1: Install MariaDB 11.8+

**Ubuntu/Debian:**
```bash
curl -sS https://downloads.mariadb.com/MariaDB/mariadb_repo_setup | sudo bash
sudo apt update
sudo apt install mariadb-server mariadb-plugin-vector
sudo systemctl start mariadb
```

**macOS (Homebrew):**
```bash
brew install mariadb
brew services start mariadb
```

**Windows:**
Download from [MariaDB Downloads](https://mariadb.org/download/) and ensure Vector plugin is included.

#### Step 2: Configure MariaDB

```sql
-- Connect to MariaDB
mysql -u root -p

-- Create database and user
CREATE DATABASE rag_demo;
CREATE USER 'rag_user'@'localhost' IDENTIFIED BY 'rag_password';
GRANT ALL PRIVILEGES ON rag_demo.* TO 'rag_user'@'localhost';
FLUSH PRIVILEGES;

-- Test Vector extension
USE rag_demo;
SELECT VEC_FromText('[0.1,0.2,0.3]') as test_vector;
```

#### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install mariadb sentence-transformers torch requests numpy jupyter ipython
```

#### Step 4: Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [Ollama website](https://ollama.ai/download)

**Start Ollama and pull a model:**
```bash
ollama serve &
ollama pull llama2  # or llama3, mistral, etc.
```

</details>

## 🔧 Configuration

Update `config.py` with your settings:

```python
# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',  # or 'rag_user'
    'password': 'demo123',  # or 'rag_password'
    'database': 'rag_demo'
}

# Ollama configuration
OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'model': 'llama2',  # or your preferred model
    'timeout': 60
}

# Model settings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384
```

## 🧪 Testing & Validation

### Validate Setup
```bash
# Check if all components are properly configured
python demo/validate_demo_setup.py
```

### Run End-to-End Tests
```bash
# Test complete RAG workflow (requires dependencies installed)
python demo/test_rag_workflow.py
```

### Manual Testing
```bash
# Test database connection
python demo/verify_sample_data.py

# Analyze sample data
python demo/analyze_sample_data.py
```

## 🎬 Demo Usage

### 1. Load Magic Commands
```python
%load_ext mariadb_rag_magics
```

### 2. Create Vector Indexes (Both Datasets)
```python
# Movies dataset
%vector_index demo_content content

# OpenFlights dataset (NEW!)
%vector_index -of
```

### 3. Semantic Search (Cross-Domain)
```python
# Movies - Find by meaning, not keywords
%semantic_search demo_content "artificial intelligence robots"
%semantic_search demo_content "romantic comedy date night"

# Aviation - Travel intelligence (NEW!)
%semantic_search -of "mountain airports skiing"
%semantic_search -of "tropical island beaches diving"
%semantic_search -of "international business hub"
```

### 4. RAG Queries (Dual Intelligence)
```python
# Movies - Ask questions about entertainment
%%rag_query demo_content
What are good sci-fi movies for beginners?
```

```python
%%rag_query demo_content
I want to watch something with strong character development and complex themes. What do you recommend?
```

```python
# Aviation - Travel intelligence (NEW!)
%%rag_query -of
I'm planning a skiing trip. Which airports would be best for accessing mountain ski resorts?
```

```python
%%rag_query -of
My family wants a tropical vacation with diving and beaches. Which airports serve destinations perfect for this?
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jupyter       │    │   Magic Commands │    │   MariaDB       │
│   Notebook      │◄──►│   - vector_index │◄──►│   + Vector      │
│                 │    │   - semantic_search│   │   Extension     │
│                 │    │   - rag_query    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Local ML Models │
                       │  - HuggingFace   │
                       │  - Ollama        │
                       └──────────────────┘
```

### Data Flow
1. **Text Input** → **HuggingFace Embeddings** → **MariaDB Vector Storage**
2. **Query** → **Embedding** → **Similarity Search** → **Context Retrieval**
3. **Context + Question** → **Ollama LLM** → **Generated Answer**

## 🚨 Troubleshooting

### Common Issues

**❌ "ModuleNotFoundError: No module named 'mariadb'"**
```bash
pip install mariadb
# On some systems, you may need:
pip install mariadb-connector-c
```

**❌ "Cannot connect to MariaDB database"**
- Check if MariaDB is running: `systemctl status mariadb`
- Verify credentials in `config.py`
- Test connection: `mysql -h localhost -u root -p`

**❌ "MariaDB Vector extension not found"**
- Ensure MariaDB 11.8+ is installed
- Test: `SELECT VEC_FromText('[0.1,0.2,0.3]');`
- Install Vector plugin if missing

**❌ "Ollama not available"**
- Check if Ollama is running: `curl http://localhost:11434/api/tags`
- Start Ollama: `ollama serve`
- Pull a model: `ollama pull llama2`

**❌ "sentence-transformers model download fails"**
- Check internet connection
- Ensure sufficient disk space (models are ~100MB)
- Try: `pip install --upgrade sentence-transformers`

### Performance Tips

- **Batch Size**: Adjust embedding batch size in config for your hardware
- **Model Choice**: Use smaller Ollama models (llama2-7b) for faster inference
- **Database**: Add indexes on frequently searched columns
- **Memory**: Ensure 4GB+ RAM for comfortable operation

## 📊 Sample Data

The demo includes 50+ movie records across genres:
- **Sci-Fi**: Inception, Matrix, Interstellar, Blade Runner 2049
- **Action**: Mad Max, John Wick, Die Hard, Terminator 2
- **Drama**: Shawshank Redemption, Forrest Gump, Goodfellas
- **Comedy**: Groundhog Day, Grand Budapest Hotel
- **Horror**: Get Out, A Quiet Place, Hereditary
- **Romance**: Princess Bride, Titanic, Casablanca
- **And more...**

## 🎯 Use Cases

### 🏢 Enterprise Applications
- **Knowledge Management**: Search internal documents by meaning
- **Customer Support**: AI-powered FAQ with company data
- **Research**: Find relevant papers and publications

### 🛒 E-commerce
- **Product Discovery**: Semantic product search
- **Recommendations**: Content-based suggestions
- **Customer Queries**: AI shopping assistant

### 📚 Content & Media
- **Content Recommendation**: Find similar articles/videos
- **Search Enhancement**: Beyond keyword matching
- **Automated Tagging**: Content categorization

## 🚀 Next Steps

### Extend the Demo
- Add more data sources (PDFs, web scraping)
- Implement multi-modal search (text + images)
- Create REST API endpoints
- Add real-time data updates

### Production Considerations
- Authentication and authorization
- Rate limiting and caching
- Monitoring and logging
- Horizontal scaling strategies

## 📁 Project Structure

```
mariadb-rag-magics/
├── 📁 mariadb_rag_magics/          # Main package
│   ├── __init__.py                 # Package initialization & magic registration
│   ├── vector_index_magic.py       # %vector_index implementation
│   ├── semantic_search_magic.py    # %semantic_search implementation
│   └── rag_query_magic.py          # %%rag_query implementation
├── 📁 demo/                        # Demo and testing files
│   ├── demo_notebook.ipynb         # 🎬 Main hackathon demo notebook
│   ├── sample_data.sql             # Movie database schema + data
│   ├── setup_database.py           # Automated database setup
│   ├── verify_sample_data.py       # Database validation
│   ├── analyze_sample_data.py      # Data analysis utilities
│   ├── validate_demo_setup.py      # Setup validation script
│   └── test_rag_workflow.py        # End-to-end testing
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This comprehensive guide
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with clear description

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- **MariaDB Foundation** for Vector extension
- **HuggingFace** for transformer models
- **Ollama** for local LLM runtime
- **Jupyter** for notebook environment

---

**🎬 Ready for your hackathon demo? Start with the [demo notebook](demo/demo_notebook.ipynb)!**

## Project Structure

```
mariadb-rag-magics/
├── mariadb_rag_magics/          # Main package
│   ├── __init__.py              # Package initialization
│   ├── vector_index_magic.py    # Vector indexing magic
│   ├── semantic_search_magic.py # Semantic search magic
│   └── rag_query_magic.py       # RAG query magic
├── demo/                        # Demo files
│   ├── demo_notebook.ipynb      # Main demo notebook
│   └── sample_data.sql          # Sample database schema and data
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## License

MIT License - see LICENSE file for details.
