"""
Configuration settings for MariaDB RAG Magic Commands
"""

import os

# Database configuration - Force TCP connection for Docker compatibility
DB_CONFIG = {
    'host': os.getenv('MARIADB_HOST', '127.0.0.1'),  # Use IP instead of localhost
    'port': int(os.getenv('MARIADB_PORT', 3306)),
    'user': os.getenv('MARIADB_USER', 'root'),
    'password': os.getenv('MARIADB_PASSWORD', 'demo123'),
    'database': os.getenv('MARIADB_DATABASE', 'rag_demo'),
    # Force TCP connection (don't use Unix socket)
    'unix_socket': None,
}

# Multiple embedding models configuration
AVAILABLE_EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimension': 384,
        'description': 'Fast and efficient, good for general purpose',
        'use_case': 'General purpose, fast inference'
    },
    'all-mpnet-base-v2': {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'dimension': 768,
        'description': 'Higher quality embeddings, better semantic understanding',
        'use_case': 'High quality semantic search, slower but more accurate'
    }
}

# Default model configuration (backward compatibility)
EMBEDDING_MODEL = AVAILABLE_EMBEDDING_MODELS['all-MiniLM-L6-v2']['model_name']
EMBEDDING_DIMENSION = AVAILABLE_EMBEDDING_MODELS['all-MiniLM-L6-v2']['dimension']

# LLM Provider Configurations
OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'llama2'),
    'timeout': int(os.getenv('OLLAMA_TIMEOUT', 30)),
}

# HuggingFace LLM Configuration
HUGGINGFACE_CONFIG = {
    'default_model': os.getenv('HUGGINGFACE_MODEL', 'google/flan-t5-base'),
    'device': os.getenv('HUGGINGFACE_DEVICE', 'auto'),  # 'auto', 'cpu', or 'cuda'
    'max_length': int(os.getenv('HUGGINGFACE_MAX_LENGTH', 200)),
    'temperature': float(os.getenv('HUGGINGFACE_TEMPERATURE', 0.7)),
    'do_sample': True,
    'pad_token_id': None,  # Will be set dynamically based on model
}

# Available LLM Providers
AVAILABLE_LLM_PROVIDERS = {
    'ollama': {
        'name': 'Ollama',
        'description': 'Local Ollama server with various models',
        'config': OLLAMA_CONFIG,
        'default_model': OLLAMA_CONFIG['model'],
        'requires_server': True,
        'supported_models': ['llama2', 'llama2:7b', 'llama2:13b', 'codellama', 'mistral']
    },
    'huggingface': {
        'name': 'HuggingFace Transformers',
        'description': 'Direct HuggingFace model inference',
        'config': HUGGINGFACE_CONFIG,
        'default_model': HUGGINGFACE_CONFIG['default_model'],
        'requires_server': False,
        'supported_models': [
            'google/flan-t5-small',
            'google/flan-t5-base',
            'google/flan-t5-large',
            'microsoft/phi-2',
            'openai-community/gpt2',
            'openai-community/gpt2-medium'
        ]
    }
}

# Default LLM Provider (maintains backward compatibility)
DEFAULT_LLM_PROVIDER = 'ollama'

# Processing limits for demo
MAX_ROWS_FOR_INDEXING = 100
MAX_SEARCH_RESULTS = 5
MAX_CONTEXT_RECORDS = 3
