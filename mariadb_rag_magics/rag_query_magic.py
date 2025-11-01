"""
RAG Query Magic Command

Implements the %%rag_query cell magic command for Retrieval-Augmented Generation
using MariaDB Vector and multiple LLM providers (Ollama, HuggingFace).
"""

import mariadb
import requests
import json
import os
import sys
import re
from IPython.core.magic import cell_magic, Magics, magics_class
from sentence_transformers import SentenceTransformer

# Import LLM provider abstraction
try:
    from .llm_providers import LLMProviderFactory, BaseLLMProvider
except ImportError:
    # Handle relative import for standalone execution
    from llm_providers import LLMProviderFactory, BaseLLMProvider

# Robust config import with fallback (mirrors other magic modules)
try:
    from config import (
        DB_CONFIG,
        EMBEDDING_MODEL,
        OLLAMA_CONFIG,
        HUGGINGFACE_CONFIG,
        AVAILABLE_LLM_PROVIDERS,
        DEFAULT_LLM_PROVIDER,
        MAX_CONTEXT_RECORDS,
        AVAILABLE_EMBEDDING_MODELS,
    )
except ImportError:
    # Attempt to add parent directory (where config.py resides) to sys.path
    try:
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import (
            DB_CONFIG,
            EMBEDDING_MODEL,
            OLLAMA_CONFIG,
            HUGGINGFACE_CONFIG,
            AVAILABLE_LLM_PROVIDERS,
            DEFAULT_LLM_PROVIDER,
            MAX_CONTEXT_RECORDS,
            AVAILABLE_EMBEDDING_MODELS,
        )  # retry import
    except Exception:
        # Final fallback defaults (kept minimal to allow extension load)
        DB_CONFIG = {
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'password': 'demo123',
            'database': 'rag_demo',
            'unix_socket': None,
        }
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
        EMBEDDING_MODEL = AVAILABLE_EMBEDDING_MODELS['all-MiniLM-L6-v2']['model_name']
        MAX_CONTEXT_RECORDS = 3
        OLLAMA_CONFIG = {
            'base_url': 'http://localhost:11434',
            'model': 'llama2',
            'timeout': 30,
        }
        HUGGINGFACE_CONFIG = {
            'default_model': 'microsoft/DialoGPT-medium',
            'device': 'auto',
            'max_length': 512,
            'temperature': 0.7,
        }
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
                    'microsoft/phi-2',
                    'microsoft/DialoGPT-medium',
                    'microsoft/DialoGPT-large', 
                    'facebook/blenderbot-400M-distill',
                    'facebook/blenderbot-1B-distill'
                ]
            }
        }
        DEFAULT_LLM_PROVIDER = 'ollama'

# Removed model_manager dependency; automatic model selection handled locally


@magics_class
class RagQueryMagic(Magics):
    """Magic command for RAG queries using database context and multiple LLM providers."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.db_connection = None
        self.embedding_model = None
        self.llm_providers = {}  # Cache for LLM provider instances
    
    def parse_cell_args(self, line, cell):
        """
        Parse cell magic arguments for the rag_query command.
        
        Expected formats:
        %%rag_query <table>
        %%rag_query <table> --top_k N
        %%rag_query <table> --llm provider
        %%rag_query <table> --top_k N --llm provider
        %%rag_query -of
        %%rag_query -of --top_k N --llm provider
        %%rag_query -openflights
        %%rag_query -openflights --top_k N --llm provider
        <user question>
        
        Args:
            line: First line containing table name or dataset flag, optionally with --top_k and --llm
            cell: Cell body containing the user question
            
        Returns:
            dict: Parsed arguments with 'dataset', 'table', 'question', 'top_k', and 'llm_provider' keys, or None if invalid
        """
        if not line.strip():
            return None
        
        # Extract parameters if present
        top_k = MAX_CONTEXT_RECORDS  # default value
        llm_provider = DEFAULT_LLM_PROVIDER  # default value
        model_name = None  # default to provider's default model
        line_parts = line.strip()
        
        # Check for --top_k argument
        top_k_match = re.search(r'--top_k\s+(\d+)', line_parts)
        if top_k_match:
            top_k = int(top_k_match.group(1))
            # Remove the --top_k argument from the line
            line_parts = re.sub(r'\s*--top_k\s+\d+', '', line_parts).strip()
        
        # Check for --llm argument
        llm_match = re.search(r'--llm\s+(\w+)', line_parts)
        if llm_match:
            llm_provider = llm_match.group(1).lower()
            # Remove the --llm argument from the line
            line_parts = re.sub(r'\s*--llm\s+\w+', '', line_parts).strip()
        
        # Check for --model argument
        model_match = re.search(r'--model\s+([\w/\-.:]+)', line_parts)
        if model_match:
            model_name = model_match.group(1)
            # Remove the --model argument from the line
            line_parts = re.sub(r'\s*--model\s+[\w/\-.:]+', '', line_parts).strip()
        
        # Validate LLM provider
        if llm_provider not in AVAILABLE_LLM_PROVIDERS:
            print(f"Error: Unknown LLM provider '{llm_provider}'")
            print(f"Available providers: {', '.join(AVAILABLE_LLM_PROVIDERS.keys())}")
            return None
        
        # Check for dataset flag
        dataset = 'movies'  # default
        table_name = None
        
        # Parse line arguments (after removing --top_k and --llm)
        line_args = line_parts.split()
        
        if len(line_args) >= 1:
            if line_args[0] in ['-of', '-openflights']:
                dataset = 'openflights'
                table_name = 'airports'
            else:
                dataset = 'movies'
                table_name = line_args[0]
        else:
            return None
        
        # Extract question from cell body
        question = cell.strip() if cell else None
        
        # Basic validation
        if not table_name or not question:
            return None
        
        return {
            'dataset': dataset,
            'table': table_name,
            'question': question,
            'top_k': top_k,
            'llm_provider': llm_provider,
            'model_name': model_name
        }
    
    def validate_arguments(self, args):
        """
        Validate parsed arguments for the RAG query command.
        
        Args:
            args: Dictionary with 'table' and 'question' keys
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if args is None:
            return False, "Invalid argument format. Use: %%rag_query <table> with question in cell body"
        
        table_name = args.get('table', '').strip()
        question = args.get('question', '').strip()
        
        # Validate table name
        if not table_name:
            return False, "Table name is required on the first line"
        
        if not table_name.replace('_', '').isalnum():
            return False, f"Invalid table name '{table_name}'. Use only letters, numbers, and underscores."
        
        # Validate question
        if not question:
            return False, "Question text is required in the cell body"
        
        if len(question) < 5:
            return False, "Question must be at least 5 characters long"
        
        if len(question) > 1000:
            return False, "Question is too long (maximum 1000 characters)"
        
        return True, ""
    
    def get_db_connection(self):
        """Get database connection with error handling."""
        try:
            if self.db_connection is None or not self.db_connection.open:
                self.db_connection = mariadb.connect(**DB_CONFIG)
                # Ensure we're using the correct database
                cursor = self.db_connection.cursor()
                cursor.execute(f"USE {DB_CONFIG['database']}")
                cursor.close()
            return self.db_connection
        except mariadb.Error as e:
            print(f"Error: Cannot connect to MariaDB database.")
            print(f"Details: {e}")
            print("Please check that:")
            print("- MariaDB server is running")
            print("- Database credentials are correct")
            print("- Database exists")
            return None
        except Exception as e:
            print(f"Error: Unexpected database connection error: {e}")
            return None
    
    def check_table_exists(self, table_name):
        """
        Check if the specified table exists in the database.
        
        Args:
            table_name: Name of the table to check
        
        Returns:
            bool: True if table exists, False otherwise
        """
        conn = self.get_db_connection()
        if conn is None:
            return False
        
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s
            """
            
            cursor.execute(query, (DB_CONFIG['database'], table_name))
            result = cursor.fetchone()
            
            cursor.close()
            return result[0] > 0
            
        except mariadb.Error as e:
            print(f"Error checking table existence: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking table: {e}")
            return False
    
    def find_vector_column(self, table_name):
        """
        Find the vector column in the specified table.
        
        Args:
            table_name: Name of the table to check
        
        Returns:
            str: Name of the vector column, or None if not found
        """
        conn = self.get_db_connection()
        if conn is None:
            return None
        
        try:
            cursor = conn.cursor()
            
            # Look for vector columns in the table
            query = """
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s 
                AND DATA_TYPE = 'vector'
                ORDER BY COLUMN_NAME
            """
            
            cursor.execute(query, (DB_CONFIG['database'], table_name))
            results = cursor.fetchall()
            
            cursor.close()
            
            if results:
                # For airports table, prefer description_vector
                if table_name == 'airports':
                    for result in results:
                        if result[0] == 'description_vector':
                            return result[0]
                # Return first vector column found
                return results[0][0]
            
            return None
            
        except mariadb.Error as e:
            print(f"Error finding vector column: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error finding vector column: {e}")
            return None
    
    def get_model_for_table(self, table_name):
        """
        Determine the correct embedding model for a given table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            str: Model key to use for this table
        """
        # Model selection based on table
        if table_name == 'airports':
            return 'all-mpnet-base-v2'  # High-quality model for aviation data
        elif table_name == 'demo_content':
            return 'all-MiniLM-L6-v2'   # Fast model for movies
        else:
            # Default to fast model for unknown tables
            return 'all-MiniLM-L6-v2'

    def load_embedding_model(self, model_name=None):
        """
        Load the HuggingFace sentence transformer model with error handling.
        
        Args:
            model_name: Specific model to load (if None, uses default EMBEDDING_MODEL)
        
        Returns:
            SentenceTransformer: Loaded model or None if loading fails
        """
        # Use specified model or default
        target_model = model_name if model_name else EMBEDDING_MODEL
        
        # Check if we already have this model loaded
        if (self.embedding_model is not None and 
            hasattr(self, '_current_model_name') and 
            self._current_model_name == target_model):
            return self.embedding_model
        
        try:
            print(f"Loading embedding model: {target_model}")
            self.embedding_model = SentenceTransformer(target_model)
            self._current_model_name = target_model
            return self.embedding_model
            
        except ImportError as e:
            print("Error: sentence-transformers library not found.")
            print("Please install it using: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Please check that:")
            print("- You have internet connection (for first-time model download)")
            print("- sentence-transformers is properly installed")
            return None
    
    def retrieve_context_from_database(self, table_name, question, dataset='movies', top_k=None):
        """
        Retrieve relevant context from database using semantic search.
        
        Args:
            table_name: Name of the table to search
            question: User question to find relevant context for
            dataset: Dataset type ('movies' or 'openflights') to determine column schema
            top_k: Number of context records to retrieve (defaults to MAX_CONTEXT_RECORDS)
        
        Returns:
            list: List of relevant records or None if retrieval fails
        """
        print("🔍 Retrieving relevant context from database...")
        
        # Use provided top_k or fall back to default
        if top_k is None:
            top_k = MAX_CONTEXT_RECORDS
        
        # Check if table has vector data
        vector_column = self.find_vector_column(table_name)
        if not vector_column:
            print(f"Error: No vector column found in table '{table_name}'")
            print("Please run %vector_index first to create embeddings for this table.")
            return None
        
        # Auto-detect the model to use for this table
        model_key = self.get_model_for_table(table_name)
        
        # Get model info for display
        from config import AVAILABLE_EMBEDDING_MODELS
        model_info = AVAILABLE_EMBEDDING_MODELS.get(model_key, {})
        model_name = model_info.get('model_name', model_key)
        print(f"📊 Using model: {model_key} ({model_info.get('description', 'Unknown model')})")
        
        model = self.load_embedding_model(model_name)
        
        if model is None:
            return None
        
        # Generate embedding for the question
        try:
            question_embedding = model.encode([question], convert_to_numpy=True)[0]
        except Exception as e:
            print(f"Error generating question embedding: {e}")
            return None
        
        # Execute similarity search
        conn = self.get_db_connection()
        if conn is None:
            return None
        
        try:
            cursor = conn.cursor()
            
            # Convert question embedding to string format for VEC_FromText
            embedding_str = '[' + ','.join(map(str, question_embedding.tolist())) + ']'
            
            # Determine column names based on dataset
            if dataset == 'openflights' or table_name == 'airports':
                content_col = 'description'
                title_col = 'name'
            else:
                content_col = 'content'
                title_col = 'title'
            
            # Execute similarity search query to get relevant context
            query = f"""
                SELECT id, {content_col}, {title_col},
                       VEC_DISTANCE_COSINE({vector_column}, VEC_FromText(%s)) as similarity_score
                FROM {table_name}
                WHERE {vector_column} IS NOT NULL
                ORDER BY similarity_score ASC
                LIMIT %s
            """
            
            cursor.execute(query, (embedding_str, top_k))
            results = cursor.fetchall()
            
            cursor.close()
            
            print(f"   Found {len(results)} relevant records for context (top {top_k})")
            return results
            
        except mariadb.Error as e:
            print(f"Error retrieving context: {e}")
            print("Please check that:")
            print(f"- Vector column '{vector_column}' contains valid embeddings")
            print("- MariaDB Vector extension is properly installed")
            if dataset == 'openflights' or table_name == 'airports':
                print("- Table has 'id', 'name', and 'description' columns")
            else:
                print("- Table has 'id', 'title', and 'content' columns")
            return None
        except Exception as e:
            print(f"Unexpected error retrieving context: {e}")
            return None
    
    def format_context_for_llm(self, context_records):
        """
        Format retrieved records into a context string for the LLM.
        
        Args:
            context_records: List of tuples (id, content, title, similarity_score)
        
        Returns:
            str: Formatted context string
        """
        if not context_records:
            return "No relevant context found in the database."
        
        context_parts = []
        context_parts.append("Here is relevant information from the database:")
        context_parts.append("")
        
        for i, (record_id, content, title, similarity_score) in enumerate(context_records, 1):
            context_parts.append(f"Source {i}:")
            
            # Add title if available
            if title and str(title).strip():
                context_parts.append(f"Title: {str(title).strip()}")
            
            # Add content
            if content and str(content).strip():
                context_parts.append(f"Content: {str(content).strip()}")
            else:
                context_parts.append("Content: [No content available]")
            
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    def check_ollama_connection(self):
        """
        Check if Ollama server is running and accessible.
        
        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{OLLAMA_CONFIG['base_url']}/api/tags", 
                                  timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def build_rag_prompt(self, context, question):
        """
        Build a prompt template combining context and user question for the LLM.
        
        Args:
            context: Formatted context string from database
            question: User's question
        
        Returns:
            str: Complete prompt for the LLM
        """
        prompt_template = f"""You are a helpful assistant that answers questions based on the provided context from a database.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based only on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but informative in your response
- Reference specific sources when possible (e.g., "According to Source 1...")
- Do not make up information that is not in the context

Answer:"""
        
        return prompt_template
    
    def get_llm_provider(self, provider_type: str) -> BaseLLMProvider:
        """
        Get or create LLM provider instance with caching.
        
        Args:
            provider_type: Type of LLM provider ('ollama' or 'huggingface')
            
        Returns:
            BaseLLMProvider: Provider instance or None if creation fails
        """
        # Return cached provider if available
        if provider_type in self.llm_providers:
            return self.llm_providers[provider_type]
        
        try:
            # Get provider configuration
            provider_config = AVAILABLE_LLM_PROVIDERS[provider_type]['config']
            
            # Create new provider instance
            provider = LLMProviderFactory.create_provider(provider_type, provider_config)
            
            if provider and provider.initialize():
                # Cache successful provider
                self.llm_providers[provider_type] = provider
                return provider
            else:
                print(f"Failed to initialize {provider_type} provider")
                return None
                
        except Exception as e:
            print(f"Error creating {provider_type} provider: {e}")
            return None
    
    def call_llm_api(self, prompt: str, provider_type: str = None, model_name: str = None) -> str:
        """
        Send prompt to specified LLM provider and get response.
        
        Args:
            prompt: Complete prompt to send to the LLM
            provider_type: Type of LLM provider to use (defaults to DEFAULT_LLM_PROVIDER)
            model_name: Specific model to use (overrides provider default)
        
        Returns:
            str: Generated response from LLM or None if call fails
        """
        if provider_type is None:
            provider_type = DEFAULT_LLM_PROVIDER
        
        print(f"🧠 Generating answer using {AVAILABLE_LLM_PROVIDERS[provider_type]['name']}...")
        
        # If a specific model is requested, create a temporary provider with that model
        if model_name and provider_type == 'huggingface':
            # Create custom config with the specified model
            custom_config = AVAILABLE_LLM_PROVIDERS[provider_type]['config'].copy()
            custom_config['default_model'] = model_name
            
            # Create temporary provider
            provider = LLMProviderFactory.create_provider(provider_type, custom_config)
            if provider and provider.initialize():
                print(f"   ✅ Using custom model: {model_name}")
            else:
                print(f"   ❌ Failed to load custom model: {model_name}, using default")
                provider = self.get_llm_provider(provider_type)
        else:
            # Get cached LLM provider
            provider = self.get_llm_provider(provider_type)
        
        if not provider:
            self._show_llm_troubleshooting(provider_type)
            return None
        
        # Check if provider is available
        if not provider.is_available():
            print(f"Error: {AVAILABLE_LLM_PROVIDERS[provider_type]['name']} is not available")
            self._show_llm_troubleshooting(provider_type)
            return None
        
        # Generate response
        try:
            response = provider.generate_response(prompt)
            if response:
                print("   ✅ Answer generated successfully")
                return response
            else:
                print("   ⚠️  Empty response from LLM")
                return "No response generated by the LLM."
                
        except Exception as e:
            print(f"   ❌ Error generating response: {e}")
            return None
    
    def _show_llm_troubleshooting(self, provider_type: str):
        """Show troubleshooting information for LLM provider."""
        if provider_type == 'ollama':
            print("Please check that:")
            print("- Ollama is installed and running")
            print("- Start Ollama with: ollama serve")
            print(f"- Model '{OLLAMA_CONFIG['model']}' is available")
            print("- Check available models with: ollama list")
        elif provider_type == 'huggingface':
            print("Please check that:")
            print("- transformers library is installed: pip install transformers")
            print("- You have sufficient memory for the model")
            print("- Internet connection is available for first-time download")
            print("- CUDA is available if using GPU acceleration")
    
    # Backward compatibility: keep old method name
    def call_ollama_api(self, prompt: str) -> str:
        """Legacy method for backward compatibility."""
        return self.call_llm_api(prompt, 'ollama')
    
    def display_rag_response(self, question, answer, context_records, llm_provider=None):
        """
        Display the generated RAG response with source information.
        
        Args:
            question: Original user question
            answer: Generated answer from LLM
            context_records: List of context records used for generation
            llm_provider: LLM provider used for generation
        """
        provider_name = AVAILABLE_LLM_PROVIDERS.get(llm_provider, {}).get('name', 'LLM') if llm_provider else 'LLM'
        
        print("\n" + "="*80)
        print("🤖 RAG QUERY RESPONSE")
        print("="*80)
        
        print(f"\n🔮 LLM Provider: {provider_name}")
        if llm_provider:
            provider_info = self.llm_providers.get(llm_provider)
            if provider_info:
                model_info = provider_info.get_provider_info()
                print(f"   Model: {model_info.get('model', 'Unknown')}")
        
        print(f"\n❓ Question:")
        print(f"   {question}")
        
        print(f"\n💡 Answer:")
        # Format answer with proper line breaks
        answer_lines = answer.split('\n')
        for line in answer_lines:
            if line.strip():
                print(f"   {line}")
            else:
                print()
        
        print(f"\n📚 Sources Used:")
        if context_records:
            for i, (record_id, content, title, similarity_score) in enumerate(context_records, 1):
                similarity_percentage = max(0, (2 - similarity_score) / 2 * 100)
                
                print(f"\n   Source {i} (ID: {record_id}, Similarity: {similarity_percentage:.1f}%):")
                
                if title and str(title).strip():
                    title_display = str(title).strip()
                    if len(title_display) > 70:
                        title_display = title_display[:67] + "..."
                    print(f"   Title: {title_display}")
                
                if content and str(content).strip():
                    content_display = str(content).strip()
                    if len(content_display) > 100:
                        content_display = content_display[:97] + "..."
                    print(f"   Content: {content_display}")
        else:
            print("   No sources were used (no relevant context found)")
        
        print("\n" + "="*80)
        print("💡 Tip: The answer is based only on information from your database")
        print("   Use %semantic_search to explore more records on this topic")
        if llm_provider != DEFAULT_LLM_PROVIDER:
            print(f"   Used --llm {llm_provider} for this response")
        print("="*80 + "\n")
        
        print(f"\n❓ Question:")
        print(f"   {question}")
        
        print(f"\n💡 Answer:")
        # Format answer with proper line breaks
        answer_lines = answer.split('\n')
        for line in answer_lines:
            if line.strip():
                print(f"   {line}")
            else:
                print()
        
        print(f"\n📚 Sources Used:")
        if context_records:
            for i, (record_id, content, title, similarity_score) in enumerate(context_records, 1):
                similarity_percentage = max(0, (2 - similarity_score) / 2 * 100)
                
                print(f"\n   Source {i} (ID: {record_id}, Similarity: {similarity_percentage:.1f}%):")
                
                if title and str(title).strip():
                    title_display = str(title).strip()
                    if len(title_display) > 70:
                        title_display = title_display[:67] + "..."
                    print(f"   Title: {title_display}")
                
                if content and str(content).strip():
                    content_display = str(content).strip()
                    if len(content_display) > 100:
                        content_display = content_display[:97] + "..."
                    print(f"   Content: {content_display}")
        else:
            print("   No sources were used (no relevant context found)")
        
        print("\n" + "="*80)
        print("💡 Tip: The answer is based only on information from your database")
        print("   Use %semantic_search to explore more records on this topic")
        print("="*80 + "\n")
    
    def show_help(self):
        """Display help text for the rag_query magic command."""
        available_providers = ', '.join(AVAILABLE_LLM_PROVIDERS.keys())
        help_text = f"""
%%rag_query - Answer questions using database context via multiple LLM providers

Usage:
    %%rag_query <table>
    <your question here>
    
    %%rag_query <table> --top_k N
    <your question here>
    
    %%rag_query <table> --llm provider
    <your question here>
    
    %%rag_query <table> --top_k N --llm provider
    <your question here>

Arguments:
    table       Name of the database table to use for context
    --top_k N   Number of context records to retrieve (optional, default: {MAX_CONTEXT_RECORDS})
    --llm provider  LLM provider to use (optional, default: {DEFAULT_LLM_PROVIDER})

Available LLM Providers:
    {available_providers}

Examples:
    %%rag_query movies
    What are the best sci-fi movies for beginners?
    
    %%rag_query movies --top_k 5
    What are the best sci-fi movies for beginners?
    
    %%rag_query movies --llm huggingface
    What are the best sci-fi movies for beginners?
    
    %%rag_query movies --top_k 5 --llm ollama
    What are the best sci-fi movies for beginners?
    
    %%rag_query articles --top_k 2 --llm huggingface
    How does machine learning impact healthcare?

Description:
    This cell magic command performs Retrieval-Augmented Generation (RAG) by:
    
    1. Finding relevant context from the database using semantic search
    2. Combining the context with your question
    3. Sending to specified LLM provider for answer generation
    4. Displaying the generated answer with source information
    
LLM Providers:
    - ollama: Local Ollama server (requires ollama serve)
    - huggingface: Direct HuggingFace model inference (local processing)
    
Requirements:
    - Table must have vector embeddings (use %vector_index first)
    - For Ollama: Server running locally with a model loaded
    - For HuggingFace: transformers library installed
    - MariaDB with Vector extension
    - sentence-transformers library
    
Configuration:
    - Default provider: {DEFAULT_LLM_PROVIDER}
    - Ollama URL: {OLLAMA_CONFIG['base_url']}
    - Ollama model: {OLLAMA_CONFIG['model']}
    - HuggingFace model: {HUGGINGFACE_CONFIG['default_model']}
    - Default context records: {MAX_CONTEXT_RECORDS}
        """
        print(help_text)
    
    @cell_magic
    def rag_query(self, line, cell):
        """
        Answer questions using database context via multiple LLM providers.
        
        Usage: 
        %%rag_query <table>                       # Movies dataset (default provider & context)
        %%rag_query <table> --top_k N             # Movies dataset (N context records)
        %%rag_query <table> --llm provider        # Movies dataset (specific LLM provider)
        %%rag_query <table> --top_k N --llm provider # Movies dataset (N records, specific provider)
        %%rag_query -of                           # OpenFlights dataset  
        %%rag_query -of --top_k N --llm provider  # OpenFlights dataset (N records, specific provider)
        %%rag_query -openflights                  # OpenFlights dataset
        <your question>
        
        Available LLM providers: ollama, huggingface
        """
        # Show help if no arguments or help requested
        if not line.strip() or line.strip() in ['--help', '-h', 'help']:
            self.show_help()
            return
        
        # Parse arguments
        args = self.parse_cell_args(line, cell)
        
        # Validate arguments
        is_valid, error_message = self.validate_arguments(args)
        if not is_valid:
            print(f"❌ Error: {error_message}")
            print("Usage Examples:")
            print("   %%rag_query demo_content                        # Movies (default)")
            print("   What are good sci-fi movies?")
            print()
            print("   %%rag_query demo_content --top_k 5              # Movies (5 records)")
            print("   What are good sci-fi movies?")
            print()
            print("   %%rag_query demo_content --llm huggingface      # Movies (HuggingFace)")
            print("   What are good sci-fi movies?")
            print()
            print("   %%rag_query -of --top_k 2 --llm ollama         # OpenFlights (Ollama)")
            print("   Which airports are best for skiing?")
            print("Use '%%rag_query --help' for usage information.")
            return
        
        dataset = args.get('dataset', 'movies')
        table_name = args['table']
        question = args['question']
        llm_provider = args.get('llm_provider', DEFAULT_LLM_PROVIDER)
        
        # Display dataset-specific header
        provider_name = AVAILABLE_LLM_PROVIDERS.get(llm_provider, {}).get('name', llm_provider)
        
        if dataset == 'openflights':
            print("🛫" + "="*70)
            print(f"🌍 OPENFLIGHTS RAG QUERY")
            print(f"   Dataset: Global Airport Information")
            print(f"   LLM Provider: {provider_name}")
            print(f"   Question: {question}")
            print("="*70)
        else:
            print("🎬" + "="*70)
            print(f"🤖 MOVIES RAG QUERY")
            print(f"   Dataset: Movie Database")
            print(f"   LLM Provider: {provider_name}")
            print(f"   Question: {question}")
            print("="*70)
        
        print(f"🤖 Processing RAG query for table '{table_name}'...")
        print(f"Question: \"{question}\"")
        print()
        
        # Check if table exists
        if not self.check_table_exists(table_name):
            print(f"Error: Table '{table_name}' does not exist in database '{DB_CONFIG['database']}'")
            return None
        
        # Step 1: Retrieve relevant context from database
        context_records = self.retrieve_context_from_database(table_name, question, dataset, args.get('top_k'))
        if context_records is None:
            print("Failed to retrieve context from database. RAG query cannot proceed.")
            return None
        
        # Step 2: Format context for LLM
        formatted_context = self.format_context_for_llm(context_records)
        
        # Step 3: Build prompt combining context and question
        rag_prompt = self.build_rag_prompt(formatted_context, question)
        
        # Step 4: Generate answer using specified LLM provider
        generated_answer = self.call_llm_api(rag_prompt, llm_provider, args.get('model_name'))
        if generated_answer is None:
            print(f"Failed to generate answer using {provider_name}.")
            return None
        
        # Step 5: Display the complete RAG response
        self.display_rag_response(question, generated_answer, context_records, llm_provider)
        
        return {
            'table': table_name,
            'question': question,
            'answer': generated_answer,
            'context_records': context_records,
            'llm_provider': llm_provider,
            'status': 'complete'
        }
    
    def __del__(self):
        """Clean up database connection when object is destroyed."""
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            try:
                self.db_connection.close()
            except:
                pass  # Ignore errors during cleanup
