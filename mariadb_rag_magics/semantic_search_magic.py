"""
Semantic Search Magic Command

Implements the %semantic_search magic command for finding similar records
using vector similarity search.
"""

import mariadb
import os
import sys
from IPython.core.magic import line_magic, Magics, magics_class
from sentence_transformers import SentenceTransformer
import re

# Import configuration with robust fallback
try:
    from config import (
        DB_CONFIG,
        EMBEDDING_MODEL,
        MAX_SEARCH_RESULTS,
        AVAILABLE_EMBEDDING_MODELS,
    )
except ImportError:
    try:
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config import (
            DB_CONFIG,
            EMBEDDING_MODEL,
            MAX_SEARCH_RESULTS,
            AVAILABLE_EMBEDDING_MODELS,
        )
    except Exception:
        # Final fallback configuration
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
        MAX_SEARCH_RESULTS = 5

# Removed model_manager dependency; automatic model selection handled locally


@magics_class
class SemanticSearchMagic(Magics):
    """Magic command for semantic search using vector similarity."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.db_connection = None
        self.embedding_model = None
    
    def parse_args(self, line):
        """
        Parse command line arguments for the semantic_search magic command.
        
        Expected formats: 
        %semantic_search <table> "<query text>" [--top_k N]
        %semantic_search -of "<query text>" [--top_k N]
        %semantic_search -openflights "<query text>" [--top_k N]
        
        Args:
            line: Command line string
            
        Returns:
            dict: Parsed arguments with 'dataset', 'table', 'query', and 'top_k' keys, or None if invalid
        """
        if not line.strip():
            return None
        
        # Extract top_k parameter if present
        top_k = MAX_SEARCH_RESULTS  # default value
        line_parts = line.strip()
        
        # Check for --top_k argument
        top_k_match = re.search(r'--top_k\s+(\d+)', line_parts)
        if top_k_match:
            top_k = int(top_k_match.group(1))
            # Remove the --top_k argument from the line
            line_parts = re.sub(r'\s*--top_k\s+\d+', '', line_parts).strip()
        
        # Check for dataset flag first
        dataset = 'movies'  # default
        
        # Handle OpenFlights dataset flags
        if line_parts.startswith('-of ') or line_parts.startswith('-openflights '):
            dataset = 'openflights'
            if line_parts.startswith('-of '):
                query_part = line_parts[4:]  # Remove '-of '
                table = 'airports'
            else:
                query_part = line_parts[13:]  # Remove '-openflights '
                table = 'airports'
            
            # Extract query from quotes
            query_match = re.match(r'^["\']([^"\']+)["\']$', query_part.strip())
            if query_match:
                query = query_match.group(1)
            else:
                # Try without quotes
                query = query_part.strip()
                
            if query:
                return {
                    'dataset': dataset,
                    'table': table,
                    'query': query,
                    'top_k': top_k
                }
        
        # Original format: table_name "query text" or table_name 'query text'
        pattern = r'^(\w+)\s+["\']([^"\']+)["\']$'
        match = re.match(pattern, line_parts)
        
        if match:
            table = match.group(1)
            # Determine dataset based on table name
            if table in ['airports', 'openflights']:
                dataset = 'openflights'
            else:
                dataset = 'movies'
                
            return {
                'dataset': dataset,
                'table': table,
                'query': match.group(2),
                'top_k': top_k
            }
        
        # Try alternative pattern without quotes (single word queries)
        simple_pattern = r'^(\w+)\s+(\w+)$'
        simple_match = re.match(simple_pattern, line_parts)
        
        if simple_match:
            return {
                'dataset': 'movies',
                'table': simple_match.group(1),
                'query': simple_match.group(2),
                'top_k': top_k
            }
        
        return None
    
    def validate_arguments(self, args):
        """
        Validate parsed arguments for the semantic search command.
        
        Args:
            args: Dictionary with 'table' and 'query' keys
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if args is None:
            return False, "Invalid argument format. Use: %semantic_search <table> \"<query text>\""
        
        table_name = args.get('table', '').strip()
        query_text = args.get('query', '').strip()
        
        # Validate table name
        if not table_name:
            return False, "Table name is required"
        
        if not table_name.replace('_', '').isalnum():
            return False, f"Invalid table name '{table_name}'. Use only letters, numbers, and underscores."
        
        # Validate query text
        if not query_text:
            return False, "Query text is required"
        
        if len(query_text) < 2:
            return False, "Query text must be at least 2 characters long"
        
        if len(query_text) > 500:
            return False, "Query text is too long (maximum 500 characters)"
        
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

    def load_embedding_model(self, model_key=None, table_name=None):
        """
        Load the HuggingFace sentence transformer model with error handling.
        
        Args:
            model_key: Key of the model to load, or None to auto-detect
            table_name: Table name for automatic model selection
        
        Returns:
            SentenceTransformer: Loaded model or None if loading fails
        """
        # Auto-detect model based on table if no model_key provided
        if model_key is None and table_name:
            model_key = self.get_model_for_table(table_name)
        
        # Get model info
        if model_key:
            model_info = AVAILABLE_EMBEDDING_MODELS.get(model_key)
            if model_info is None:
                print(f"Error: Unknown model key '{model_key}'")
                return None
            model_name = model_info['model_name']
        else:
            # Use default model
            model_name = EMBEDDING_MODEL
            model_key = 'all-MiniLM-L6-v2'  # Default key
        
        # Check if we already have this model loaded
        if (self.embedding_model is not None and 
            hasattr(self, '_current_model_key') and 
            self._current_model_key == model_key):
            return self.embedding_model
        
        try:
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self._current_model_key = model_key
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
    
    def generate_query_embedding(self, query_text, model_key=None, table_name=None):
        """
        Generate embedding for the search query using HuggingFace sentence transformers.
        
        Args:
            query_text: Text query to embed
            model_key: Key of the model to use (auto-detects if None)
            table_name: Table name for automatic model selection
        
        Returns:
            numpy.ndarray: Query embedding vector or None if generation fails
        """
        model = self.load_embedding_model(model_key, table_name)
        if model is None:
            return None
        
        try:
            print("Generating embedding for search query...")
            
            # Generate embedding for the query
            query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
            
            print("Query embedding generated successfully")
            return query_embedding
            
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None
    
    def execute_similarity_search(self, table_name, vector_column, query_embedding, dataset='movies', top_k=None):
        """
        Execute similarity search using VEC_DISTANCE_COSINE function.
        
        Args:
            table_name: Name of the table to search
            vector_column: Name of the vector column
            query_embedding: Query embedding vector
            dataset: Dataset type ('movies' or 'openflights') to determine column schema
            top_k: Number of results to return (defaults to MAX_SEARCH_RESULTS)
        
        Returns:
            list: List of tuples (id, content, title, similarity_score) or None if search fails
        """
        conn = self.get_db_connection()
        if conn is None:
            return None
        
        # Use provided top_k or default
        limit = top_k if top_k is not None else MAX_SEARCH_RESULTS
        
        try:
            cursor = conn.cursor()
            
            # Convert query embedding to string format for VEC_FromText
            embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
            
            # Determine column names based on dataset
            if dataset == 'openflights' or table_name == 'airports':
                content_col = 'description'
                title_col = 'name'
            else:
                content_col = 'content'
                title_col = 'title'
            
            # Execute similarity search query
            query = f"""
                SELECT id, {content_col}, {title_col},
                       VEC_DISTANCE_COSINE({vector_column}, VEC_FromText(%s)) as similarity_score
                FROM {table_name}
                WHERE {vector_column} IS NOT NULL
                ORDER BY similarity_score ASC
                LIMIT %s
            """
            
            cursor.execute(query, (embedding_str, limit))
            results = cursor.fetchall()
            
            cursor.close()
            
            print(f"Similarity search completed, found {len(results)} results (top {limit})")
            return results
            
        except mariadb.Error as e:
            print(f"Error executing similarity search: {e}")
            print("Please check that:")
            print(f"- Vector column '{vector_column}' contains valid embeddings")
            print("- MariaDB Vector extension is properly installed")
            if dataset == 'openflights' or table_name == 'airports':
                print("- Table has 'id', 'name', and 'description' columns")
            else:
                print("- Table has 'id', 'title', and 'content' columns")
            return None
        except Exception as e:
            print(f"Unexpected error during similarity search: {e}")
            return None
    
    def format_and_display_results(self, query_text, search_results):
        """
        Format and display search results in a readable format.
        
        Args:
            query_text: Original search query
            search_results: List of tuples (id, content, title, similarity_score)
        """
        print("\n" + "="*80)
        print(f"🔍 SEMANTIC SEARCH RESULTS")
        print(f"Query: \"{query_text}\"")
        print(f"Found {len(search_results)} similar records")
        print("="*80)
        
        for i, (record_id, content, title, similarity_score) in enumerate(search_results, 1):
            # Convert similarity score to percentage (lower cosine distance = higher similarity)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # Convert to similarity percentage: (2 - distance) / 2 * 100
            # Handle None similarity scores gracefully
            if similarity_score is None:
                similarity_percentage = 0.0
                print(f"⚠️ Warning: Null similarity score for record {record_id}")
            else:
                similarity_percentage = max(0, (2 - similarity_score) / 2 * 100)
            
            print(f"\n📄 Result #{i}")
            print(f"   ID: {record_id}")
            print(f"   Similarity: {similarity_percentage:.1f}%")
            
            # Display title if available
            if title and str(title).strip():
                title_display = str(title).strip()
                if len(title_display) > 60:
                    title_display = title_display[:57] + "..."
                print(f"   Title: {title_display}")
            
            # Display content preview
            if content and str(content).strip():
                content_display = str(content).strip()
                if len(content_display) > 150:
                    content_display = content_display[:147] + "..."
                print(f"   Content: {content_display}")
            else:
                print(f"   Content: [No content available]")
            
            # Display distance score with null handling
            if similarity_score is not None:
                print(f"   Distance Score: {similarity_score:.4f}")
            else:
                print(f"   Distance Score: [Unable to calculate]")
        
        print("\n" + "="*80)
        print("💡 Tip: Lower distance scores indicate higher similarity")
        print("   Use the record IDs to retrieve full content from the database")
        print("="*80 + "\n")
    
    def show_help(self):
        """Display help text for the semantic_search magic command."""
        help_text = f"""
%semantic_search - Find similar records using vector similarity search

Usage:
    %semantic_search <table> "<query text>" [--top_k N]

Arguments:
    table       Name of the database table to search
    query text  Text query to find similar records (must be quoted)
    --top_k N   Number of results to return (optional, default: {MAX_SEARCH_RESULTS})

Examples:
    %semantic_search movies "space adventure"
    %semantic_search movies "romantic comedy" --top_k 5
    %semantic_search articles "machine learning" --top_k 10
    %semantic_search products "wireless headphones" --top_k 3

OpenFlights Dataset:
    %semantic_search -of "mountain airports" --top_k 5
    %semantic_search -openflights "business hubs" --top_k 3

Description:
    This magic command performs semantic search on a database table that has
    vector embeddings. It will:
    
    1. Generate an embedding for your query text
    2. Find the most similar records using cosine similarity
    3. Display results with similarity scores
    
Requirements:
    - Table must have vector embeddings (use %vector_index first)
    - MariaDB with Vector extension
    - sentence-transformers library
    
Note: Query text should be enclosed in quotes for multi-word queries.
        """
        print(help_text)
    
    @line_magic
    def semantic_search(self, line):
        """
        Perform semantic search on database table using vector similarity.
        
        Usage: 
        %semantic_search <table> "<query text>" [--top_k N]           # Movies dataset
        %semantic_search -of "<query text>" [--top_k N]               # OpenFlights dataset
        %semantic_search -openflights "<query text>" [--top_k N]      # OpenFlights dataset
        """
        # Show help if no arguments or help requested
        if not line.strip() or line.strip() in ['--help', '-h', 'help']:
            self.show_help()
            return
        
        # Parse arguments
        args = self.parse_args(line)
        
        # Validate arguments
        is_valid, error_message = self.validate_arguments(args)
        if not is_valid:
            print(f"❌ Error: {error_message}")
            print("Usage Examples:")
            print("   %semantic_search demo_content \"space adventure\" --top_k 3      # Movies")
            print("   %semantic_search -of \"international airports\" --top_k 5       # OpenFlights")
            print("   %semantic_search -openflights \"mountain airports\" --top_k 2   # OpenFlights")
            print("Use '%semantic_search --help' for usage information.")
            return
        
        dataset = args.get('dataset', 'movies')
        table_name = args['table']
        query_text = args['query']
        
        # Display dataset-specific info
        if dataset == 'openflights':
            print(f"🛫 OPENFLIGHTS SEMANTIC SEARCH")
            print(f"   Dataset: Global Airport Information")
            print(f"   Query: \"{query_text}\"")
            print("="*70)
        else:
            print(f"🎬 MOVIES SEMANTIC SEARCH")
            print(f"   Dataset: Movie Descriptions")
            print(f"   Query: \"{query_text}\"")
            print("="*70)
        
        # Check if table exists and has vector data
        if not self.check_table_exists(table_name):
            print(f"❌ Error: Table '{table_name}' does not exist in database '{DB_CONFIG['database']}'")
            return None
        
        # Check if table has vector data and get the model used for indexing
        vector_column = self.find_vector_column(table_name)
        if not vector_column:
            print(f"Error: No vector column found in table '{table_name}'")
            print("Please run %vector_index first to create embeddings for this table.")
            return None

        # Auto-detect which model to use for this table (fallback only)
        model_key = self.get_model_for_table(table_name)
        print(f"🤖 Using model: {model_key}")
        query_embedding = self.generate_query_embedding(query_text, model_key, table_name)
        if query_embedding is None:
            print("Failed to generate query embedding. Search cannot proceed.")
            return None

        # Execute similarity search
        search_results = self.execute_similarity_search(table_name, vector_column, query_embedding, dataset, args.get('top_k'))
        if search_results is None:
            print("Failed to execute similarity search.")
            return None

        if not search_results:
            print("No similar records found.")
            return {
                'table': table_name,
                'query': query_text,
                'results': [],
                'status': 'no_results'
            }

        # Format and display search results
        self.format_and_display_results(query_text, search_results)

        return {
            'table': table_name,
            'query': query_text,
            'results': search_results,
            'top_k': args.get('top_k'),
            'status': 'complete'
        }
    
    def __del__(self):
        """Clean up database connection when object is destroyed."""
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            try:
                self.db_connection.close()
            except:
                pass  # Ignore errors during cleanup
