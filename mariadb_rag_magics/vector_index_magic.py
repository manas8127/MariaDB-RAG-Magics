"""
Vector Index Magic Command

Implements the %vector_index magic command for creating vector embeddings
and storing them in MariaDB.
"""

import mariadb
import os
import sys
from IPython.core.magic import line_magic, Magics, magics_class
from sentence_transformers import SentenceTransformer
import numpy as np

# Import configuration with robust fallback (adds parent path on failure)
try:
    from config import (
        DB_CONFIG,
        EMBEDDING_DIMENSION,
        EMBEDDING_MODEL,
        MAX_ROWS_FOR_INDEXING,
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
            EMBEDDING_DIMENSION,
            EMBEDDING_MODEL,
            MAX_ROWS_FOR_INDEXING,
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
        EMBEDDING_DIMENSION = AVAILABLE_EMBEDDING_MODELS['all-MiniLM-L6-v2']['dimension']
        MAX_ROWS_FOR_INDEXING = 100

# Removed model_manager dependency; using static table->model mapping


@magics_class
class VectorIndexMagic(Magics):
    """Magic command for creating vector indexes on database tables."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.db_connection = None
        self.embedding_model = None
    
    def parse_args(self, line):
        """
        Parse command line arguments for the vector_index magic command.
        
        Supported formats:
        %vector_index <table> <column>                          # Default model (all-MiniLM-L6-v2)
        %vector_index <table> <column> --model <model_key>      # Specify model
        %vector_index -of                                       # OpenFlights default
        %vector_index -of --model <model_key>                   # OpenFlights with model
        %vector_index --models                                  # List available models
        """
        if not line.strip():
            return None
        
        # Split the line into arguments
        args = line.strip().split()
        
        # Handle special commands
        if '--models' in args:
            return {'action': 'list_models'}
        
        # Default values
        dataset = 'movies'
        table = None
        column = None
        model_key = 'all-MiniLM-L6-v2'  # Default model
        
        # Parse model specification
        if '--model' in args:
            try:
                model_index = args.index('--model')
                if model_index + 1 < len(args):
                    model_key = args[model_index + 1]
                    # Remove --model and its value from args
                    args = args[:model_index] + args[model_index + 2:]
                else:
                    print("Error: --model flag requires a model name")
                    return None
            except ValueError:
                pass
        
        # Validate model key
        if model_key not in AVAILABLE_EMBEDDING_MODELS:
            print(f"Error: Unknown model '{model_key}'")
            print("Available models:")
            for key, info in AVAILABLE_EMBEDDING_MODELS.items():
                print(f"  - {key}: {info['description']}")
            return None
        
        # Parse dataset and table/column
        if len(args) >= 1 and args[0] in ['-of', '-openflights']:
            dataset = 'openflights'
            if len(args) >= 3:
                table = args[1]
                column = args[2]
            else:
                # Default openflights configuration
                table = 'airports'
                column = 'description'
        elif len(args) >= 2:
            # Traditional format: table column
            dataset = 'movies'
            table = args[0]
            column = args[1]
        elif len(args) == 1:
            # Could be just a table name for movies dataset
            dataset = 'movies'
            table = args[0]
            column = 'content'  # default column
        else:
            return None
        
        # Ensure we have valid table and column
        if not table or not column:
            return None
        
        return {
            'action': 'index',
            'dataset': dataset,
            'table': table,
            'column': column,
            'model_key': model_key
        }
    
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
    
    def load_embedding_model(self, model_key='all-MiniLM-L6-v2'):
        """
        Load the HuggingFace sentence transformer model with error handling.
        
        Args:
            model_key: Key of the model to load from AVAILABLE_EMBEDDING_MODELS
        
        Returns:
            SentenceTransformer: Loaded model or None if loading fails
        """
        # Get model info
        model_info = AVAILABLE_EMBEDDING_MODELS.get(model_key)
        if model_info is None:
            print(f"Error: Unknown model key '{model_key}'")
            return None
        
        model_name = model_info['model_name']
        
        # Check if we already have this model loaded
        if (self.embedding_model is not None and 
            hasattr(self, '_current_model_key') and 
            self._current_model_key == model_key):
            return self.embedding_model
        
        try:
            print(f"Loading embedding model: {model_name}")
            print(f"Model: {model_key} ({model_info['description']})")
            print("This may take a moment on first run as the model is downloaded...")
            
            self.embedding_model = SentenceTransformer(model_name)
            self._current_model_key = model_key
            
            print(f"✅ Successfully loaded embedding model: {model_name}")
            print(f"   Dimensions: {model_info['dimension']}")
            print(f"   Use case: {model_info['use_case']}")
            return self.embedding_model
            
        except ImportError as e:
            print("Error: sentence-transformers library not found.")
            print("Please install it using: pip install sentence-transformers")
            print(f"Details: {e}")
            return None
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Please check that:")
            print("- You have internet connection (for first-time model download)")
            print("- You have sufficient disk space for model files")
            print("- sentence-transformers is properly installed")
            print("\nSetup instructions:")
            print("1. Install dependencies: pip install sentence-transformers torch")
            print("2. Ensure internet connection for model download")
            print("3. Retry the command")
            return None
    
    def generate_embeddings_batch(self, texts, model_key='all-MiniLM-L6-v2', batch_size=32):
        """
        Generate embeddings for a batch of texts using HuggingFace sentence transformers.
        
        Args:
            texts: List of text strings to embed
            model_key: Key of the embedding model to use
            batch_size: Number of texts to process in each batch
        
        Returns:
            list: List of embedding vectors or None if generation fails
        """
        model = self.load_embedding_model(model_key)
        if model is None:
            return None
        
        try:
            # Filter out None/empty texts
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text is not None and str(text).strip():
                    valid_texts.append(str(text).strip())
                    valid_indices.append(i)
            
            if not valid_texts:
                print("Warning: No valid text data found to embed")
                return []
            
            print(f"Generating embeddings for {len(valid_texts)} text entries...")
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for this batch
                batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings)
            
            print(f"Successfully generated {len(all_embeddings)} embeddings")
            
            # Create full result list with None for invalid entries
            result_embeddings = [None] * len(texts)
            for i, embedding in enumerate(all_embeddings):
                result_embeddings[valid_indices[i]] = embedding
            
            return result_embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            print("This could be due to:")
            print("- Insufficient memory for model processing")
            print("- Invalid text data format")
            print("- Model loading issues")
            return None
    
    def fetch_text_data(self, table_name, column_name, limit=None):
        """
        Fetch text data from the specified table and column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the text column
            limit: Maximum number of rows to fetch (defaults to MAX_ROWS_FOR_INDEXING)
        
        Returns:
            list: List of tuples (id, text) or None if fetch fails
        """
        conn = self.get_db_connection()
        if conn is None:
            return None
        
        if limit is None:
            limit = MAX_ROWS_FOR_INDEXING
        
        try:
            cursor = conn.cursor()
            
            # Fetch text data with row IDs (assuming 'id' column exists)
            query = f"""
                SELECT id, {column_name} 
                FROM {table_name} 
                WHERE {column_name} IS NOT NULL 
                AND {column_name} != ''
                LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            cursor.close()
            
            print(f"Fetched {len(results)} text records from table '{table_name}'")
            return results
            
        except mariadb.Error as e:
            print(f"Error fetching text data: {e}")
            print("Please check that:")
            print(f"- Table '{table_name}' exists")
            print(f"- Column '{column_name}' exists")
            print("- Table has an 'id' column")
            return None
        except Exception as e:
            print(f"Unexpected error fetching text data: {e}")
            return None
    
    def store_embeddings(self, table_name, vector_column_name, embeddings_data):
        """
        Store embeddings in the database vector column.
        
        Args:
            table_name: Name of the table
            vector_column_name: Name of the vector column
            embeddings_data: List of tuples (id, embedding_vector)
        
        Returns:
            bool: True if storage successful, False otherwise
        """
        conn = self.get_db_connection()
        if conn is None:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Prepare update query
            update_query = f"""
                UPDATE {table_name} 
                SET {vector_column_name} = VEC_FromText(%s)
                WHERE id = %s
            """
            
            successful_updates = 0
            
            for record_id, embedding in embeddings_data:
                if embedding is not None:
                    try:
                        # Ensure embedding is a numpy array and convert to proper format
                        # Convert to numpy array if not already
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding, dtype=np.float32)
                        
                        # Ensure it's float32 for consistent formatting
                        embedding = embedding.astype(np.float32)
                        
                        # Convert numpy array to string format for VEC_FromText
                        # Use proper float formatting to avoid scientific notation issues
                        embedding_values = [f"{float(val):.8f}" for val in embedding.tolist()]
                        embedding_str = '[' + ','.join(embedding_values) + ']'
                        
                        cursor.execute(update_query, (embedding_str, record_id))
                        successful_updates += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to store embedding for record {record_id}: {e}")
                        continue
            
            conn.commit()
            cursor.close()
            
            print(f"Successfully stored {successful_updates} embeddings in '{vector_column_name}' column")
            return True
            
        except mariadb.Error as e:
            print(f"Error storing embeddings: {e}")
            print("Please check that:")
            print(f"- Vector column '{vector_column_name}' exists")
            print("- MariaDB Vector extension is properly installed")
            print("- You have UPDATE privileges on the table")
            return False
        except Exception as e:
            print(f"Unexpected error storing embeddings: {e}")
            return False
    
    def check_vector_column_exists(self, table_name, vector_column_name=None):
        """
        Check if vector column exists in the specified table.
        
        Args:
            table_name: Name of the table to check
            vector_column_name: Name of vector column (defaults to {column}_vector)
        
        Returns:
            bool: True if vector column exists, False otherwise
        """
        conn = self.get_db_connection()
        if conn is None:
            return False
        
        try:
            cursor = conn.cursor()
            
            # If no vector column name provided, use default naming convention
            if vector_column_name is None:
                vector_column_name = f"{table_name}_vector"
            
            # Check if vector column exists
            query = """
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s 
                AND COLUMN_NAME = %s
                AND DATA_TYPE = 'vector'
            """
            
            cursor.execute(query, (DB_CONFIG['database'], table_name, vector_column_name))
            result = cursor.fetchone()
            
            cursor.close()
            return result[0] > 0
            
        except mariadb.Error as e:
            print(f"Error checking vector column: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking vector column: {e}")
            return False
    
    def add_vector_column(self, table_name, vector_column_name=None, dimension=None):
        """
        Add VECTOR column to the specified table if it doesn't exist.
        
        Args:
            table_name: Name of the table to modify
            vector_column_name: Name of vector column (defaults to {table}_vector)
            dimension: Vector dimension (defaults to EMBEDDING_DIMENSION from config)
        
        Returns:
            bool: True if column was added or already exists, False on error
        """
        conn = self.get_db_connection()
        if conn is None:
            return False
        
        try:
            # If no vector column name provided, use default naming convention
            if vector_column_name is None:
                vector_column_name = f"{table_name}_vector"
            
            # Use provided dimension or default
            if dimension is None:
                dimension = EMBEDDING_DIMENSION
            
            # Check if vector column already exists
            if self.check_vector_column_exists(table_name, vector_column_name):
                print(f"Vector column '{vector_column_name}' already exists in table '{table_name}'")
                return True
            
            cursor = conn.cursor()
            
            # Add VECTOR column with specified dimension
            alter_query = f"""
                ALTER TABLE {table_name} 
                ADD COLUMN {vector_column_name} VECTOR({dimension})
            """
            
            cursor.execute(alter_query)
            conn.commit()
            cursor.close()
            
            print(f"Successfully added vector column '{vector_column_name}' to table '{table_name}' (dimension: {dimension})")
            return True
            
        except mariadb.Error as e:
            print(f"Error adding vector column to table '{table_name}': {e}")
            print("Please check that:")
            print(f"- Table '{table_name}' exists")
            print("- You have ALTER privileges on the table")
            print("- MariaDB Vector extension is installed")
            return False
        except Exception as e:
            print(f"Unexpected error adding vector column: {e}")
            return False
    
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

    def show_help(self):
        """Display help text for the vector_index magic command."""
        help_text = f"""
%vector_index - Create vector embeddings for text data in MariaDB

Usage:
    %vector_index <table> <column>

Arguments:
    table    Name of the database table
    column   Name of the text column to create embeddings for

Examples:
    %vector_index movies description
    %vector_index articles content
    %vector_index products title

Description:
    This magic command creates vector embeddings for text data in a specified
    database table and column. It will:
    
    1. Add a vector column to store embeddings (if not exists)
    2. Fetch text data from the specified column (up to {MAX_ROWS_FOR_INDEXING} rows)
    3. Generate embeddings using {EMBEDDING_MODEL}
    4. Store embeddings in MariaDB vector format
    
Requirements:
    - MariaDB with Vector extension
    - sentence-transformers library
    - Internet connection (for first-time model download)
        """
        print(help_text)
    
    @line_magic
    def vector_index(self, line):
        """
        Create vector embeddings for text data and store in MariaDB.
        
        Usage: 
        %vector_index <table> <column>                          # Default model
        %vector_index <table> <column> --model <model_key>      # Specify model
        %vector_index -of                                       # OpenFlights default
        %vector_index -of --model <model_key>                   # OpenFlights with model
        %vector_index --models                                  # List available models
        """
        # Show help if no arguments or help requested
        if not line.strip() or line.strip() in ['--help', '-h', 'help']:
            self.show_help()
            return
        
        # Parse arguments
        args = self.parse_args(line)
        
        if args is None:
            print("❌ Error: Invalid arguments provided.")
            print("Usage Examples:")
            print("   %vector_index demo_content content                     # Default model")
            print("   %vector_index demo_content content --model all-mpnet-base-v2  # High quality model")
            print("   %vector_index -of                                      # OpenFlights default")
            print("   %vector_index -of --model all-mpnet-base-v2            # OpenFlights with model")
            print("   %vector_index --models                                 # List available models")
            print("Use '%vector_index --help' for more information.")
            return
        
        # Handle special actions
        if args.get('action') == 'list_models':
            print("Available embedding models:")
            for key, info in AVAILABLE_EMBEDDING_MODELS.items():
                print(f"  - {key}: {info['description']} ({info['dimension']} dims)")
            return
        
        # Extract arguments
        dataset = args['dataset']
        table_name = args['table']
        column_name = args['column']
        model_key = args['model_key']
        
        # Get model info
        model_info = AVAILABLE_EMBEDDING_MODELS[model_key]
        
        # Display dataset info with model information
        if dataset == 'openflights':
            print(f"🛫 OPENFLIGHTS DATASET - Vector Indexing")
            print(f"   Dataset: Global Airport Information")
            print(f"   Table: {table_name}")
            print(f"   Column: {column_name}")
            print(f"   🤖 Model: {model_key} ({model_info['description']})")
            print(f"   📊 Dimensions: {model_info['dimension']}")
            print("="*70)
        else:
            print(f"🎬 MOVIES DATASET - Vector Indexing")
            print(f"   Dataset: Movie Descriptions")
            print(f"   Table: {table_name}")
            print(f"   Column: {column_name}")
            print(f"   🤖 Model: {model_key} ({model_info['description']})")
            print(f"   📊 Dimensions: {model_info['dimension']}")
            print("="*70)
        
    # (Model change warnings disabled - simplified workflow)
        
        print(f"Starting vector index creation for table '{table_name}', column '{column_name}'...")
        
        # Basic argument validation
        if not table_name or not column_name:
            print("Error: Both table and column arguments are required.")
            return
        
        # Check for valid table/column names (basic validation)
        if not table_name.replace('_', '').isalnum():
            print(f"Error: Invalid table name '{table_name}'. Use only letters, numbers, and underscores.")
            return
        
        if not column_name.replace('_', '').isalnum():
            print(f"Error: Invalid column name '{column_name}'. Use only letters, numbers, and underscores.")
            return
        
        # Check if table exists
        if not self.check_table_exists(table_name):
            print(f"Error: Table '{table_name}' does not exist in database '{DB_CONFIG['database']}'")
            return None
        
        # Add vector column if needed (use model dimension)
        vector_column_name = f"{column_name}_vector"
        if not self.add_vector_column(table_name, vector_column_name, model_info['dimension']):
            print("Failed to add vector column. Cannot proceed with vector indexing.")
            return None
        
        print(f"Database setup complete for table '{table_name}'")
        
        # Fetch text data from the table
        print(f"Fetching text data from column '{column_name}'...")
        text_data = self.fetch_text_data(table_name, column_name)
        
        if text_data is None:
            print("Failed to fetch text data. Cannot proceed with embedding generation.")
            return None
        
        if not text_data:
            print("No text data found in the specified column.")
            return None
        
        # Extract texts for embedding generation
        texts = [row[1] for row in text_data]  # row[1] is the text content
        
        # Generate embeddings using specified model
        print("Generating embeddings using HuggingFace sentence transformers...")
        embeddings = self.generate_embeddings_batch(texts, model_key)
        
        if embeddings is None:
            print("Failed to generate embeddings. Vector indexing incomplete.")
            return None
        
        # Prepare data for storage (id, embedding pairs)
        embeddings_data = []
        for i, (record_id, text) in enumerate(text_data):
            if embeddings[i] is not None:
                embeddings_data.append((record_id, embeddings[i]))
        
        # Store embeddings in database
        print(f"Storing embeddings in vector column '{vector_column_name}'...")
        if not self.store_embeddings(table_name, vector_column_name, embeddings_data):
            print("Failed to store embeddings. Vector indexing incomplete.")
            return None
        
        # Record model usage (simple informative print)
        print(f"Model mapping recorded: {table_name}.{column_name} -> {model_key}")

        print(f"✅ Vector indexing complete for table '{table_name}', column '{column_name}'")
        print(f"   - Processed {len(embeddings_data)} records")
        print(f"   - Embeddings stored in column '{vector_column_name}'")
        print(f"   - Model used: {model_key} ({model_info['model_name']})")
        print(f"   - Dimensions: {model_info['dimension']}")

        return {
            'table': table_name,
            'column': column_name,
            'vector_column': vector_column_name,
            'records_processed': len(embeddings_data),
            'model_key': model_key,
            'model_name': model_info['model_name'],
            'dimensions': model_info['dimension'],
            'status': 'complete'
        }
    
    def __del__(self):
        """Clean up database connection when object is destroyed."""
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            try:
                self.db_connection.close()
            except:
                pass  # Ignore errors during cleanup
