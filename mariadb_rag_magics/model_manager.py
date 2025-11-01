"""
Embedding Model Management System

Handles multiple embedding models, tracks which model is used for each table,
and ensures consistency across vector indexing, semantic search, and RAG queries.
"""

import mariadb
import json
from typing import Dict, Optional, Tuple
from config import DB_CONFIG, AVAILABLE_EMBEDDING_MODELS


class ModelManager:
    """
    Manages embedding models and their association with database tables.
    Ensures consistency across all RAG operations.
    """
    
    def __init__(self):
        self.db_connection = None
        self._ensure_metadata_table()
    
    def get_db_connection(self):
        """Get database connection with error handling."""
        if self.db_connection is None:
            try:
                self.db_connection = mariadb.connect(**DB_CONFIG)
            except mariadb.Error as e:
                print(f"Error connecting to database: {e}")
                return None
        return self.db_connection
    
    def _ensure_metadata_table(self):
        """Create the model metadata table if it doesn't exist."""
        conn = self.get_db_connection()
        if conn is None:
            return
        
        try:
            cursor = conn.cursor()
            
            # Create metadata table to track which model is used for each table/column
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS rag_model_metadata (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    table_name VARCHAR(255) NOT NULL,
                    column_name VARCHAR(255) NOT NULL,
                    vector_column VARCHAR(255) NOT NULL,
                    model_key VARCHAR(100) NOT NULL,
                    model_name VARCHAR(500) NOT NULL,
                    model_dimension INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_table_column (table_name, column_name)
                )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            cursor.close()
            
        except mariadb.Error as e:
            print(f"Error creating metadata table: {e}")
    
    def get_available_models(self) -> Dict:
        """Get list of available embedding models."""
        return AVAILABLE_EMBEDDING_MODELS
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return AVAILABLE_EMBEDDING_MODELS.get(model_key)
    
    def register_table_model(self, table_name: str, column_name: str, 
                           vector_column: str, model_key: str) -> bool:
        """
        Register which embedding model is used for a specific table/column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the text column
            vector_column: Name of the vector column
            model_key: Key of the embedding model used
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_db_connection()
        if conn is None:
            return False
        
        model_info = self.get_model_info(model_key)
        if model_info is None:
            print(f"Error: Unknown model key '{model_key}'")
            return False
        
        try:
            cursor = conn.cursor()
            
            # Insert or update the model metadata
            sql = """
                INSERT INTO rag_model_metadata 
                (table_name, column_name, vector_column, model_key, model_name, model_dimension)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                vector_column = VALUES(vector_column),
                model_key = VALUES(model_key),
                model_name = VALUES(model_name),
                model_dimension = VALUES(model_dimension),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(sql, (
                table_name, column_name, vector_column, 
                model_key, model_info['model_name'], model_info['dimension']
            ))
            
            conn.commit()
            cursor.close()
            
            print(f"✅ Registered model '{model_key}' for {table_name}.{column_name}")
            return True
            
        except mariadb.Error as e:
            print(f"Error registering model metadata: {e}")
            return False
    
    def get_table_model(self, table_name: str, column_name: str = None) -> Optional[Tuple[str, Dict]]:
        """
        Get the embedding model used for a specific table/column.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column (optional, will use first found if not specified)
            
        Returns:
            Tuple of (model_key, model_info) or None if not found
        """
        conn = self.get_db_connection()
        if conn is None:
            return None
        
        try:
            cursor = conn.cursor()
            
            if column_name:
                sql = """
                    SELECT model_key, model_name, model_dimension, vector_column
                    FROM rag_model_metadata 
                    WHERE table_name = %s AND column_name = %s
                """
                cursor.execute(sql, (table_name, column_name))
            else:
                # Get first model for this table
                sql = """
                    SELECT model_key, model_name, model_dimension, vector_column
                    FROM rag_model_metadata 
                    WHERE table_name = %s
                    LIMIT 1
                """
                cursor.execute(sql, (table_name,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                model_key, model_name, model_dimension, vector_column = result
                
                # Get full model info from config
                model_info = self.get_model_info(model_key)
                if model_info:
                    # Add vector column info
                    model_info = model_info.copy()
                    model_info['vector_column'] = vector_column
                    return model_key, model_info
            
            return None
            
        except mariadb.Error as e:
            print(f"Error retrieving model metadata: {e}")
            return None
    
    def list_table_models(self) -> Dict:
        """List all tables and their associated models."""
        conn = self.get_db_connection()
        if conn is None:
            return {}
        
        try:
            cursor = conn.cursor()
            
            sql = """
                SELECT table_name, column_name, vector_column, model_key, created_at
                FROM rag_model_metadata 
                ORDER BY table_name, column_name
            """
            
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            
            tables = {}
            for table_name, column_name, vector_column, model_key, created_at in results:
                if table_name not in tables:
                    tables[table_name] = {}
                
                tables[table_name][column_name] = {
                    'vector_column': vector_column,
                    'model_key': model_key,
                    'model_info': self.get_model_info(model_key),
                    'created_at': created_at
                }
            
            return tables
            
        except mariadb.Error as e:
            print(f"Error listing table models: {e}")
            return {}
    
    def validate_model_consistency(self, table_name: str, column_name: str, 
                                 model_key: str) -> bool:
        """
        Validate that the requested model matches the one used for indexing.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            model_key: Requested model key
            
        Returns:
            bool: True if consistent, False otherwise
        """
        existing_model = self.get_table_model(table_name, column_name)
        
        if existing_model is None:
            # No existing model, so any model is acceptable
            return True
        
        existing_model_key, _ = existing_model
        if existing_model_key != model_key:
            print(f"⚠️  Model mismatch detected!")
            print(f"   Table '{table_name}' was indexed with model '{existing_model_key}'")
            print(f"   But you're trying to use model '{model_key}'")
            print(f"   For consistent results, please use the same model.")
            return False
        
        return True
    
    def get_model_summary(self) -> str:
        """Get a formatted summary of available models and table assignments."""
        summary = ["🤖 EMBEDDING MODELS OVERVIEW"]
        summary.append("=" * 50)
        
        # Available models
        summary.append("\n📚 Available Models:")
        for key, info in AVAILABLE_EMBEDDING_MODELS.items():
            summary.append(f"   • {key}")
            summary.append(f"     Model: {info['model_name']}")
            summary.append(f"     Dimensions: {info['dimension']}")
            summary.append(f"     Use case: {info['use_case']}")
            summary.append("")
        
        # Table assignments
        table_models = self.list_table_models()
        if table_models:
            summary.append("🗄️ Table Model Assignments:")
            for table_name, columns in table_models.items():
                summary.append(f"   📊 {table_name}:")
                for column_name, info in columns.items():
                    summary.append(f"      └─ {column_name} → {info['model_key']} ({info['vector_column']})")
            summary.append("")
        else:
            summary.append("🗄️ No table model assignments found yet.")
            summary.append("")
        
        return "\n".join(summary)


# Global instance for use across magic commands
model_manager = ModelManager()
