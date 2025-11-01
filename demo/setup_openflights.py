#!/usr/bin/env python3
"""
OpenFlights Data Setup Script

This script sets up the OpenFlights airport dataset for the MariaDB RAG demo.
It creates the airports table and loads sample airport data with rich descriptions
for better RAG demonstrations.
"""

import mariadb
import sys
import os

# Add the parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONFIG

def setup_openflights_database():
    """Set up the OpenFlights airport database with sample data."""
    
    print("🛫 Setting up OpenFlights Dataset for MariaDB RAG Demo")
    print("="*70)
    
    try:
        # Connect to MariaDB
        print("📡 Connecting to MariaDB...")
        conn = mariadb.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Use the correct database
        cursor.execute(f"USE {DB_CONFIG['database']}")
        print(f"✅ Connected to database: {DB_CONFIG['database']}")
        
        # Load the OpenFlights SQL file
        sql_file = os.path.join(os.path.dirname(__file__), 'openflights_data.sql')
        
        print("📊 Loading OpenFlights airport data...")
        
        # Read and execute the SQL file
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split by statements and execute each one
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement.strip():
                cursor.execute(statement)
        
        conn.commit()
        
        # Verify the setup
        cursor.execute("SELECT COUNT(*) FROM airports")
        airport_count = cursor.fetchone()[0]
        print(f"✅ Loaded {airport_count} airports successfully")
        
        # Show sample data
        cursor.execute("""
            SELECT name, city, country, iata_code 
            FROM airports 
            ORDER BY id 
            LIMIT 5
        """)
        
        print("\n📍 Sample airports loaded:")
        for name, city, country, iata in cursor.fetchall():
            print(f"   • {name} ({iata}) - {city}, {country}")
        
        # Show distribution by country
        cursor.execute("""
            SELECT country, COUNT(*) as count 
            FROM airports 
            GROUP BY country 
            ORDER BY count DESC 
            LIMIT 10
        """)
        
        print(f"\n🌍 Top countries by airport count:")
        for country, count in cursor.fetchall():
            print(f"   • {country}: {count} airports")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 OpenFlights database setup completed successfully!")
        print("\nNext steps:")
        print("1. Start Jupyter: jupyter notebook demo_notebook.ipynb")
        print("2. Load magic commands: %load_ext mariadb_rag_magics")
        print("3. Create vectors: %vector_index -of")
        print("4. Try search: %semantic_search -of \"mountain airports\"")
        print("5. Ask questions: %%rag_query -of")
        print("                  Which airports are best for skiing?")
        
    except mariadb.Error as e:
        print(f"❌ MariaDB Error: {e}")
        print("\nTroubleshooting:")
        print("- Check if MariaDB server is running")
        print("- Verify database credentials in config.py")
        print("- Ensure database 'rag_demo' exists")
        sys.exit(1)
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find openflights_data.sql")
        print(f"   Expected location: {sql_file}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def validate_openflights_setup():
    """Validate that the OpenFlights setup was successful."""
    
    print("\n🔍 Validating OpenFlights setup...")
    
    try:
        conn = mariadb.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Check if airports table exists
        cursor.execute("SHOW TABLES LIKE 'airports'")
        if not cursor.fetchone():
            print("❌ Airports table not found")
            return False
        
        # Check if table has data
        cursor.execute("SELECT COUNT(*) FROM airports")
        count = cursor.fetchone()[0]
        if count == 0:
            print("❌ Airports table is empty")
            return False
        
        # Check if description column has content
        cursor.execute("SELECT COUNT(*) FROM airports WHERE description IS NOT NULL AND description != ''")
        desc_count = cursor.fetchone()[0]
        if desc_count == 0:
            print("❌ No airport descriptions found")
            return False
        
        print(f"✅ Validation passed: {count} airports with {desc_count} descriptions")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("OpenFlights Database Setup")
    print("This will set up the global airport dataset for RAG demonstrations.")
    print()
    
    # Run setup
    setup_openflights_database()
    
    # Validate setup
    if validate_openflights_setup():
        print("\n🎯 Setup validation successful!")
        print("The OpenFlights dataset is ready for RAG demonstrations.")
    else:
        print("\n⚠️  Setup validation failed. Please check the errors above.")
        sys.exit(1)
