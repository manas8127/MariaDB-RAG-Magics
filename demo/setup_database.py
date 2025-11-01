#!/usr/bin/env python3
"""
Database setup script for MariaDB RAG Magic Commands Demo

This script helps set up the demo database with proper schema and sample data.
It also verifies that MariaDB Vector extension is available.
"""

import mariadb
import sys
import os
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_CONFIG

def test_mariadb_connection():
    """Test basic MariaDB connection"""
    try:
        conn = mariadb.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print("✓ MariaDB connection successful")
        return conn
    except mariadb.Error as e:
        print(f"✗ MariaDB connection failed: {e}")
        return None

def test_vector_extension(conn):
    """Test if MariaDB Vector extension is available"""
    try:
        cursor = conn.cursor()
        # Test basic vector function
        cursor.execute("SELECT VEC_FromText('[0.1,0.2,0.3]') as test_vector")
        result = cursor.fetchone()
        if result:
            print("✓ MariaDB Vector extension is available")
            return True
    except mariadb.Error as e:
        print(f"✗ MariaDB Vector extension test failed: {e}")
        print("  Make sure you're using MariaDB 11.8+ with Vector extension installed")
        return False
    finally:
        cursor.close()
    return False

def create_database(conn):
    """Create the demo database"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        print(f"✓ Database '{DB_CONFIG['database']}' created/verified")
        cursor.close()
        return True
    except mariadb.Error as e:
        print(f"✗ Failed to create database: {e}")
        return False

def run_sql_file(conn, sql_file_path):
    """Execute SQL commands from file"""
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        cursor = conn.cursor()
        for statement in statements:
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                except mariadb.Error as e:
                    # Skip errors for statements that might already exist
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
        
        conn.commit()
        cursor.close()
        print(f"✓ SQL file '{sql_file_path}' executed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to execute SQL file: {e}")
        return False

def insert_sample_movies_manually(conn):
    """Insert sample movie data manually if SQL file insertion failed"""
    sample_movies = [
        ('Inception', 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O. Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state.', 'Sci-Fi', 2010),
        ('The Matrix', 'A computer programmer is led to fight an underground war against powerful computers who have constructed his entire reality with a system called the Matrix. Neo discovers that reality as he knows it is actually a computer simulation.', 'Sci-Fi', 1999),
        ('Interstellar', 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival. Cooper, a former NASA pilot, must leave his family behind to lead an expedition beyond our galaxy to discover whether mankind has a future among the stars.', 'Sci-Fi', 2014),
        ('Blade Runner 2049', 'A young blade runner discovery of a long-buried secret leads him to track down former blade runner Rick Deckard. Officer K unearths a secret that could plunge what left of society into chaos.', 'Sci-Fi', 2017),
        ('The Shawshank Redemption', 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A story of hope, friendship, and the human spirit triumph over adversity.', 'Drama', 1994),
        ('Mad Max: Fury Road', 'In a post-apocalyptic wasteland, Max teams up with Furiosa to flee from cult leader Immortan Joe and his army in an armored war rig. A high-octane chase across the desert with spectacular practical effects and minimal dialogue.', 'Action', 2015),
        ('The Princess Bride', 'A bedridden boy grandfather reads him the story of a farmboy-turned-pirate who encounters numerous obstacles, enemies and allies in his quest to be reunited with his true love. A perfect blend of romance, adventure, and comedy.', 'Romance', 1987),
        ('Get Out', 'A young African-American visits his white girlfriend family estate, where he learns that many of its black visitors have gone missing. Jordan Peele masterful social thriller disguised as horror.', 'Horror', 2017),
        ('Groundhog Day', 'A weatherman finds himself living the same day over and over again. Bill Murray delivers perfect comedic timing in this philosophical comedy about second chances and personal growth.', 'Comedy', 1993),
        ('The Lord of the Rings: The Fellowship of the Ring', 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron. Peter Jackson epic adaptation of Tolkien masterpiece.', 'Fantasy', 2001),
        ('Spirited Away', 'During her family move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts. Hayao Miyazaki masterpiece of imagination.', 'Animation', 2001),
        ('The Good, the Bad and the Ugly', 'A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery. Sergio Leone spaghetti western masterpiece.', 'Western', 1966),
        ('Se7en', 'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives. David Fincher dark thriller with an unforgettable ending.', 'Thriller', 1995),
        ('John Wick', 'An ex-hitman comes out of retirement to track down the gangsters that took everything from him. Keanu Reeves delivers intense action sequences in this stylish revenge thriller with incredible choreography.', 'Action', 2014),
        ('Forrest Gump', 'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man with an IQ of 75. Tom Hanks delivers a heartwarming performance in this American epic.', 'Drama', 1994),
        ('Avatar', 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home. Jake Sully must choose between his human origins and the alien world he has come to love.', 'Sci-Fi', 2009),
        ('The Dark Knight', 'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy. Heath Ledger iconic performance elevates this superhero film to new heights of psychological complexity.', 'Action', 2008),
        ('Pulp Fiction', 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption. Quentin Tarantino nonlinear narrative revolutionized modern cinema.', 'Drama', 1994),
        ('Titanic', 'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic. James Cameron epic romance set against historical tragedy.', 'Romance', 1997),
        ('A Quiet Place', 'A family lives in silence while hiding from creatures that hunt by sound. John Krasinski creates tension through minimal dialogue and maximum suspense in this innovative horror film.', 'Horror', 2018)
    ]
    
    try:
        cursor = conn.cursor()
        
        # Clear any existing data first
        cursor.execute("DELETE FROM demo_content")
        print("  Cleared existing data")
        
        # Insert sample movies
        cursor.executemany("""
        INSERT INTO demo_content (title, content, genre, year) 
        VALUES (%s, %s, %s, %s)
        """, sample_movies)
        
        conn.commit()
        cursor.close()
        
        print(f"✓ Manually inserted {len(sample_movies)} sample movies")
        return True
        
    except mariadb.Error as e:
        print(f"✗ Failed to insert sample movies manually: {e}")
        return False

def verify_sample_data(conn):
    """Verify that sample data was loaded correctly"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")
        cursor.execute("SELECT COUNT(*) FROM demo_content")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT genre) FROM demo_content WHERE genre IS NOT NULL")
        genre_count = cursor.fetchone()[0]
        
        print(f"✓ Sample data loaded: {count} movies across {genre_count} genres")
        
        # Show sample of data
        if count > 0:
            cursor.execute("SELECT title, genre FROM demo_content LIMIT 5")
            samples = cursor.fetchall()
            print("  Sample movies:")
            for title, genre in samples:
                print(f"    - {title} ({genre})")
        
        cursor.close()
        return count > 0
        
    except mariadb.Error as e:
        print(f"✗ Failed to verify sample data: {e}")
        return False

def main():
    """Main setup function"""
    print("MariaDB RAG Demo Database Setup")
    print("=" * 40)
    
    # Test connection
    conn = test_mariadb_connection()
    if not conn:
        print("\nSetup failed. Please check your MariaDB configuration.")
        return False
    
    # Test vector extension
    if not test_vector_extension(conn):
        print("\nSetup failed. Vector extension is required.")
        conn.close()
        return False
    
    # Create database
    if not create_database(conn):
        conn.close()
        return False
    
    # Connect to the specific database
    conn.close()
    try:
        conn = mariadb.connect(**DB_CONFIG)
        print(f"✓ Connected to database '{DB_CONFIG['database']}'")
    except mariadb.Error as e:
        print(f"✗ Failed to connect to demo database: {e}")
        return False
    
    # Run SQL setup file
    sql_file = Path(__file__).parent / "sample_data.sql"
    if not run_sql_file(conn, sql_file):
        conn.close()
        return False
    
    # Verify setup and auto-fix if needed
    if not verify_sample_data(conn):
        print("⚠️  SQL file didn't insert sample data properly. Trying manual insertion...")
        if not insert_sample_movies_manually(conn):
            print("✗ Failed to insert sample data both ways. Setup incomplete.")
            conn.close()
            return False
        
        # Verify again after manual insertion
        if not verify_sample_data(conn):
            print("✗ Sample data verification failed even after manual insertion.")
            conn.close()
            return False
        
        print("✓ Sample data fixed using manual insertion!")
    
    conn.close()
    print("\n" + "=" * 40)
    print("✓ Database setup completed successfully!")
    print("\nYou can now run the demo notebook or use the magic commands.")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"Table: demo_content")
    print("Next steps:")
    print("1. Start Jupyter notebook")
    print("2. Load the magic commands: %load_ext mariadb_rag_magics")
    print("3. Create vector index: %vector_index demo_content content")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
