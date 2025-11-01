#!/usr/bin/env python3
"""
Sample data verification script for MariaDB RAG Magic Commands Demo

This script verifies that the sample movie data meets the requirements:
- 50+ movie records
- Variety of genres for interesting search results
- Good data quality for demo experience
"""

import mariadb
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_CONFIG

def verify_sample_data():
    """Verify sample data meets requirements"""
    try:
        conn = mariadb.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("Sample Data Verification Report")
        print("=" * 50)
        
        # Check total count (requirement: 50+ records)
        cursor.execute("SELECT COUNT(*) FROM demo_content")
        total_count = cursor.fetchone()[0]
        print(f"Total movie records: {total_count}")
        
        if total_count >= 50:
            print("✓ Meets requirement: 50+ records")
        else:
            print("✗ Does not meet requirement: needs 50+ records")
        
        # Check genre variety
        cursor.execute("SELECT genre, COUNT(*) FROM demo_content WHERE genre IS NOT NULL GROUP BY genre ORDER BY COUNT(*) DESC")
        genres = cursor.fetchall()
        
        print(f"\nGenre distribution ({len(genres)} genres):")
        for genre, count in genres:
            print(f"  {genre}: {count} movies")
        
        if len(genres) >= 8:
            print("✓ Good genre variety for interesting search results")
        else:
            print("✗ Limited genre variety")
        
        # Check data quality
        print("\nData Quality Checks:")
        
        # Check for empty titles
        cursor.execute("SELECT COUNT(*) FROM demo_content WHERE title IS NULL OR title = ''")
        empty_titles = cursor.fetchone()[0]
        print(f"Empty titles: {empty_titles}")
        
        # Check for short content (less than 50 characters)
        cursor.execute("SELECT COUNT(*) FROM demo_content WHERE CHAR_LENGTH(content) < 50")
        short_content = cursor.fetchone()[0]
        print(f"Short content (< 50 chars): {short_content}")
        
        # Check for missing years
        cursor.execute("SELECT COUNT(*) FROM demo_content WHERE year IS NULL")
        missing_years = cursor.fetchone()[0]
        print(f"Missing years: {missing_years}")
        
        # Show sample records for manual quality check
        print("\nSample Records (for manual quality review):")
        cursor.execute("SELECT title, genre, year, SUBSTRING(content, 1, 100) as content_preview FROM demo_content ORDER BY RAND() LIMIT 5")
        samples = cursor.fetchall()
        
        for i, (title, genre, year, content_preview) in enumerate(samples, 1):
            print(f"\n{i}. {title} ({genre}, {year})")
            print(f"   Content: {content_preview}...")
        
        # Check for potential search variety
        print("\nSearch Variety Analysis:")
        
        # Keywords that should appear in content for good search results
        search_keywords = [
            ('space', 'space OR galaxy OR planet OR alien'),
            ('love', 'love OR romance OR relationship'),
            ('action', 'action OR fight OR chase OR battle'),
            ('comedy', 'comedy OR funny OR humor'),
            ('thriller', 'thriller OR suspense OR mystery'),
            ('family', 'family OR children OR kid')
        ]
        
        for keyword, search_term in search_keywords:
            cursor.execute(f"SELECT COUNT(*) FROM demo_content WHERE content LIKE '%{keyword}%' OR genre LIKE '%{keyword}%'")
            count = cursor.fetchone()[0]
            print(f"  Movies mentioning '{keyword}': {count}")
        
        # Overall assessment
        print("\n" + "=" * 50)
        quality_score = 0
        
        if total_count >= 50:
            quality_score += 25
        if len(genres) >= 8:
            quality_score += 25
        if empty_titles == 0:
            quality_score += 20
        if short_content <= 2:  # Allow a couple of short descriptions
            quality_score += 15
        if missing_years <= 5:  # Allow some missing years
            quality_score += 15
        
        print(f"Data Quality Score: {quality_score}/100")
        
        if quality_score >= 80:
            print("✓ Excellent data quality for demo")
        elif quality_score >= 60:
            print("✓ Good data quality for demo")
        else:
            print("✗ Data quality needs improvement")
        
        cursor.close()
        conn.close()
        
        return quality_score >= 60
        
    except mariadb.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def suggest_demo_queries():
    """Suggest good demo queries based on the data"""
    print("\nSuggested Demo Queries:")
    print("-" * 30)
    
    queries = [
        "space adventure movies",
        "romantic comedies",
        "psychological thrillers", 
        "movies about artificial intelligence",
        "films with time travel",
        "superhero action movies",
        "animated family films",
        "western movies",
        "movies about friendship",
        "dystopian future films"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"{i:2d}. {query}")
    
    print("\nSample RAG Questions:")
    print("-" * 20)
    
    rag_questions = [
        "What are the best sci-fi movies for beginners?",
        "Recommend some movies about space exploration",
        "What are good romantic movies from the 1990s?",
        "Which movies feature artificial intelligence themes?",
        "What are some critically acclaimed animated films?",
        "Suggest action movies with great choreography",
        "What are classic western films I should watch?",
        "Which movies explore themes of identity and reality?"
    ]
    
    for i, question in enumerate(rag_questions, 1):
        print(f"{i:2d}. {question}")

if __name__ == "__main__":
    success = verify_sample_data()
    suggest_demo_queries()
    
    if success:
        print("\n✓ Sample data verification passed!")
    else:
        print("\n✗ Sample data verification failed!")
    
    sys.exit(0 if success else 1)