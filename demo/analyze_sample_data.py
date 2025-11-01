#!/usr/bin/env python3
"""
Analyze sample data from SQL file without requiring database connection

This script parses the SQL file to verify data quality and variety.
"""

import re
from pathlib import Path
from collections import Counter

def parse_sql_inserts(sql_file_path):
    """Parse INSERT statements from SQL file"""
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find INSERT statements
    insert_pattern = r"INSERT INTO demo_content \(title, content, genre, year\) VALUES\s*(.*?);"
    matches = re.findall(insert_pattern, content, re.DOTALL)
    
    if not matches:
        return []
    
    # Parse individual value tuples
    values_text = matches[0]
    
    # Split by lines and parse each movie entry
    movies = []
    current_entry = ""
    paren_count = 0
    
    for line in values_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('--'):
            continue
            
        current_entry += line + " "
        paren_count += line.count('(') - line.count(')')
        
        # If we have a complete entry (balanced parentheses ending with comma or end)
        if paren_count == 0 and (line.endswith(',') or line.endswith(');')):
            # Parse the entry
            entry = current_entry.strip().rstrip(',').rstrip(');')
            if entry.startswith('('):
                entry = entry[1:]  # Remove leading (
            if entry.endswith(')'):
                entry = entry[:-1]  # Remove trailing )
            
            # Split by comma, but be careful with quoted strings
            parts = []
            current_part = ""
            in_quotes = False
            quote_char = None
            
            i = 0
            while i < len(entry):
                char = entry[i]
                if char in ('"', "'") and (i == 0 or entry[i-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                    i += 1
                    continue
                
                current_part += char
                i += 1
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            if len(parts) >= 4:
                title = parts[0].strip('\'"')
                content = parts[1].strip('\'"')
                genre = parts[2].strip('\'"')
                year = parts[3].strip('\'"')
                
                try:
                    year = int(year) if year.isdigit() else None
                except:
                    year = None
                
                movies.append({
                    'title': title,
                    'content': content,
                    'genre': genre,
                    'year': year
                })
            
            current_entry = ""
    
    return movies

def analyze_data_quality(movies):
    """Analyze the quality and variety of movie data"""
    print("Sample Data Analysis Report")
    print("=" * 50)
    
    total_count = len(movies)
    print(f"Total movie records: {total_count}")
    
    # Check if meets 50+ requirement
    if total_count >= 50:
        print("✓ Meets requirement: 50+ records")
    else:
        print("✗ Does not meet requirement: needs 50+ records")
    
    # Analyze genres
    genres = Counter(movie['genre'] for movie in movies if movie['genre'])
    print(f"\nGenre distribution ({len(genres)} genres):")
    for genre, count in genres.most_common():
        print(f"  {genre}: {count} movies")
    
    if len(genres) >= 8:
        print("✓ Excellent genre variety for interesting search results")
    elif len(genres) >= 5:
        print("✓ Good genre variety")
    else:
        print("✗ Limited genre variety")
    
    # Analyze years
    years = [movie['year'] for movie in movies if movie['year']]
    if years:
        year_range = f"{min(years)}-{max(years)}"
        print(f"\nYear range: {year_range} ({len(years)} movies with years)")
    
    # Content quality checks
    print("\nContent Quality Analysis:")
    
    # Check content length
    content_lengths = [len(movie['content']) for movie in movies]
    avg_length = sum(content_lengths) / len(content_lengths)
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    
    print(f"Content length - Avg: {avg_length:.0f}, Min: {min_length}, Max: {max_length}")
    
    short_content = sum(1 for length in content_lengths if length < 50)
    print(f"Short content (< 50 chars): {short_content}")
    
    # Check for variety in content
    keywords_found = {}
    search_keywords = ['space', 'love', 'action', 'comedy', 'thriller', 'family', 'war', 'crime', 'adventure']
    
    for keyword in search_keywords:
        count = sum(1 for movie in movies if keyword.lower() in movie['content'].lower() or keyword.lower() in movie['genre'].lower())
        keywords_found[keyword] = count
    
    print(f"\nKeyword variety (for search testing):")
    for keyword, count in keywords_found.items():
        print(f"  '{keyword}': {count} movies")
    
    # Sample movies for manual review
    print(f"\nSample Movies (first 5):")
    for i, movie in enumerate(movies[:5], 1):
        content_preview = movie['content'][:80] + "..." if len(movie['content']) > 80 else movie['content']
        print(f"{i}. {movie['title']} ({movie['genre']}, {movie['year']})")
        print(f"   {content_preview}")
    
    # Quality score
    quality_score = 0
    
    if total_count >= 50:
        quality_score += 30
    elif total_count >= 30:
        quality_score += 20
    
    if len(genres) >= 8:
        quality_score += 25
    elif len(genres) >= 5:
        quality_score += 15
    
    if short_content <= 2:
        quality_score += 20
    elif short_content <= 5:
        quality_score += 10
    
    if avg_length >= 100:
        quality_score += 15
    elif avg_length >= 50:
        quality_score += 10
    
    variety_score = sum(1 for count in keywords_found.values() if count > 0)
    if variety_score >= 7:
        quality_score += 10
    elif variety_score >= 5:
        quality_score += 5
    
    print(f"\n" + "=" * 50)
    print(f"Data Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("✓ Excellent data quality for demo")
    elif quality_score >= 60:
        print("✓ Good data quality for demo")
    else:
        print("✗ Data quality needs improvement")
    
    return quality_score >= 60

def suggest_demo_scenarios(movies):
    """Suggest demo scenarios based on available data"""
    print(f"\nDemo Scenarios Based on Available Data:")
    print("-" * 40)
    
    genres = set(movie['genre'] for movie in movies if movie['genre'])
    
    print("Semantic Search Examples:")
    search_examples = [
        "space adventure movies",
        "romantic films", 
        "action thrillers",
        "animated movies",
        "psychological dramas",
        "superhero films",
        "western movies",
        "horror films"
    ]
    
    for example in search_examples:
        print(f"  %semantic_search demo_content \"{example}\"")
    
    print(f"\nRAG Query Examples:")
    rag_examples = [
        "What are the best sci-fi movies for beginners?",
        "Recommend some romantic movies from the 1990s",
        "Which movies feature space exploration themes?",
        "What are good action movies with great choreography?",
        "Suggest some animated films for family viewing"
    ]
    
    for example in rag_examples:
        print(f"  %%rag_query demo_content")
        print(f"  {example}")
        print()

if __name__ == "__main__":
    sql_file = Path(__file__).parent / "sample_data.sql"
    
    if not sql_file.exists():
        print(f"Error: {sql_file} not found")
        exit(1)
    
    print("Parsing sample data from SQL file...")
    movies = parse_sql_inserts(sql_file)
    
    if not movies:
        print("Error: No movie data found in SQL file")
        exit(1)
    
    success = analyze_data_quality(movies)
    suggest_demo_scenarios(movies)
    
    if success:
        print(f"\n✓ Sample data analysis passed! Ready for demo.")
    else:
        print(f"\n✗ Sample data needs improvement.")
    
    exit(0 if success else 1)