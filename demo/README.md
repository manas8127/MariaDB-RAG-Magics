# MariaDB RAG Magic Commands Demo

This directory contains the demo database setup and sample data for the MariaDB RAG Magic Commands project.

## Demo Database Overview

The demo uses a movie database with 54+ carefully curated movie records across 10 different genres, providing excellent variety for demonstrating semantic search and RAG capabilities.

### Database Schema

**Table: `demo_content`**
- `id` - Primary key (auto-increment)
- `title` - Movie title (VARCHAR(255), NOT NULL)
- `content` - Movie description (TEXT, NOT NULL) 
- `genre` - Movie genre (VARCHAR(100))
- `year` - Release year (INT)
- `content_vector` - Vector embeddings (VECTOR(384))
- `created_at` - Timestamp (auto-generated)
- `updated_at` - Timestamp (auto-updated)

### Sample Data Statistics

- **54 movie records** (exceeds 50+ requirement)
- **10 genres**: Sci-Fi, Action, Drama, Comedy, Horror, Romance, Thriller, Fantasy, Animation, Western
- **Year range**: 1942-2018 (spanning 76 years)
- **Content quality**: Average 206 characters per description, all descriptions 50+ characters
- **Search variety**: Rich keyword coverage for diverse search testing

## Files

### Core Files
- `sample_data.sql` - Complete database schema and sample movie data
- `setup_database.py` - Automated database setup script with verification
- `demo_notebook.ipynb` - Jupyter notebook for demonstrating the magic commands

### Verification Scripts
- `verify_sample_data.py` - Database verification (requires MariaDB connection)
- `analyze_sample_data.py` - SQL file analysis (no database required)

## Quick Setup

### Prerequisites
- MariaDB 11.8+ with Vector extension
- Python 3.8+
- Required Python packages: `mariadb`, `sentence-transformers`, `requests`

### Setup Steps

1. **Start MariaDB** (with Vector extension enabled)
   ```bash
   # Using Docker (recommended for demo)
   docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=demo123 mariadb:11.8
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Run database setup**
   ```bash
   python setup_database.py
   ```

4. **Verify setup**
   ```bash
   python analyze_sample_data.py
   ```

### Manual Setup (Alternative)

If you prefer manual setup:

1. Connect to MariaDB and run:
   ```sql
   SOURCE sample_data.sql;
   ```

2. Verify the setup:
   ```sql
   USE rag_demo;
   SELECT COUNT(*) FROM demo_content;
   SELECT genre, COUNT(*) FROM demo_content GROUP BY genre;
   ```

## Demo Scenarios

### Semantic Search Examples
```python
%semantic_search demo_content "space adventure movies"
%semantic_search demo_content "romantic films"
%semantic_search demo_content "psychological thrillers"
%semantic_search demo_content "animated family movies"
```

### RAG Query Examples
```python
%%rag_query demo_content
What are the best sci-fi movies for beginners?

%%rag_query demo_content
Recommend some romantic movies from the 1990s

%%rag_query demo_content
Which movies feature space exploration themes?
```

## Data Quality Features

### Genre Distribution
- **Sci-Fi**: 10 movies (great for space/technology queries)
- **Action**: 5 movies (good for action/adventure searches)
- **Drama**: 5 movies (character-driven stories)
- **Comedy**: 5 movies (humor and entertainment)
- **Horror**: 5 movies (suspense and thriller elements)
- **Romance**: 5 movies (love and relationship themes)
- **Thriller**: 5 movies (psychological and suspense)
- **Fantasy**: 5 movies (magical and fantastical elements)
- **Animation**: 5 movies (family-friendly content)
- **Western**: 4 movies (classic American genre)

### Search Keyword Coverage
The sample data includes rich keyword variety:
- **Space themes**: 7 movies
- **Love/Romance**: 8 movies  
- **Action sequences**: 6 movies
- **Comedy elements**: 6 movies
- **Thriller aspects**: 7 movies
- **Family content**: 7 movies

## Troubleshooting

### Common Issues

1. **MariaDB Vector extension not available**
   - Ensure you're using MariaDB 11.8+
   - Verify Vector extension is installed and enabled

2. **Database connection failed**
   - Check MariaDB is running on localhost:3306
   - Verify credentials in `../config.py`
   - Default password is `demo123`

3. **Python dependencies missing**
   - Run: `pip install mariadb sentence-transformers requests`
   - For development: `pip install jupyter`

### Verification Commands

```bash
# Test database connection and data
python verify_sample_data.py

# Analyze data without database connection  
python analyze_sample_data.py

# Check MariaDB Vector extension
mysql -u root -p -e "SELECT VEC_FromText('[0.1,0.2,0.3]');"
```

## Next Steps

After successful setup:

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook demo_notebook.ipynb
   ```

2. **Load Magic Commands**
   ```python
   %load_ext mariadb_rag_magics
   ```

3. **Create Vector Index**
   ```python
   %vector_index demo_content content
   ```

4. **Test Semantic Search**
   ```python
   %semantic_search demo_content "space movies"
   ```

5. **Try RAG Query**
   ```python
   %%rag_query demo_content
   What are good sci-fi movies for someone new to the genre?
   ```

The demo database is now ready for showcasing MariaDB RAG capabilities!