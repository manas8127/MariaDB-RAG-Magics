#!/usr/bin/env python3
"""
End-to-End RAG Workflow Test Script

This script tests the complete RAG workflow to ensure all components work together:
1. Vector indexing on sample data
2. Semantic search with various queries
3. RAG query responses with context validation

Run this before the demo to verify everything works correctly.
"""

import sys
import os
import time
import traceback
import importlib.util

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'mariadb',
        'sentence_transformers', 
        'requests',
        'numpy',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n🔧 Install missing packages with:")
        print("   pip install mariadb sentence-transformers requests numpy torch")
        return False
    
    print("✅ All required packages are available")
    return True

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check dependencies first
if not check_dependencies():
    print("\n⚠️  Cannot run full tests without dependencies.")
    print("   Run 'pip install -r requirements.txt' first.")
    sys.exit(1)

try:
    from mariadb_rag_magics.vector_index_magic import VectorIndexMagic
    from mariadb_rag_magics.semantic_search_magic import SemanticSearchMagic
    from mariadb_rag_magics.rag_query_magic import RagQueryMagic
except ImportError as e:
    print(f"❌ Error importing magic commands: {e}")
    print("   Make sure the mariadb_rag_magics package is properly installed.")
    sys.exit(1)


class RAGWorkflowTester:
    """Test harness for the complete RAG workflow."""
    
    def __init__(self):
        self.vector_magic = VectorIndexMagic()
        self.search_magic = SemanticSearchMagic()
        self.rag_magic = RagQueryMagic()
        self.test_results = []
    
    def log_test(self, test_name, success, message=""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        print()
    
    def test_database_connection(self):
        """Test database connectivity."""
        print("🔍 Testing database connection...")
        try:
            conn = self.vector_magic.get_db_connection()
            if conn is None:
                self.log_test("Database Connection", False, "Cannot connect to MariaDB")
                return False
            
            # Test if demo_content table exists
            if not self.vector_magic.check_table_exists("demo_content"):
                self.log_test("Database Connection", False, "demo_content table not found")
                return False
            
            self.log_test("Database Connection", True, "Connected to MariaDB with demo_content table")
            return True
            
        except Exception as e:
            self.log_test("Database Connection", False, f"Error: {str(e)}")
            return False
    
    def test_vector_indexing(self):
        """Test vector index creation."""
        print("🔍 Testing vector indexing...")
        try:
            # Test vector index creation
            result = self.vector_magic.vector_index("demo_content content")
            
            if result is None:
                self.log_test("Vector Indexing", False, "Vector indexing returned None")
                return False
            
            if result.get('status') != 'complete':
                self.log_test("Vector Indexing", False, f"Unexpected status: {result.get('status')}")
                return False
            
            records_processed = result.get('records_processed', 0)
            if records_processed == 0:
                self.log_test("Vector Indexing", False, "No records were processed")
                return False
            
            self.log_test("Vector Indexing", True, f"Processed {records_processed} records successfully")
            return True
            
        except Exception as e:
            self.log_test("Vector Indexing", False, f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        print("🔍 Testing semantic search...")
        
        test_queries = [
            "space adventure",
            "artificial intelligence",
            "romantic comedy",
            "psychological thriller"
        ]
        
        successful_searches = 0
        
        for query in test_queries:
            try:
                print(f"  Testing query: '{query}'")
                result = self.search_magic.semantic_search(f'demo_content "{query}"')
                
                if result is None:
                    print(f"    ❌ Query '{query}' returned None")
                    continue
                
                if result.get('status') not in ['complete', 'no_results']:
                    print(f"    ❌ Query '{query}' unexpected status: {result.get('status')}")
                    continue
                
                results_count = len(result.get('results', []))
                print(f"    ✅ Query '{query}' returned {results_count} results")
                successful_searches += 1
                
            except Exception as e:
                print(f"    ❌ Query '{query}' failed: {str(e)}")
                continue
        
        if successful_searches == len(test_queries):
            self.log_test("Semantic Search", True, f"All {len(test_queries)} test queries successful")
            return True
        else:
            self.log_test("Semantic Search", False, 
                         f"Only {successful_searches}/{len(test_queries)} queries successful")
            return False
    
    def test_rag_queries(self):
        """Test RAG query functionality."""
        print("🔍 Testing RAG queries...")
        
        test_questions = [
            "What are good sci-fi movies for beginners?",
            "Recommend a romantic movie for date night",
            "What movies have artificial intelligence themes?"
        ]
        
        successful_queries = 0
        
        for question in test_questions:
            try:
                print(f"  Testing question: '{question[:50]}...'")
                
                # Simulate cell magic call
                result = self.rag_magic.rag_query("demo_content", question)
                
                if result is None:
                    print(f"    ❌ Question failed - returned None")
                    continue
                
                if result.get('status') != 'complete':
                    print(f"    ❌ Question failed - status: {result.get('status')}")
                    continue
                
                answer = result.get('answer', '')
                context_records = result.get('context_records', [])
                
                if not answer or len(answer.strip()) < 10:
                    print(f"    ❌ Question failed - empty or too short answer")
                    continue
                
                if len(context_records) == 0:
                    print(f"    ⚠️  Question succeeded but no context records found")
                
                print(f"    ✅ Question succeeded - answer length: {len(answer)} chars, "
                      f"context records: {len(context_records)}")
                successful_queries += 1
                
            except Exception as e:
                print(f"    ❌ Question failed: {str(e)}")
                continue
        
        if successful_queries == len(test_questions):
            self.log_test("RAG Queries", True, f"All {len(test_questions)} test questions successful")
            return True
        else:
            self.log_test("RAG Queries", False, 
                         f"Only {successful_queries}/{len(test_questions)} questions successful")
            return False
    
    def test_ollama_connection(self):
        """Test Ollama server connectivity."""
        print("🔍 Testing Ollama connection...")
        try:
            if self.rag_magic.check_ollama_connection():
                self.log_test("Ollama Connection", True, "Ollama server is accessible")
                return True
            else:
                self.log_test("Ollama Connection", False, 
                             "Ollama server not accessible - check if 'ollama serve' is running")
                return False
                
        except Exception as e:
            self.log_test("Ollama Connection", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting RAG Workflow End-to-End Tests")
        print("=" * 60)
        print()
        
        start_time = time.time()
        
        # Run tests in order
        tests = [
            ("Database Connection", self.test_database_connection),
            ("Ollama Connection", self.test_ollama_connection),
            ("Vector Indexing", self.test_vector_indexing),
            ("Semantic Search", self.test_semantic_search),
            ("RAG Queries", self.test_rag_queries),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                else:
                    # If a critical test fails, we might want to skip dependent tests
                    if test_name in ["Database Connection", "Ollama Connection"]:
                        print(f"⚠️  Critical test '{test_name}' failed. Some subsequent tests may fail.")
                        print()
            except Exception as e:
                self.log_test(test_name, False, f"Unexpected error: {str(e)}")
                traceback.print_exc()
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Duration: {duration:.1f} seconds")
        print()
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! RAG workflow is ready for demo.")
            print()
            print("✅ You can now run the demo notebook with confidence!")
            print("✅ All magic commands are working correctly")
            print("✅ Database and Ollama connections are stable")
            return True
        else:
            print("❌ SOME TESTS FAILED! Please fix issues before demo.")
            print()
            print("🔧 Check the following:")
            print("   - MariaDB server is running with Vector extension")
            print("   - demo_content table is loaded with sample data")
            print("   - Ollama server is running ('ollama serve')")
            print("   - Required Python dependencies are installed")
            print("   - Network connectivity is working")
            return False


def main():
    """Main test execution."""
    print("🎬 MariaDB RAG Magic Commands - End-to-End Test")
    print("Testing complete workflow before hackathon demo")
    print()
    
    tester = RAGWorkflowTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()