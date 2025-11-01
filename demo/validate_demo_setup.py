#!/usr/bin/env python3
"""
Demo Setup Validation Script

This script validates that the demo components are properly structured
and ready for presentation, without requiring full database/Ollama setup.
"""

import os
import sys
import json
import importlib.util


def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False


def validate_notebook_structure(notebook_path):
    """Validate that the demo notebook has proper structure."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells = notebook.get('cells', [])
        if len(cells) < 10:
            print(f"⚠️  Notebook has only {len(cells)} cells - might be incomplete")
            return False
        
        # Check for key sections
        notebook_text = json.dumps(notebook).lower()
        required_sections = [
            'vector_index',
            'semantic_search', 
            'rag_query',
            'setup',
            'demo'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in notebook_text:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️  Notebook missing sections: {missing_sections}")
            return False
        
        print(f"✅ Notebook structure looks good ({len(cells)} cells)")
        return True
        
    except Exception as e:
        print(f"❌ Error validating notebook: {e}")
        return False


def check_package_structure():
    """Check that the magic commands package is properly structured."""
    package_dir = "mariadb_rag_magics"
    
    required_files = [
        "__init__.py",
        "vector_index_magic.py", 
        "semantic_search_magic.py",
        "rag_query_magic.py"
    ]
    
    all_good = True
    
    for filename in required_files:
        filepath = os.path.join(package_dir, filename)
        if not check_file_exists(filepath, f"Magic command: {filename}"):
            all_good = False
        elif filename.endswith('.py'):
            if not check_python_syntax(filepath):
                all_good = False
    
    return all_good


def check_demo_files():
    """Check that demo files are present and valid."""
    demo_files = [
        ("demo/demo_notebook.ipynb", "Demo Notebook"),
        ("demo/sample_data.sql", "Sample Data SQL"),
        ("demo/setup_database.py", "Database Setup Script"),
        ("config.py", "Configuration File"),
        ("requirements.txt", "Requirements File")
    ]
    
    all_good = True
    
    for filepath, description in demo_files:
        if not check_file_exists(filepath, description):
            all_good = False
        elif filepath.endswith('.py'):
            if not check_python_syntax(filepath):
                all_good = False
        elif filepath.endswith('.ipynb'):
            if not validate_notebook_structure(filepath):
                all_good = False
    
    return all_good


def check_dependencies_installable():
    """Check if required dependencies can be imported or are installable."""
    required_packages = [
        ('mariadb', 'MariaDB Python connector'),
        ('sentence_transformers', 'HuggingFace sentence transformers'),
        ('requests', 'HTTP requests library'),
        ('numpy', 'NumPy for numerical operations'),
        ('torch', 'PyTorch for ML models'),
        ('IPython', 'IPython for magic commands')
    ]
    
    available_packages = []
    missing_packages = []
    
    for package_name, description in required_packages:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            available_packages.append((package_name, description))
        else:
            missing_packages.append((package_name, description))
    
    print(f"\n📦 Package Status:")
    for package_name, description in available_packages:
        print(f"✅ {package_name}: {description}")
    
    for package_name, description in missing_packages:
        print(f"❌ {package_name}: {description} (needs installation)")
    
    if missing_packages:
        print(f"\n🔧 To install missing packages:")
        print("   pip install " + " ".join([pkg[0] for pkg in missing_packages]))
        return False
    
    return True


def main():
    """Main validation function."""
    print("🎬 MariaDB RAG Magic Commands - Demo Setup Validation")
    print("=" * 60)
    print()
    
    validation_steps = [
        ("Magic Commands Package", check_package_structure),
        ("Demo Files", check_demo_files),
        ("Dependencies", check_dependencies_installable)
    ]
    
    passed_validations = 0
    total_validations = len(validation_steps)
    
    for step_name, validation_func in validation_steps:
        print(f"\n🔍 Validating: {step_name}")
        print("-" * 40)
        
        try:
            if validation_func():
                print(f"✅ {step_name}: PASSED")
                passed_validations += 1
            else:
                print(f"❌ {step_name}: FAILED")
        except Exception as e:
            print(f"❌ {step_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Validations Passed: {passed_validations}/{total_validations}")
    print(f"Success Rate: {(passed_validations/total_validations)*100:.1f}%")
    
    if passed_validations == total_validations:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Demo structure is ready")
        print("✅ All required files are present")
        print("✅ Python syntax is valid")
        print("✅ Dependencies are available")
        print("\n🚀 Ready for demo presentation!")
        return True
    else:
        print(f"\n⚠️  {total_validations - passed_validations} VALIDATION(S) FAILED")
        print("🔧 Please fix the issues above before running the demo")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)