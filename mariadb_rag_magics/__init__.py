"""
MariaDB RAG Magic Commands for Jupyter

A collection of Jupyter magic commands that demonstrate RAG (Retrieval-Augmented Generation) 
workflows using MariaDB Vector with local models.
"""

__version__ = "0.1.0"
__author__ = "MariaDB RAG Demo"

from .vector_index_magic import VectorIndexMagic
from .semantic_search_magic import SemanticSearchMagic
from .rag_query_magic import RagQueryMagic

def load_ipython_extension(ipython):
    """Load the magic commands when the extension is loaded."""
    # Register all magic commands
    ipython.register_magic_function(VectorIndexMagic(ipython).vector_index, 'line', 'vector_index')
    ipython.register_magic_function(SemanticSearchMagic(ipython).semantic_search, 'line', 'semantic_search')
    ipython.register_magic_function(RagQueryMagic(ipython).rag_query, 'cell', 'rag_query')