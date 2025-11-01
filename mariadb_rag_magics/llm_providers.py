"""
LLM Provider Abstraction for RAG Query Magic Commands

This module provides a unified interface for different LLM providers
including Ollama and HuggingFace Transformers.
"""

import requests
import json
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the LLM provider. Returns True if successful."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from prompt. Returns None if failed."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready."""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the provider."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local server-based models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama2')
        self.timeout = config.get('timeout', 30)
    
    def initialize(self) -> bool:
        """Initialize Ollama connection."""
        try:
            self.is_initialized = self.is_available()
            return self.is_initialized
        except Exception:
            self.is_initialized = False
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model.get('name', '').split(':')[0] for model in models]
            return any(self.model.startswith(name) for name in model_names if name)
            
        except Exception:
            return False
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using Ollama API."""
        if not self.is_initialized and not self.initialize():
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return None
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get Ollama provider information."""
        return {
            'provider': 'Ollama',
            'model': self.model,
            'base_url': self.base_url,
            'requires_server': 'Yes',
            'status': 'Available' if self.is_available() else 'Not Available'
        }


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers LLM provider for direct model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('default_model', 'google/flan-t5-base')
        self.device = config.get('device', 'auto')
        self.max_length = config.get('max_length', 512)
        self.temperature = config.get('temperature', 0.7)
        self.do_sample = config.get('do_sample', True)
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def initialize(self) -> bool:
        """Initialize HuggingFace model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library not available. Install with: pip install transformers")
            return False
        
        try:
            print(f"Loading HuggingFace model: {self.model_name}")
            
            # Determine device
            if self.device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = self.device
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine if this is a T5 model (text-to-text) or causal LM
            self.is_t5_model = 't5' in self.model_name.lower() or 'flan' in self.model_name.lower()
            
            # Create appropriate pipeline based on model type
            if self.is_t5_model:
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            self.is_initialized = True
            print("✅ HuggingFace model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            self.is_initialized = False
            return False
    
    def is_available(self) -> bool:
        """Check if HuggingFace transformers is available."""
        return TRANSFORMERS_AVAILABLE and self.is_initialized
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using HuggingFace model."""
        if not self.is_initialized and not self.initialize():
            return None

        try:
            # For T5 models, format prompt for text-to-text generation
            if self.is_t5_model:
                # T5 models work better with instruction-style prompts
                formatted_prompt = f"Answer the following question: {prompt}"
                
                # Generate response using text2text pipeline
                outputs = self.pipeline(
                    formatted_prompt,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    num_return_sequences=1
                )
                
                if outputs and len(outputs) > 0:
                    response = outputs[0]['generated_text'].strip()
                    return response if response else "No response generated."
            else:
                # For causal language models (GPT-style)
                # Truncate prompt if too long
                max_prompt_length = self.max_length - 100  # Leave room for generation
                inputs = self.tokenizer.encode(prompt, truncation=True, max_length=max_prompt_length)
                prompt_truncated = self.tokenizer.decode(inputs, skip_special_tokens=True)
                
                # Generate response
                outputs = self.pipeline(
                    prompt_truncated,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0]['generated_text']
                    # Remove the original prompt from the response
                    if generated_text.startswith(prompt_truncated):
                        response = generated_text[len(prompt_truncated):].strip()
                    else:
                        response = generated_text.strip()
                    
                    return response if response else "No response generated."
            
            return None
            
        except Exception as e:
            print(f"Error generating HuggingFace response: {e}")
            return None
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get HuggingFace provider information."""
        return {
            'provider': 'HuggingFace',
            'model': self.model_name,
            'device': str(self.device),
            'requires_server': 'No',
            'status': 'Available' if self.is_available() else 'Not Available'
        }


class LLMProviderFactory:
    """Factory class for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> Optional[BaseLLMProvider]:
        """Create LLM provider instance."""
        if provider_type.lower() == 'ollama':
            return OllamaProvider(config)
        elif provider_type.lower() == 'huggingface':
            return HuggingFaceProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get list of available providers."""
        return {
            'ollama': True,  # Always available (depends on server)
            'huggingface': TRANSFORMERS_AVAILABLE
        }
