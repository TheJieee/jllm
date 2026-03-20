"""
Internal wrapper for the pybind11-exported JLLM engine.

This module wraps the native C++ bindings (_jllm_engine.so) and provides
a more Pythonic interface with proper error handling and documentation.
"""

import warnings
from typing import List, Tuple, Optional, Dict, Any

# Import the native pybind11-generated module
try:
    from . import jllm_engine
except ImportError as e:
    raise ImportError(
        "Failed to import _jllm_engine.so. "
        "Please ensure the C++ extension is built with 'xmake'."
    ) from e


class Request:
    """
    A request to the JLLM inference engine.
    
    Attributes:
        prompt: List of token IDs or text to generate from
        request_id: Unique identifier for this request
    """
    
    def __init__(self, prompt: Optional[List[int]] = None, request_id: int = 0):
        """
        Initialize a Request.
        
        Args:
            prompt: Token IDs to use as input. Defaults to empty list.
            request_id: Unique identifier for this request. Defaults to 0.
        """
        self._native = jllm_engine.Request()
        if prompt is not None:
            self.prompt = prompt
        self.request_id = request_id
    
    @property
    def prompt(self) -> List[int]:
        """Get the prompt tokens."""
        return self._native.prompt
    
    @prompt.setter
    def prompt(self, value: List[int]):
        """Set the prompt tokens."""
        self._native.prompt = value
    
    @property
    def request_id(self) -> int:
        """Get the request ID."""
        return self._native.request_id
    
    @request_id.setter
    def request_id(self, value: int):
        """Set the request ID."""
        self._native.request_id = value
    
    def __repr__(self) -> str:
        return (f"Request(prompt_len={len(self.prompt)}, "
                f"request_id={self.request_id})")


class Config:
    """
    Configuration for the JLLM engine.
    
    Attributes:
        cache_num_block: Number of blocks in the KV cache
        cache_block_size: Size of each cache block
    """
    
    def __init__(self, cache_num_block: int = 128, 
                 cache_block_size: int = 1024):
        """
        Initialize engine configuration.
        
        Args:
            cache_num_block: Number of blocks in KV cache. Defaults to 128.
            cache_block_size: Size of each cache block. Defaults to 1024.
        """
        self._native = jllm_engine.Config()
        self.cache_num_block = cache_num_block
        self.cache_block_size = cache_block_size
    
    @property
    def cache_num_block(self) -> int:
        """Get the number of cache blocks."""
        return self._native.cache_num_block
    
    @cache_num_block.setter
    def cache_num_block(self, value: int):
        """Set the number of cache blocks."""
        self._native.cache_num_block = value
    
    @property
    def cache_block_size(self) -> int:
        """Get the cache block size."""
        return self._native.cache_block_size
    
    @cache_block_size.setter
    def cache_block_size(self, value: int):
        """Set the cache block size."""
        self._native.cache_block_size = value
    
    def __repr__(self) -> str:
        return (f"Config(cache_num_block={self.cache_num_block}, "
                f"cache_block_size={self.cache_block_size})")


class Engine:
    """
    Base JLLM inference engine.
    
    Provides synchronous inference capabilities with automatic
    resource management.
    """
    
    def __init__(self):
        """Initialize the JLLM engine."""
        self._native = jllm_engine.Engine()
        self._is_initialized = True
    
    def generate(self, request: Request) -> List[int]:
        """
        Generate output tokens from a request.
        
        Args:
            request: The inference request containing prompt tokens
            
        Returns:
            List of generated token IDs
            
        Raises:
            RuntimeError: If engine is not properly initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        if not isinstance(request, Request):
            raise TypeError(f"Expected Request, got {type(request)}")
        
        return self._native.generate(request._native)
    
    def step(self) -> None:
        """
        Execute one inference step.
        
        This processes pending requests and generates one token
        for each active sequence.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        self._native.step()

    def model_path(self) -> str:
        """
        Get model path.
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        return self._native.model_path()
    
    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "uninitialized"
        return f"Engine({status})"


class AsyncEngine(Engine):
    """
    Asynchronous JLLM inference engine.
    
    Provides batch inference with a request queue. Multiple requests
    can be pushed and processed concurrently.
    """
    
    def __init__(self):
        """Initialize the async JLLM engine."""
        self._native = jllm_engine.AsyncEngine()
        self._is_initialized = True
        self._pending_requests: Dict[int, Request] = {}
    
    def set_up(self) -> None:
        """
        Initialize the async engine.
        
        Should be called after creating the engine and before
        pushing requests.
        
        Raises:
            RuntimeError: If setup fails
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        self._native.set_up()
    
    def push(self, request: Request) -> int:
        """
        Push a request to the inference queue.
        
        Args:
            request: The inference request to queue
            
        Returns:
            The request ID assigned
            
        Raises:
            TypeError: If request is not a Request object
            RuntimeError: If push fails
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        if not isinstance(request, Request):
            raise TypeError(f"Expected Request, got {type(request)}")
        
        request_id = request.request_id
        self._pending_requests[request_id] = request
        self._native.push(request._native)
        
        return request_id
    
    def has_output(self) -> bool:
        """
        Check if there are completed inference results available.
        
        Returns:
            True if at least one result is available, False otherwise
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        return self._native.has_output()
    
    def get_all(self) -> List[Tuple[int, List[int]]]:
        """
        Retrieve all completed inference results.
        
        Returns:
            List of (request_id, output_tokens) tuples
        """
        if not self._is_initialized:
            raise RuntimeError("Engine is not initialized")
        
        results = self._native.get_all()
        
        # Clean up the requests from pending dict
        for request_id, _ in results:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
        
        return results
    
    def pending_count(self) -> int:
        """
        Get the number of pending requests.
        
        Returns:
            Number of requests still in the queue
        """
        return len(self._pending_requests)
    
    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "uninitialized"
        pending = len(self._pending_requests)
        return f"AsyncEngine({status}, pending={pending})"