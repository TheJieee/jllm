"""
Utility functions and helpers for the JLLM engine.

This module provides convenient utility functions for common operations
like token processing, batch handling, and result formatting.
"""

from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
from ._engine_wrapper import Request, Config, AsyncEngine


@dataclass
class InferenceResult:
    """Container for inference results."""
    request_id: int
    tokens: List[int]
    tokenizer: Optional[Any] = None
    
    def __str__(self) -> str:
        return f"InferenceResult(id={self.request_id}, tokens={len(self.tokens)})"
    
    def decode(self, tokenizer: Optional[Any] = None) -> str:
        """
        Decode token IDs to text using AutoTokenizer.
        
        Args:
            tokenizer: Optional AutoTokenizer instance. If not provided, uses stored tokenizer.
            
        Returns:
            Decoded text string
        """
        decoder = tokenizer or self.tokenizer
        if decoder is None:
            raise ValueError("No tokenizer provided or stored in InferenceResult")
        return decoder.decode(self.tokens, skip_special_tokens=True)


class BatchProcessor:
    """
    Helper class for batch inference processing.
    
    Simplifies handling multiple requests with the AsyncEngine.
    """
    
    def __init__(self, engine: AsyncEngine, batch_size: int = 32, tokenizer: Optional[Any] = None):
        """
        Initialize the batch processor.
        
        Args:
            engine: AsyncEngine instance to use
            batch_size: Maximum number of pending requests
            tokenizer: Optional AutoTokenizer instance for decoding results
        """
        self.engine = engine
        self.batch_size = batch_size
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(engine.model_path())
        self.results_cache: List[InferenceResult] = []
    
    def process_batch(self, prompts: List[List[int]]) -> List[InferenceResult]:
        """
        Process a batch of prompts.
        
        Args:
            prompts: List of token sequences to process
            
        Returns:
            List of InferenceResult objects with tokenizer for decoding
        """
        results = []
        request_id = 0
        
        # Push all requests
        for prompt in prompts:
            if len(self.engine._pending_requests) >= self.batch_size:
                # Wait for some results before pushing more
                while self.engine.has_output():
                    req_id, tokens = self.engine.get_one()
                    results.append(InferenceResult(req_id, tokens, self.tokenizer))
            
            request = Request(prompt=prompt, request_id=request_id)
            self.engine.push(request)
            request_id += 1
        
        # Collect remaining results
        all_results = self.engine.get_all()
        for req_id, tokens in all_results:
            results.append(InferenceResult(req_id, tokens, self.tokenizer))
        
        return results
    
    def process_streaming(self, prompts: List[List[int]], 
                         callback: Optional[Callable[[InferenceResult], None]] = None) -> None:
        """
        Process prompts in a streaming fashion.
        
        Args:
            prompts: List of token sequences to process
            callback: Optional callback function called for each result
        """
        request_id = 0
        
        # Push all requests
        for prompt in prompts:
            request = Request(prompt=prompt, request_id=request_id)
            self.engine.push(request)
            request_id += 1
        
        # Stream results
        while self.engine.pending_count() > 0 or self.engine.has_output():
            if self.engine.has_output():
                req_id, tokens = self.engine.get_one()
                result = InferenceResult(req_id, tokens)
                if callback:
                    callback(result)
                else:
                    self.results_cache.append(result)


class TokenMapper:
    """
    Utility for mapping between tokens and strings.
    
    Placeholder for vocabulary management.
    """
    
    def __init__(self):
        """Initialize the token mapper."""
        self.token_to_string: Dict[int, str] = {}
        self.string_to_token: Dict[str, int] = {}
    
    def add_mapping(self, token_id: int, token_str: str) -> None:
        """Add a token-string mapping."""
        self.token_to_string[token_id] = token_str
        self.string_to_token[token_str] = token_id
    
    def tokens_to_string(self, tokens: List[int]) -> str:
        """Convert token IDs to string."""
        return "".join(self.token_to_string.get(t, f"<{t}>") for t in tokens)
    
    def string_to_tokens(self, text: str) -> List[int]:
        """Convert string to token IDs (requires proper tokenizer)."""
        raise NotImplementedError(
            "Use a proper tokenizer like transformers.AutoTokenizer"
        )


def create_debug_engine(verbose: bool = True) -> AsyncEngine:
    """
    Create an AsyncEngine with debug logging.
    
    Args:
        verbose: Whether to print debug information
        
    Returns:
        Initialized AsyncEngine
    """
    engine = AsyncEngine()
    
    if verbose:
        print("[DEBUG] Creating AsyncEngine...")
    
    engine.set_up()
    
    if verbose:
        print("[DEBUG] AsyncEngine initialized")
    
    return engine


def format_results(results: List[InferenceResult]) -> str:
    """
    Format inference results for display.
    
    Args:
        results: List of InferenceResult objects
        
    Returns:
        Formatted string representation
    """
    lines = [
        "=" * 60,
        f"Inference Results ({len(results)} total)",
        "=" * 60,
    ]
    
    for result in results:
        lines.append(f"Request {result.request_id:3d}: {result.tokens[:10]}...")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def load_tokenizer(model_name_or_path: str) -> Any:
    """
    Load a tokenizer using AutoTokenizer.
    
    Args:
        model_name_or_path: Model name (HuggingFace Hub) or path to local model
        
    Returns:
        Loaded tokenizer instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {model_name_or_path}: {e}")


def decode_tokens(tokens: List[int], tokenizer: Any, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs to text using the provided tokenizer.
    
    Args:
        tokens: List of token IDs
        tokenizer: AutoTokenizer instance
        skip_special_tokens: Whether to skip special tokens in decoding
        
    Returns:
        Decoded text string
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def encode_text(text: str, tokenizer: Any, return_tensors: Optional[str] = None) -> List[int]:
    """
    Encode text to token IDs using the provided tokenizer.
    
    Args:
        text: Input text to encode
        tokenizer: AutoTokenizer instance
        return_tensors: Optional format for returned tensors (e.g., 'pt' for PyTorch)
        
    Returns:
        List of token IDs (or tensor if return_tensors is specified)
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    
    encoded = tokenizer.encode(text, return_tensors=return_tensors)
    return encoded


def encode_chat(messages: List[Dict[str, str]], tokenizer: Any, 
                return_tensors: Optional[str] = None, add_generation_prompt: bool = True) -> List[int]:
    """
    Encode chat messages to token IDs using the model's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  Example: [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
        tokenizer: AutoTokenizer instance
        return_tensors: Optional format for returned tensors (e.g., 'pt' for PyTorch)
        add_generation_prompt: Whether to append Generation prompt tokens
        
    Returns:
        List of token IDs
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates. Try using a model that supports chat mode.")
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
    
    # Encode the formatted text
    encoded = tokenizer.encode(text, return_tensors=return_tensors)
    return encoded


def decode_chat_response(tokens: List[int], tokenizer: Any, skip_special_tokens: bool = True) -> str:
    """
    Decode chat response tokens to text.
    
    Args:
        tokens: List of token IDs from model output
        tokenizer: AutoTokenizer instance
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded response text
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def create_inference_result_with_decode(request_id: int, tokens: List[int], 
                                        tokenizer: Any) -> tuple[InferenceResult, str]:
    """
    Create an InferenceResult and immediately decode it to text.
    
    Args:
        request_id: Request ID
        tokens: List of token IDs
        tokenizer: AutoTokenizer instance
        
    Returns:
        Tuple of (InferenceResult, decoded_text)
    """
    result = InferenceResult(request_id, tokens, tokenizer)
    decoded_text = result.decode(tokenizer)
    return result, decoded_text


def batch_decode_results(results: List[InferenceResult], 
                        tokenizer: Any) -> List[str]:
    """
    Batch decode multiple InferenceResults to text.
    
    Args:
        results: List of InferenceResult objects
        tokenizer: AutoTokenizer instance for decoding
        
    Returns:
        List of decoded text strings
    """
    decoded_texts = []
    for result in results:
        try:
            text = result.decode(tokenizer)
            decoded_texts.append(text)
        except Exception as e:
            print(f"Warning: Failed to decode result {result.request_id}: {e}")
            decoded_texts.append("")
    return decoded_texts


class TokenizerWrapper:
    """
    Wrapper around AutoTokenizer providing additional utilities.
    
    Simplifies token encoding/decoding operations.
    """
    
    def __init__(self, model_name_or_path: str):
        """
        Initialize the tokenizer wrapper.
        
        Args:
            model_name_or_path: Model name or path to load tokenizer from
        """
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.model_name = model_name_or_path
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> List[int]:
        """Encode text to token IDs."""
        return encode_text(text, self.tokenizer, return_tensors)
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return decode_tokens(tokens, self.tokenizer, skip_special_tokens)
    
    def encode_chat(self, messages: List[Dict[str, str]], return_tensors: Optional[str] = None,
                    add_generation_prompt: bool = True) -> List[int]:
        """
        Encode chat messages to token IDs.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            return_tensors: Optional format for returned tensors
            add_generation_prompt: Whether to append generation prompt tokens
            
        Returns:
            List of token IDs
        """
        return encode_chat(messages, self.tokenizer, return_tensors, add_generation_prompt)
    
    def decode_chat_response(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode chat response tokens to text.
        
        Args:
            tokens: List of token IDs from model output
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded response text
        """
        return decode_chat_response(tokens, self.tokenizer, skip_special_tokens)
    
    def process_inference_result(self, result: InferenceResult) -> str:
        """Process an InferenceResult and return decoded text."""
        result.tokenizer = self.tokenizer
        return result.decode()
    
    def process_batch(self, results: List[InferenceResult]) -> List[str]:
        """Process a batch of InferenceResults."""
        for result in results:
            result.tokenizer = self.tokenizer
        return batch_decode_results(results, self.tokenizer)
    
    def has_chat_template(self) -> bool:
        """Check if the tokenizer supports chat templates."""
        return hasattr(self.tokenizer, "apply_chat_template")
