"""
Advanced examples demonstrating JLLM features.

Shows batch processing, streaming inference, and other advanced usage patterns.
"""

import sys
from transformers import AutoTokenizer
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from jllm import AsyncEngine, InferenceResult, BatchProcessor


def tokenize(engine, string):
    tokenizer = AutoTokenizer.from_pretrained(engine.model_path())
        
    input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": string}],
    add_generation_prompt=True,
    tokenize=False,
    )
    return tokenizer.encode(input_content)

def example_batch_processing():
    """Example: Process multiple requests with BatchProcessor."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Batch Processing with BatchProcessor")
    print("=" * 70)
    
    try:
        # Create engine and processor
        engine = AsyncEngine()
        engine.set_up()
        processor = BatchProcessor(engine, batch_size=16)
        
        # Example prompts (token sequences)
        tokenizer = AutoTokenizer.from_pretrained(engine.model_path())
        
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "Explain how large language models work and their main applications in industry."}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs1 = tokenizer.encode(input_content)
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "What are the key differences between supervised learning and reinforcement learning?"}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs2 = tokenizer.encode(input_content)
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "How can we optimize model inference speed while maintaining accuracy?"}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs3 = tokenizer.encode(input_content)

        prompts = [
            inputs1,
            inputs2,
            inputs3,
        ]
        
        print(f"Processing {len(prompts)} prompts in batch...")
        results = processor.process_batch(prompts)
        
        for result in results:
            print(f"  {result}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_streaming_inference():
    """Example: Stream results as they become available."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Streaming Inference")
    print("=" * 70)
    
    try:
        engine = AsyncEngine()
        engine.set_up()
        processor = BatchProcessor(engine)
        
        result_count = [0]  # Use list to allow modification in nested function
        
        def on_result(result: InferenceResult):
            result_count[0] += 1
            print(f"  [{result_count[0]}] Received result: {result}")
        
        # Process with streaming callback
        tokenizer = AutoTokenizer.from_pretrained(engine.model_path())
        
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "Describe the architecture and training process of transformer models in detail."}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs1 = tokenizer.encode(input_content)
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "What challenges do we face when deploying large models to production systems?"}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs2 = tokenizer.encode(input_content)
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "How does attention mechanism work in neural networks?"}],
        add_generation_prompt=True,
        tokenize=False,
        )
        inputs3 = tokenizer.encode(input_content)

        prompts = [
            inputs1,
            inputs2,
            inputs3,
        ]
        
        print(f"Processing {len(prompts)} prompts with streaming callback...")
        processor.process_streaming(prompts, callback=on_result)
        
    except Exception as e:
        print(f"Error: {e}")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Error Handling")
    print("=" * 70)
    
    from jllm import Request, Engine
    
    try:
        # Example 1: Type checking
        engine = Engine()
        
        try:
            engine.generate("not a request")  # Wrong type
        except TypeError as e:
            print(f"✓ Caught TypeError: {e}")
        
        # Example 2: Uninitialized engine
        from jllm import AsyncEngine as AsyncEng
        async_engine = AsyncEng()
        
        try:
            # Trying to push without set_up()
            req = Request(prompt=[1, 2], request_id=1)
            async_engine.push(req)
        except RuntimeError as e:
            print(f"✓ Caught RuntimeError: {e}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_result_formatting():
    """Example: Format and display results."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Result Formatting")
    print("=" * 70)
    
    from jllm.utils import format_results
    
    # Create sample results
    sample_results = [
        InferenceResult(request_id=1, tokens=[10, 20, 30, 40, 50]),
        InferenceResult(request_id=2, tokens=[5, 15, 25, 35, 45, 55]),
        InferenceResult(request_id=3, tokens=[100, 200, 300, 400, 500, 600, 700]),
    ]
    
    formatted = format_results(sample_results)
    print(formatted)


def example_request_lifecycle():
    """Example: Complete request lifecycle."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Request Lifecycle")
    print("=" * 70)
    
    from jllm import Request, Config, AsyncEngine
    
    try:
        # Step 1: Create configuration
        config = Config(cache_num_block=128, cache_block_size=1024)
        print(f"1. Created config: {config}")
        
        # Step 2: Create requests
        requests = []
        for i in range(3):
            req = Request(
                prompt=[i*10 + j for j in range(5)],
                request_id=i
            )
            requests.append(req)
            print(f"2. Created {req}")
        
        # Step 3: Initialize engine
        engine = AsyncEngine()
        engine.set_up()
        print("3. Engine initialized")
        
        # Step 4: Push requests
        for req in requests:
            engine.push(req)
            print(f"4. Pushed request {req.request_id}")
        
        print(f"   Pending: {engine.pending_count()}")
        
        # Step 5: Get results (if available)
        if engine.has_output():
            results = engine.get_all()
            for req_id, tokens in results:
                print(f"5. Retrieved result for request {req_id}: {tokens}")
        else:
            print("5. No results available yet (expected for mock engine)")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "JLLM Advanced Examples" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Run examples
    example_request_lifecycle()
    example_error_handling()
    example_result_formatting()
    example_batch_processing()
    example_streaming_inference()
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "All examples completed" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")