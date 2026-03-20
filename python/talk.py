"""
Example usage of the JLLM Python wrapper.

Demonstrates both synchronous and asynchronous inference.
"""

from typing import Any

from transformers import AutoTokenizer

from jllm import Engine, AsyncEngine, Request, Config, TokenizerWrapper

if __name__ == "__main__":
    print("\n")
    print("JLLM Python Wrapper Examples")
    print("=" * 60)
    
    # Note: These examples will only work if the C++ engine is properly
    # initialized. For now, they demonstrate the API usage.
    
    try:
        engine = Engine()
        print(engine.model_path())
        tokenizer = AutoTokenizer.from_pretrained(engine.model_path())
        
        input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "Hello!"}],
        add_generation_prompt=True,
        tokenize=False,
        )
        print(input_content)
        inputs = tokenizer.encode(input_content)
        print(type(inputs))
        print(inputs)
        output = engine.generate(Request(inputs))
        print(tokenizer.decode(output))

    except Exception as e:
        print(f"Error: {e}")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)