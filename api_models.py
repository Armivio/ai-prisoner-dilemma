import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API libraries (these need to be installed)
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def get_prediction(model_input: Dict[str, Any]) -> str:
    """
    Get a prediction from the specified AI model.
    
    Args:
        model_input: Dictionary containing:
            - model: The EXACT model identifier (e.g., "claude-3-5-haiku-20241022")
            - prompt: The complete formatted prompt
            - Other metadata (task, round, opponent, history, cross_memory)
    
    Returns:
        str: Either "c" (cooperate) or "d" (defect)
    """
    model_name = model_input["model"]
    prompt = model_input["prompt"]
    
    try:
        # Anthropic models
        if any(name in model_name for name in ["claude", "anthropic", "opus", "sonnet", "haiku"]):
            return get_anthropic_prediction(model_name, prompt)
        
        # OpenAI models
        elif any(name in model_name for name in ["gpt", "openai", "o1"]):
            return get_openai_prediction(model_name, prompt)
        
        # Google Gemini models
        elif any(name in model_name for name in ["gemini", "google"]):
            return get_gemini_prediction(model_name, prompt)
        
        else:
            print(f"Unknown model type: {model_name}. Defaulting to defect.")
            return "d"
    
    except Exception as e:
        print(f"Error getting prediction from {model_name}: {e}. Defaulting to defect.")
        return "d"


def get_anthropic_prediction(model_name: str, prompt: str) -> str:
    """Get prediction from Anthropic Claude models."""
    if not anthropic:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    
    
    actual_model = model_name
    
    # Create message
    message = client.messages.create(
        model=actual_model,
        max_tokens=10,  # We only need a single character response
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Extract the response
    response_text = message.content[0].text.strip().lower()
    
    # Extract 'c' or 'd' from the response
    if 'c' in response_text and 'd' not in response_text:
        return 'c'
    elif 'd' in response_text and 'c' not in response_text:
        return 'd'
    else:
        # If unclear, look for the first occurrence
        for char in response_text:
            if char in ['c', 'd']:
                return char
    
    return 'd'  # Default to defect if unclear


def get_openai_prediction(model_name: str, prompt: str) -> str:
    """Get prediction from OpenAI models."""
    if not openai:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = openai.OpenAI(api_key=api_key)
    
    
    actual_model = model_name
    
    # For O1 models, use a different approach
    if "o1" in actual_model:
        # O1 models don't support system messages or temperature
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "user", "content": prompt}
            ], # add service_tier="flex" for cheaper responses
            max_tokens=10
        )
    else:
        # Regular models
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": "You are playing the Prisoner's Dilemma. Respond with only 'c' or 'd'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=10,
            response_format={"type": "text"}  # Ensure we get text response
        )
    
    # Extract the response
    response_text = response.choices[0].message.content.strip().lower()
    
    # Extract 'c' or 'd' from the response
    if 'c' in response_text and 'd' not in response_text:
        return 'c'
    elif 'd' in response_text and 'c' not in response_text:
        return 'd'
    else:
        # If unclear, look for the first occurrence
        for char in response_text:
            if char in ['c', 'd']:
                return char
    
    return 'd'  # Default to defect if unclear


def get_gemini_prediction(model_name: str, prompt: str) -> str:
    """Get prediction from Google Gemini models."""
    if not genai:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    
    
    actual_model = model_name
    
    # Create the model
    model = genai.GenerativeModel(actual_model)
    
    # Configure generation settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 10,
    }
    
    # Generate response
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    # Extract the response
    response_text = response.text.strip().lower()
    
    # Extract 'c' or 'd' from the response
    if 'c' in response_text and 'd' not in response_text:
        return 'c'
    elif 'd' in response_text and 'c' not in response_text:
        return 'd'
    else:
        # If unclear, look for the first occurrence
        for char in response_text:
            if char in ['c', 'd']:
                return char
    
    return 'd'  # Default to defect if unclear


# Helper function to test API connectivity
def test_api_connectivity():
    """Test if API keys are set and libraries are installed."""
    print("Testing API connectivity...\n")
    
    # Test Anthropic
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        print("✓ Anthropic API key found and library installed")
    else:
        if not anthropic:
            print("✗ Anthropic library not installed. Run: pip install anthropic")
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("✗ ANTHROPIC_API_KEY not found in environment")
    
    # Test OpenAI
    if openai and os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found and library installed")
    else:
        if not openai:
            print("✗ OpenAI library not installed. Run: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            print("✗ OPENAI_API_KEY not found in environment")
    
    # Test Google
    if genai and os.getenv("GOOGLE_API_KEY"):
        print("✓ Google API key found and library installed")
    else:
        if not genai:
            print("✗ Google library not installed. Run: pip install google-generativeai")
        if not os.getenv("GOOGLE_API_KEY"):
            print("✗ GOOGLE_API_KEY not found in environment")


if __name__ == "__main__":
    # Test API connectivity when run directly
    test_api_connectivity()