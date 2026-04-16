import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

# Model registry — optimized for speed and reliability.
NVIDIA_MODELS = {
    # Fast lightweight models for routing & logic (instant response)    
    "security": "mistralai/mistral-large-2-instruct",
    "planner": "meta/llama-3.1-70b-instruct",
    "router": "meta/llama-3.1-8b-instruct",
    "validator": "meta/llama-3.1-70b-instruct",
    
    # High capacity intelligent models for heavy lifting
    "evaluator": "nvidia/nemotron-4-340b-instruct",
    "worker_general": "meta/llama-3.1-70b-instruct",
    "worker_coding": "meta/llama-3.1-405b-instruct",
    "worker_creative": "mistralai/mixtral-8x22b-instruct-v0.1",
}

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


def get_llm(agent_type: str, max_tokens: int = 4096) -> BaseChatModel:
    """Return a chat model for the given agent role."""
    
    model_name = NVIDIA_MODELS.get(agent_type)
    if not model_name:
        raise ValueError(f"Unknown agent_type '{agent_type}'.")
        
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set in the .env file.")
        
    return ChatOpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
        model=model_name,
        max_tokens=max_tokens,
        timeout=300,
        max_retries=2,
    )
