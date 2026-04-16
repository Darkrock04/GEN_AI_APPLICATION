import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

# Model registry — optimized for speed and reliability.
SILICONFLOW_MODELS = {
    # Fast lightweight models for routing & logic (instant response)    
    "security": "Qwen/Qwen2.5-7B-Instruct",
    "planner": "Qwen/Qwen2.5-7B-Instruct",
    "router": "Qwen/Qwen2.5-7B-Instruct",
    "validator": "Qwen/Qwen2.5-7B-Instruct",
    
    # High capacity intelligent models for heavy lifting
    "evaluator": "deepseek-ai/DeepSeek-V3",
    "worker_general": "deepseek-ai/DeepSeek-V3",
    "worker_coding": "deepseek-ai/DeepSeek-V3",
    "worker_creative": "deepseek-ai/DeepSeek-V3",
}

SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")


def get_llm(agent_type: str, max_tokens: int = 4096) -> BaseChatModel:
    """Return a chat model for the given agent role."""
    
    model_name = SILICONFLOW_MODELS.get(agent_type)
    if not model_name:
        raise ValueError(f"Unknown agent_type '{agent_type}'.")
        
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("SILICONFLOW_API_KEY is not set in the .env file.")
        
    return ChatOpenAI(
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        model=model_name,
        max_tokens=max_tokens,
        timeout=300,
        max_retries=2,
    )
