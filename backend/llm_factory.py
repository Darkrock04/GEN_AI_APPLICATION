import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

# MULTI-CLOUD ARCHITECTURE ASSIGNMENTS
# ------------------------------------
# Ollama: Free cloud models (Router: gemma4:31b, Workers: gpt-oss:120b)
# Google: Excellent context and RPM (Perfect for heavy coding)
# Nvidia: Final evaluator / embeddings

AGENT_CONFIG = {
    "security":        {"provider": "ollama",   "model": "nemotron-3-nano:30b"},
    "planner":         {"provider": "google",   "model": "gemini-3.1-flash-lite"},
    "router":          {"provider": "ollama",   "model": "gemma4:31b"},
    "worker_general":  {"provider": "ollama",   "model": "gpt-oss:120b"},
    "worker_creative": {"provider": "ollama",   "model": "gpt-oss:120b"},
    "worker_coding":   {"provider": "google",   "model": "gemini-3.5-flash"}, 
    "validator":       {"provider": "google",   "model": "gemini-3.1-flash-lite"},
    "evaluator":       {"provider": "ollama",   "model": "nemotron-3-super"},
}

# API URLs
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# API Keys
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

def get_llm(agent_type: str, max_tokens: int = 4096) -> BaseChatModel:
    """Return a chat model for the given agent role routed to the correct cloud provider."""
    
    config = AGENT_CONFIG.get(agent_type)
    if not config:
        raise ValueError(f"Unknown agent_type '{agent_type}'.")
        
    provider = config["provider"]
    model_name = config["model"]
    
    if provider == "ollama":
        if not OLLAMA_API_KEY:
            raise RuntimeError("OLLAMA_API_KEY is not set in the .env file.")
        return ChatOpenAI(
            api_key=OLLAMA_API_KEY,
            base_url=OLLAMA_BASE_URL,
            model=model_name,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=1,
        )
        
    elif provider == "google":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in the .env file.")
        return ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=model_name,
            max_output_tokens=max_tokens,
            temperature=0.7,
            timeout=30,
            max_retries=1,
        )
        
    elif provider == "nvidia":
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set in the .env file.")
        return ChatOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            model=model_name,
            max_tokens=max_tokens,
            timeout=60,
            max_retries=1,
        )
        
    else:
        raise ValueError(f"Unknown provider '{provider}'")
