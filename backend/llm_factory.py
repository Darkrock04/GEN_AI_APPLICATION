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

NVIDIA_MODELS = {
    # Fast lightweight models for routing & logic (instant response)
    "security": "meta/llama-3.1-8b-instruct",
    "planner": "meta/llama-3.1-8b-instruct",
    "router": "meta/llama-3.1-8b-instruct",
    "validator": "meta/llama-3.1-8b-instruct",
    "evaluator": "meta/llama-3.1-8b-instruct",
    
    # High Speed Workers:
    "worker_general": "meta/llama-3.1-8b-instruct",
    "worker_coding": "meta/llama-3.1-8b-instruct",
    "worker_creative": "meta/llama-3.1-8b-instruct",
}

NVIDIA_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "siliconflow").lower().strip()

# OpenAI (official API)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Self-hosted or any OpenAI-compatible gateway
CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "").rstrip("/")
CUSTOM_LLM_API_KEY = os.getenv("CUSTOM_LLM_API_KEY", "dummy")
CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_GENAI_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")


def get_llm(agent_type: str, max_tokens: int = 4096) -> BaseChatModel:
    """Return a chat model for the given agent role based on LLM_PROVIDER."""

    if LLM_PROVIDER == "siliconflow":
        model_name = SILICONFLOW_MODELS.get(agent_type)
        if not model_name:
            raise ValueError(f"Unknown agent_type '{agent_type}'.")
        if not SILICONFLOW_API_KEY:
            raise RuntimeError("SILICONFLOW_API_KEY is not set (required for LLM_PROVIDER=siliconflow).")
        return ChatOpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
            model=model_name,
            max_tokens=max_tokens,
            timeout=300,
            max_retries=2,
        )

    if LLM_PROVIDER == "nvidia":
        model_name = NVIDIA_MODELS.get(agent_type)
        if not model_name:
            raise ValueError(f"Unknown agent_type '{agent_type}'.")
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set (required for LLM_PROVIDER=nvidia).")
        return ChatOpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            model=model_name,
            max_tokens=max_tokens,
            timeout=90,
            max_retries=1,
        )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (required for LLM_PROVIDER=openai).")
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_CHAT_MODEL,
            max_tokens=max_tokens,
            timeout=90,
            max_retries=1,
        )

    if LLM_PROVIDER == "openai_compatible":
        if not CUSTOM_LLM_BASE_URL:
            raise RuntimeError("CUSTOM_LLM_BASE_URL is not set (required for LLM_PROVIDER=openai_compatible).")
        return ChatOpenAI(
            api_key=CUSTOM_LLM_API_KEY,
            base_url=CUSTOM_LLM_BASE_URL,
            model=CUSTOM_LLM_MODEL,
            max_tokens=max_tokens,
            timeout=90,
            max_retries=1,
        )

    if LLM_PROVIDER == "google_genai":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set (required for LLM_PROVIDER=google_genai).")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=google_genai requires langchain-google-genai. "
                "Install with: pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model=GOOGLE_GENAI_MODEL,
            max_output_tokens=max_tokens,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
        "Use: siliconflow | nvidia | openai | openai_compatible | google_genai"
    )
