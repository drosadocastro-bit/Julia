"""
Julia LLM Configuration — Environment-adaptive settings.

Detects whether running on Windows (prototype) or Jetson (production)
and adjusts LLM settings accordingly.
"""

import platform
import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration for Julia's brain."""
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    timeout: int
    context_window: int


def get_llm_config() -> LLMConfig:
    """
    Auto-detect environment and return appropriate config.
    
    Windows (LM Studio):  localhost:1234, full context
    Jetson (Ollama):      localhost:11434, optimized context
    
    Can be overridden with environment variables:
        JULIA_LLM_URL=http://localhost:1234/v1
        JULIA_LLM_MODEL=qwen3-8b
    """
    
    # Check for environment variable overrides first
    custom_url = os.environ.get("JULIA_LLM_URL")
    custom_model = os.environ.get("JULIA_LLM_MODEL")
    
    system = platform.system()
    machine = platform.machine()
    
    # Detect Jetson (aarch64 Linux)
    is_jetson = system == "Linux" and machine == "aarch64"
    
    if is_jetson:
        # --- JETSON ORIN NANO (Production) ---
        return LLMConfig(
            base_url=custom_url or "http://localhost:11434/v1",  # Ollama
            model=custom_model or "qwen3:8b-q4_K_M",
            max_tokens=512,          # Shorter responses, save memory
            temperature=0.7,
            timeout=180,             # CPU/GPU inference may be slower
            context_window=4096      # Conservative for 8GB
        )
    else:
        # --- WINDOWS / DEV (Prototype via LM Studio) ---
        return LLMConfig(
            base_url=custom_url or "http://127.0.0.1:1234/v1",  # LM Studio
            model=custom_model or "qwen/qwen3-8b",
            max_tokens=1024,
            temperature=0.7,
            timeout=120,
            context_window=8192      # More room on dev machine
        )
