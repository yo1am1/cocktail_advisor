from typing import List, Tuple, Dict, Any, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from app.config import Config


def get_openai_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        openai_api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        base_url=Config.AZURE_OPENAI_BASE_URL,
        default_headers={
            "Ocp-Apim-Subscription-Key": Config.AZURE_OPENAI_SUBSCRIPTION_KEY
        },
        model=Config.EMBEDDING_MODEL,
    )


def get_openai_model(
    model: str = Config.CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 30,
    max_retries: int = 2,
) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        model_name=model,
        openai_api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.AZURE_OPENAI_API_VERSION,
        base_url=Config.AZURE_OPENAI_BASE_URL,
        default_headers={
            "Ocp-Apim-Subscription-Key": Config.AZURE_OPENAI_SUBSCRIPTION_KEY
        },
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
