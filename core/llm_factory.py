"""
LLM and Embeddings Factory
--------------------------
Provides a modular abstraction layer for Chat Models, Multimodal Vision Models,
and Embeddings. Allows seamless swapping of the underlying model provider
(Google Gemini, OpenAI, Anthropic Claude, or local Ollama) by changing
configuration without touching business logic or agent code.
"""

import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

import config


class LLMFactory:
    """
    Factory class to instantiate LLMs and multimodal models dynamically
    based on the configured LLM_PROVIDER.
    """

    @staticmethod
    def get_chat_model(
        temperature: float = 0.2,
        streaming: bool = True,
        provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseChatModel:
        """
        Instantiate a chat model based on the selected provider.
        
        Args:
            temperature: Creativity/variance parameter (0.0 to 1.0)
            streaming: Whether the model supports streaming chunks
            provider: Override global LLM_PROVIDER ('gemini', 'openai', 'anthropic', 'ollama')
            model_name: Optional specific model name override
            
        Returns:
            BaseChatModel compatible with LangChain & LangGraph
        """
        active_provider = (provider or config.LLM_PROVIDER).lower()

        # ----------------------------------------------------
        # Google Gemini Provider (Default & Primary)
        # ----------------------------------------------------
        if active_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set in environment or .env file. "
                    "Please provide a valid Google Gemini API key."
                )
            
            selected_model = model_name or config.GEMINI_MODEL_NAME
            return ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=api_key,
                temperature=temperature,
                streaming=streaming,
                convert_system_message_to_human=True
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider '{active_provider}'. "
                f"Configured provider is 'gemini' with model '{config.GEMINI_MODEL_NAME}'."
            )


class EmbeddingFactory:
    """
    Factory class to instantiate text embeddings for ChromaDB vector operations.
    """

    @staticmethod
    def get_embeddings() -> Embeddings:
        """
        Returns a LangChain-compatible Embeddings model.
        Uses Google Gemini cloud embeddings for 0 MB local RAM,
        falling back to local HuggingFace Sentence-Transformers.
        """
        if getattr(config, "EMBEDDING_PROVIDER", "gemini").lower() == "gemini" and config.GEMINI_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=config.GEMINI_API_KEY
                )
            except Exception as e:
                print(f"Notice: GoogleGenerativeAIEmbeddings fallback note: {e}")

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        except Exception:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=config.GEMINI_API_KEY
            )


def extract_text_from_message_content(content: Any) -> str:
    """
    Safely extracts string text from an AIMessage content field.
    Handles plain strings, list of text/dict chunks (Gemini 3.8 / new GenAI format),
    and structured message parts without throwing AttributeError on .strip().
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
        return " ".join(parts).strip()
    return str(content).strip()

