# modules/llm.py
import os
from langchain_groq import ChatGroq
from typing import Optional

from .config import GROQ_API_KEY


class LLMManager:
    """
    Wrapper for Groq LLMs.
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ):
        if not GROQ_API_KEY:
            # Transparent error for debugging
            raise ValueError("❌ GROQ_API_KEY missing in environment or config.py")

        self.system_prompt = system_prompt or (
            "You are an expert assistant. Use ONLY the given context. "
            "If answer is not present in the context, say: "
            "'I could not find the answer in the provided documents.'"
        )

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
SYSTEM INSTRUCTION:
{self.system_prompt}

CONTEXT (retrieved documents):
{context}

QUESTION:
{query}

RESPONSE FORMAT:
- Give a concise answer.
- If unsure, explicitly say you are not sure.
- Provide bullet points when helpful.
- Mention 'Sources: <chunk numbers>' if context includes metadata.

ANSWER:
"""
        try:
            response = self.llm.invoke([prompt])
            # ChatGroq `invoke` returns a model-specific object — access safely
            if hasattr(response, "content"):
                return response.content
            # fallback: str(response)
            return str(response)
        except Exception as e:
            print("[ERROR] LLM generation failed:", e)
            return "LLM failed to generate a response."

    def stream_answer(self, query: str, context: str):
        prompt = f"""
{self.system_prompt}

Context:
{context}

Question: {query}
        """
        try:
            for token in self.llm.stream([prompt]):
                yield token
        except Exception as e:
            yield f"[ERROR] Streaming failed: {e}"


def get_llm(model_name: str, temperature: float = 0.1):
    return LLMManager(
        model_name=model_name,
        temperature=temperature
    )
