from langchain_openai import OpenAIEmbeddings
from src.config import config


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBED_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
