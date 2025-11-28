from openai import OpenAI

from app.utils.config import AI_API_KEY, AI_BASE_URL

client = OpenAI(
    api_key=AI_API_KEY, base_url=AI_BASE_URL
)
