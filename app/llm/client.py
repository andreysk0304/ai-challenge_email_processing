from openai import OpenAI

from app.utils.config import AI_API_KEY, AI_BASE_URL, FOLDER_ID

client = OpenAI(
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL or "https://rest-assistant.api.cloud.yandex.net/v1",
    project=FOLDER_ID
)