from openai import OpenAI

from app.utils.config import AI_API_KEY, AI_BASE_URL, FOLDER_ID

default_headers = {}
if FOLDER_ID:
    default_headers["x-folder-id"] = FOLDER_ID

client = OpenAI(
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL,
    default_headers=default_headers
)