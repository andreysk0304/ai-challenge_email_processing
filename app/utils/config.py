from os import getenv
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = getenv('DATABASE_URL')
AI_API_KEY = getenv('AI_API_KEY')
AI_BASE_URL = getenv('AI_BASE_URL')