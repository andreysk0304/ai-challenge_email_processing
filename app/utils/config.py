import pytz

from os import getenv
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = getenv('DATABASE_URL')
REDIS_URL = getenv("REDIS_URL", "redis://redis:6379/0")

AI_API_KEY = getenv('AI_API_KEY')
AI_BASE_URL = getenv('AI_BASE_URL')

MOSCOW_TZ = pytz.timezone("Europe/Moscow")