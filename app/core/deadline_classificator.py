import json

import chromadb
from chromadb import QueryResult

from app.constants import DOCUMENTS_JSON
from app.llm.client import client
from datetime import datetime

from app.utils.config import FOLDER_ID


class DeadlineClassificator:
    def __init__(self):
        self.client = client

    def build_system_prompt(self) -> str:
        date_now = datetime.now()
        formatted_date = datetime.strftime(date_now, '%d.%m.%Y')

        system_prompt = f"""
        Ты — классификатор срочности и дедлайнов в корпоративных письмах банка.
        Твоя задача: определить ДЕДЛАЙН ответа на письмо, если он указан в тексте.

        ТРЕБУЕМАЯ ЛОГИКА:

        1. АНАЛИЗ ДАТ:
           - Даты после слова "от" — ИГНОРИРУЙ (это даты документа)
           - Даты после слова "до" — ЭТО ДЕДЛАЙН, его нужно вернуть
           - Если встречаются формулировки типа:
               "в течение 3 дней", "до конца недели", "24 часа" —
               рассчитай дату дедлайна относительно СЕГОДНЯ.
           - СЕГОДНЯ: {formatted_date}

        2. ЕСЛИ В ТЕКСТЕ НЕ УКАЗАН НИКАКОЙ ДЕДЛАЙН:
           - Вернуть {{'deadline': None}}

        3. ЕСЛИ ДЕДЛАЙН УКАЗАН:
           - Вернуть дату в формате YYYY-MM-DD

        4. ИГНОРИРУЙ всё, что не относится к срокам ответа:
           - даты документов
           - даты договоров
           - даты прошлой переписки

        5. Правила интерпретации:
           - "в течение X дней" → дедлайн = сегодня + X дней
           - "до конца недели" → воскресенье текущей недели
           - "немедленно", "оперативно", "24 часа" → дедлайн = сегодня + 1 день
           - "до понедельника", "до вторника" → ближайшая указанная неделядата
           - "срочно", "ASAP" → дедлайн = сегодня + 1 день, если не указано иначе

        ТРЕБОВАНИЯ К ФОРМАТУ ОТВЕТА:
        — Строго JSON:
            {{'deadline': '<YYYY-MM-DD или None>'}}
        — Никаких комментариев вне JSON.
        — Не придумывай даты, если их нет.
        """

        return system_prompt.strip()

    def build_user_prompt(self, user_text: str) -> str:
        return f'Определи, как срочно нужно ответить на следующее сообщение:\n "{user_text}"'

    def classify(self, text: str):
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(text)

        response = self.client.responses.create(
            model=f"gpt://{FOLDER_ID}/yandexgpt/latest",
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.0
        )
        raw = response.output_text.strip().replace('```', '')

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Model returned invalid JSON:\n{raw}")

        return data


