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
                Ты — классификатор срочности ответа на корпоративные письма банка.
                Твоя задача - определить уровень срочности для ответа на письмо
                
                КАТЕГОРИИ СРОЧНОСТИ (только эти 5 вариантов):
                - urgent: ответ требуется в течение 1-2 рабочих дней (срочно, просрочено, менее 5 дней до дедлайна)
                - high: ответ в течение 3-5 рабочих дней (5-10 дней до дедлайна, срочные формулировки)
                - medium: ответ в течение 6-10 рабочих дней (10-20 дней до дедлайна)
                - low: ответ более чем через 10 рабочих дней (более 20 дней до дедлайна / нет срочности)
                - no_deadline: дедлайн не указан (уведомления, информирование)
                
                Никаких других критериев НЕТ.

                СЕГОДНЯ: {formatted_date}. Отталкивайся от этой даты

                КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА КЛАССИФИКАЦИИ:

                1. АНАЛИЗ ДАТ:
                   - Даты после предлога "от" — ИГНОРИРУЙ (это даты документов)
                   - Даты после предлога "до" — ЭТО ДЕДЛАЙНЫ ОТВЕТА
                   - Пример: "от 10.04.2024... до 25.11.2025" → анализируем только "до 25.11.2025"

                3. АНАЛИЗ КЛЮЧЕВЫХ СЛОВ:
                   - "немедленно", "срочно", "24 часа", "сегодня" → urgent
                   - "в течение 3 дней", "до конца недели", "требуем" → high
                   - "в течение 7 дней", "на следующей неделе" → medium
                   - "когда удобно", "не срочно" → low
                   - "уведомляем", "информируем", "ответ не требуется" → no_deadline

                4. ПРИОРИТЕТЫ:
                   - Жалобы (complaint) обычно срочнее чем запросы информации
                   - Регуляторные запросы имеют высокий приоритет
                   - Партнёрские предложения обычно не срочные

                ТРЕБОВАНИЯ К ОТВЕТУ:
                — Строго в JSON-формате: {{"deadline": "...", "reason": "..."}}
                — deadline: ТОЛЬКО ОДНО значение из списка категорий выше
                — reason: краткое объяснение (1-2 предложения) на основе анализа текста
                — Обязательно укажи в reason: нашёл ли дедлайн и как оценил срочность

                ЗАПРЕЩЕНО:
                — Добавлять любые комментарии вне JSON
                — Использовать категории не из списка
                — Игнорировать найденные дедлайны в тексте
            """

        return system_prompt.strip()

    def build_user_prompt(self, user_text: str) -> str:
        return f'Определи, как срочно нужно ответить на следующее сообщение:\n "{user_text}"'

    def classify(self, text: str):
        retrieved = self.retrieve_examples(text)
        system_prompt = self.build_system_prompt(retrieved)
        user_prompt = self.build_user_prompt(text)

        response = self.client.responses.create(
            model=f"gpt://{FOLDER_ID}/yandexgpt/latest",
            instructions=system_prompt,
            input=user_prompt
        )
        raw = response.output_text.strip().replace('```', '')

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Model returned invalid JSON:\n{raw}")

        return data
