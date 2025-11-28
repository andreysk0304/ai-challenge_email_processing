import json

import chromadb
from chromadb import QueryResult

from app.constants import DOCUMENTS_JSON
from app.llm.client import client
from datetime import datetime


class DeadlineClassificator:
    def __init__(self):
        self.client = client
        self.collection = self._init_vector_collection()

    def _init_vector_collection(self):
        chroma = chromadb.Client()

        collection = chroma.create_collection(
            name='deadline_classifier_examples',
            embedding_function=None
        )
        deadlines_examples = None
        with open(DOCUMENTS_JSON) as f:
            deadlines_examples = json.load(f)['formality'].items()
        for i, (label, texts) in enumerate(deadlines_examples):
            for text in texts:
                collection.add(
                    ids=[str(i)],
                    documents=[text],
                    metadatas=[{'label': label}]
                )

        return collection

    def retrieve_examples(self, text: str) -> QueryResult:
        return self.collection.query(
            query_texts=[text],
            n_results=3
        )

    def build_system_prompt(self, retrieved: dict) -> str:
        date_now = datetime.now()
        formatted_date = datetime.strftime(date_now, '%d.%m.%Y')
        examples_text = ''
        for doc, meta in zip(retrieved['documents'][0], retrieved['metadatas'][0]):
            examples_text += f"Тип дедлайна: {meta['label']}\nТекст: {doc}\n\n"

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

                2. ОЦЕНКА СРОЧНОСТИ ПО ДАТАМ (например, относительно 10.12.2025):
                   - Дедлайн ДО 15.12.2025 → urgent (менее 5 дней)
                   - Дедлайн ДО 20.12.2025 → high (5-10 дней)  
                   - Дедлайн ДО 30.12.2025 → medium (10-20 дней)
                   - Дедлайн ПОСЛЕ 30.12.2025 → low (более 20 дней)
                   - Дедлайн ПРОШЁЛ → urgent (просрочено)

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

                    
                ПРИМЕРЫ ИЗ БАЗЫ ЗНАНИЙ ДЛЯ ОБУЧЕНИЯ:
                {examples_text}

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

        response = self.client.chat.completions.create(
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Model returned invalid JSON:\n{raw}")

        return data
