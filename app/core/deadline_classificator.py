import chromadb
import json

from app.core.documents import DEADLINE_DOCUMENTS
from app.llm.client import client
from datetime import datetime


class DeadlineCategory:
    def __init__(self):
        self.client = client
        self.collection = self._init_vector_collection()

    def _init_vector_collection(self):
        chroma = chromadb.Client()

        collection = chroma.create_collection(
            name='deadline_classifier_examples',
            embedding_function=None
        )

        for i, (label, text) in enumerate(DEADLINE_DOCUMENTS):
            collection.add(
                ids=[str(i)],
                documents=[text],
                metadatas=[{'label': label}]
            )

        return collection

    def retrieve_examples(self, text: str) -> dict:
        return self.collection.query(
            query_texts=[text],
            n_results=3
        )

    def build_system_prompt(self, retrieved: dict, category: str, letter_date: str) -> str:
        date_now = datetime.now()
        formatted_date = datetime.strftime(date_now, '%d.%m.%Y')
        examples_text = ''
        for doc, meta in zip(retrieved['documents'][0], retrieved['metadatas'][0]):
            examples_text += f"Тип дедлайна: {meta['label']}\nТекст: {doc}\n\n"

        system_prompt = f"""
            Ты — классификатор срочности ответа на корпоративные письма банка.

            ВАЖНО: Учитывай РАЗНИЦУ между датой письма и сегодняшним днём!

            ВХОДНЫЕ ДАННЫЕ:
            - ДАТА ПОЛУЧЕНИЯ ПИСЬМА: {letter_date}
            - СЕГОДНЯШНЯЯ ДАТА: {formatted_date}
            - КАТЕГОРИЯ ПИСЬМА: {category}

            КАТЕГОРИИ ПИСЕМ (для понимания контекста):
            • information_request - запрос информации/документов
            • complaint - жалоба или претензия (ВСЕГДА СРОЧНО!)
            • regular_request - регуляторный запрос (ВСЕГДА СРОЧНО!)  
            • partner_offer - партнёрское предложение (обычно не срочно)
            • request_for_approval - запрос на согласование
            • notification - уведомление (ОТВЕТ НЕ ТРЕБУЕТСЯ)

            КАТЕГОРИИ СРОЧНОСТИ ОТВЕТА:
            - urgent: ответ в течение 1-2 дней (дедлайн просрочен или осталось меньше 3 дней)
            - high: ответ в течение 3-7 дней (3-7 дней до дедлайна)  
            - medium: ответ в течение 8-14 дней (8-14 дней до дедлайна)
            - low: ответ более чем через 15 дней (больше 14 дней до дедлайна)
            - no_deadline: дедлайн не указан (только для уведомлений)

            КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

            1. РАСЧЁТ ОТНОСИТЕЛЬНО СЕГОДНЯШНЕЙ ДАТЫ {formatted_date}:
               • ВСЕ расчёты делай относительно СЕГОДНЯШНЕЙ даты: {formatted_date}
               • Если письмо пришло {letter_date}, а сегодня {formatted_date} - учитывай эту разницу!
               • Если дедлайн уже прошёл относительно {formatted_date} → urgent
               • Если до дедлайна осталось мало времени → повышай срочность

            2. АНАЛИЗ РЕАЛЬНЫХ СРОКОВ:
               • Письмо пришло {letter_date}, сегодня {formatted_date}
               • Если в письме указано "ответить до 20.12.2025":
                 - От {letter_date} до 20.12.2025 = X дней (оригинальный срок)
                 - От {formatted_date} до 20.12.2025 = Y дней (реальный остаток)
                 - Используй Y для определения срочности!

            3. ПРИМЕРЫ РАСЧЁТА:
               • Письмо от 10.12.2025: "ответить до 20.12.2025"
                 - Сегодня 18.12.2025 → до дедлайна 2 дня → urgent
               • Письмо от 01.12.2025: "ответить в течение 10 дней" 
                 - Дедлайн = 11.12.2025
                 - Сегодня 18.12.2025 → дедлайн ПРОШЁЛ 7 дней назад → urgent
               • Письмо от 15.11.2025: "ответить до 25.12.2025"
                 - Сегодня 18.12.2025 → до дедлайна 7 дней → high

            4. УЧЁТ КАТЕГОРИИ:
               • ЖАЛОБЫ и РЕГУЛЯТОРНЫЕ ЗАПРОСЫ → всегда высокий приоритет
               • Если дедлайн не указан, но это ЖАЛОБА → high
               • УВЕДОМЛЕНИЯ → no_deadline (если нет явного дедлайна)

            5. КЛЮЧЕВЫЕ СЛОВА:
               • "немедленно", "срочно", "24 часа", "сегодня" → urgent
               • "3 дня", "до конца недели", "требуем" → high
               • "7 дней", "неделя" → medium  
               • "месяц", "когда удобно" → low
               • "уведомляем", "информируем" → no_deadline

            ПРИМЕРЫ ИЗ БАЗЫ ЗНАНИЙ:
            {examples_text}

            ТРЕБОВАНИЯ К ОТВЕТУ:
            — Строго в JSON-формате: {{
              "deadline": "...", 
              "reason": "...", 
              "deadline_date": "DD.MM.YYYY или null",
              "days_remaining": X
            }}
            — deadline: ТОЛЬКО ОДНО значение из списка категорий выше
            — reason: объяснение с расчётом дней от СЕГОДНЯШНЕЙ даты {formatted_date}
            — deadline_date: конкретная дата дедлайна (если можно определить), иначе null
            — days_remaining: сколько дней осталось до дедлайна ОТ СЕГОДНЯ (если есть дедлайн), иначе null

            ЗАПРЕЩЕНО:
            — Добавлять любые комментарии вне JSON
            — Использовать категории не из списка

            ВАЖНО: Всегда указывай в reason расчёт от СЕГОДНЯШНЕЙ даты!
            Пример: "Дедлайн 25.12.2025, от {formatted_date} осталось 7 дней → high"
        """

        return system_prompt.strip()

    def build_user_prompt(self, user_text: str) -> str:
        return f'Определи степень срочности ответа на письмо и напиши дату дедлайна:\n "{user_text}"'

    def classify(self, text: str, category: str, letter_date: str) -> str:
        retrieved = self.retrieve_examples(text)
        system_prompt = self.build_system_prompt(retrieved, category, letter_date)
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

        return response.choices[0].message.content.strip()


# TODO Классификатор дедлайнов ( побалуйся с промптами, чтобы оно в тексте и доках искало, когда надо овтетить на это письмо ) ( Артём ), погляди примерно структуру кода в category_classificator, клиент апи опенаи юзай из from app.llm.client import client