import chromadb
import json
from app.llm.client import client


class ContentAnalyzer:
    def __init__(self):
        self.client = client

    @staticmethod
    def build_system_prompt(category: str, formality_level: str) -> str:
        return f"""
            Ты — анализатор корпоративных писем в банковской сфере.

            КОНТЕКСТ АНАЛИЗА:
            - Категория письма: {category}
            - Уровень формальности: {formality_level}

            КАТЕГОРИИ ПИСЕМ (для понимания контекста):
            • information_request - запрос информации/документов
            • complaint - жалоба или претензия
            • regular_request - регуляторный запрос
            • partner_offer - партнёрское предложение
            • request_for_approval - запрос на согласование
            • notification - уведомление (ОТВЕТ НЕ ТРЕБУЕТСЯ)
            
            УРОВНИ ФОРМАЛНОСТЕЙ ПИСЕМ (для понимания контекста):
            • strict_formal_style — Строгий официальный стиль — для регуляторов и государственных органов
            • business_corporate_style — Деловой корпоративный стиль — для партнёров и контрагентов
            • customer_oriented_option — Клиентоориентированный вариант — для физических и юридических лиц
            • brief_informational_answer — Краткий информационный ответ — для простых запросов
            
            ТВОЯ ЗАДАЧА: Извлечь из текста письма информацию и вернуть строго в JSON-формате.

            ПРАВИЛА ИЗВЛЕЧЕНИЯ ДАННЫХ

            1. SUBJECT (суть запроса):
               - Извлеки основную тему письма
               - Сформулируй кратко (1-2 предложения)
               - Фокусируйся на главной проблеме или цели
               - Пример: "Нарушение условий договора №123: несвоевременное выполнение работ"
               - Если неясно → null

            2. REQUIREMENTS (требования и ожидания):
               - Что конкретно хочет отправитель?
               - Какие действия ожидает от банка?
               - Укажи сроки если есть
               - Пример: "Возврат средств в размере 100 000 руб до 25.12.2025"
               - Если требований нет → null

            3. CONTACT_INFO (контактные данные):
               - name: ФИО отправителя (только если явно указано)
               - organization: организация (только если явно указана)  
               - phone: телефоны в формате +7-XXX-XXX-XX-XX
               - email: email адреса
               - Если данных нет → null для каждого поля

            4. REQUISITES (реквизиты):
               - Только конкретные реквизиты из текста
               - Типы: "договор", "счет", "дата", "сумма", "акт" и т.д.
               - Пример: [{{"type": "договор", "value": "№БС-1456"}}, {{"type": "дата", "value": "15.03.2023"}}]
               - Если реквизитов нет → пустой массив []

            5. REGULATORY_REFERENCES (нормативные акты):
               - Только официальные названия законов, указаний, стандартов
               - Пример: [{{"type": "Указание Банка России", "value": "№55-У от 10.04.2024"}}]
               - Если ссылок нет → пустой массив []
               

            КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА
            ОБЯЗАТЕЛЬНО:
            - Возвращай ТОЛЬКО JSON без каких-либо комментариев
            - Используй только информацию из текста письма
            - Если данных нет - используй null или пустые массивы
            - Сохраняй точные формулировки из текста для реквизитов
            - НИЧЕГО СВОЕГО НЕ ПРИДУМЫВАТЬ, АНАЛИЗИРУЙ СТРОГО ПО КОНТЕКСТУ

            ЗАПРЕЩЕНО:
            - Добавлять поля которых нет в схеме
            - Придумывать информацию которой нет в тексте
            - Интерпретировать или предполагать
            - Менять структуру JSON
            - Оставлять пустые строки вместо null

            ФОРМАТ ОТВЕТА
            {{
                "subject": "текст или null",
                "requirements": "текст или null",
                "contact_info": {{
                    "name": "текст или null",
                    "organization": "текст или null", 
                    "phone": "текст или null",
                    "email": "текст или null"
                }},
                "requisites": [
                    {{"type": "тип реквизита", "value": "значение"}}
                ],
                "regulatory_references": [
                    {{"type": "тип документа", "value": "название"}}
                ]
            }}

            УЧТИ КОНТЕКСТ: Это {category} с {formality_level} уровнем формальности.
        """

    @staticmethod
    def build_user_prompt(text: str) -> str:
        return f"ПРОАНАЛИЗИРУЙ ПИСЬМО:\n\n{text}"

    def analyze_letter(self, text: str, category: str, formality_level: str) -> dict:
        system_prompt = self.build_system_prompt(category=category, formality_level=formality_level)
        user_prompt = self.build_user_prompt(text=text)

        response = self.client.chat.completions.create(
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result_text = response.choices[0].message.content.strip()

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return self._get_empty_response()

    @staticmethod
    def _get_empty_response() -> dict:
        return {
            "subject": None,
            "requirements": None,
            "contact_info": {
                "name": None,
                "organization": None,
                "phone": None,
                "email": None
            },
            "requisites": [],
            "regulatory_references": []
        }
        