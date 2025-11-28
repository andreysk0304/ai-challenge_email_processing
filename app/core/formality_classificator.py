import json

import chromadb
from chromadb import QueryResult

from app.constants import DOCUMENTS_JSON
from app.llm.client import client
from app.utils.config import FOLDER_ID


class FormalityClassificator:

    def __init__(self):
        self.client = client

        self.collection = self._init_vector_collection()

    def _init_vector_collection(self):
        chroma = chromadb.Client()

        collection = chroma.create_collection(
            name="formality_classifier_examples",
            embedding_function=None
        )
        formalities_examples = None
        with open(DOCUMENTS_JSON, encoding="utf-8") as f:
            formalities_examples = json.load(f)['formality']

        for i, (label, texts) in enumerate(formalities_examples.items()):
            for text in texts:
                collection.add(
                    ids=[str(i)],
                    documents=[text],
                    metadatas=[{"label": label}]
                )

        return collection

    def retrieve_examples(self, text: str) -> QueryResult:
        return self.collection.query(
            query_texts=[text],
            n_results=3
        )

    @staticmethod
    def build_system_prompt(user_text: str, retrieved: dict) -> str:
        examples_text = ""
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0]):
            examples_text += f"Формальность: {meta['label']}\nПример: {doc}\n\n"

        system_prompt = f"""
            Ты — классификатор корпоративных писем. 
            Твоя задача — определить стиль текста строго из списка ниже:

            1. strict_formal_style — Строгий официальный стиль — для регуляторов и государственных органов
            2. business_corporate_style — Деловой корпоративный стиль — для партнёров и контрагентов
            3. customer_oriented_option — Клиентоориентированный вариант — для физических и юридических лиц
            4. brief_informational_answer — Краткий информационный ответ — для простых запросов

            Никаких других стилей не существует.

            Используй примеры из RAG как основу классификации:

            {examples_text}

            Требования к ответу:
            — ответ строго в JSON-формате: {{"style": "...", "reason": "..."}}
            — style: только одно значение из списка категорий
            — reason: краткое объяснение (1–2 предложения) на основе сравнения с RAG
            — никаких комментариев вне JSON
            — никаких дополнительных слов
            — не создавай новые категории
        """

        return system_prompt.strip()

    @staticmethod
    def build_user_prompt(user_text: str) -> str:
        return f'Классифицируй текст:\n"{user_text}"'

    def classify(self, text: str) -> dict:
        retrieved = self.retrieve_examples(text)
        system_prompt = self.build_system_prompt(text, retrieved)
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
