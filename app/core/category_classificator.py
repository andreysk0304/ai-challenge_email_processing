import chromadb

from app.core.documents import CATEGORY_DOCUMENTS
from app.llm.client import client


class CategoryClassificator:

    def __init__(self):
        self.client = client

        self.collection = self._init_vector_collection()


    def _init_vector_collection(self):
        chroma = chromadb.Client()

        collection = chroma.create_collection(
            name="category_classifier_examples",
            embedding_function=None
        )

        for i, (label, text) in enumerate(CATEGORY_DOCUMENTS):
            collection.add(
                ids=[str(i)],
                documents=[text],
                metadatas=[{"label": label}]
            )

        return collection


    def retrieve_examples(self, text: str) -> dict:
        return self.collection.query(
            query_texts=[text],
            n_results=3
        )

    def build_system_prompt(self, user_text: str, retrieved: dict) -> str:
        examples_text = ""
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0]):
            examples_text += f"Категория: {meta['label']}\nПример: {doc}\n\n"

        system_prompt = f"""
            Ты — классификатор корпоративных писем. 
            Твоя задача — определить одну категорию текста строго из списка ниже:

            1. information_request
            2. complaint
            3. regular_request
            4. partner_offer
            5. request_for_approval
            6. notification

            Никаких других категорий не существует.

            Используй примеры из RAG как основу классификации:

            {examples_text}

            Требования к ответу:
            — ответ строго в JSON-формате: {{"category": "...", "reason": "..."}}
            — category: только одно значение из списка категорий
            — reason: краткое объяснение (1–2 предложения) на основе сравнения с RAG
            — никаких комментариев вне JSON
            — никаких дополнительных слов
            — не создавай новые категории
        """

        return system_prompt.strip()


    def build_user_prompt(self, user_text: str) -> str:
        return f'Классифицируй текст:\n"{user_text}"'


    async def classify(self, text: str) -> str:
        retrieved = self.retrieve_examples(text)
        system_prompt = self.build_system_prompt(text, retrieved)
        user_prompt = self.build_user_prompt(text)

        response = await self.client.chat.completions.create(
            model="gpt-5-nano",
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()