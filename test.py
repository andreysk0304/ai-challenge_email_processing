import asyncio

from app.core.category_classificator import CategoryClassificator
from app.core.formality_classificator import FormalityClassificator


async def main() -> None:
    formality = FormalityClassificator()
    category = CategoryClassificator()

    text =       "На основании п. 4.2 Указания Банка России №55-У от 10.04.2024 просим представить информацию о сделках с признаками возможного отмывания денежных средств за III квартал 2025 года до 25.11.2025 включительно."


    formality = formality.classify(text=text)
    category = category.classify(text=text)

    print(formality)
    print(category)

    print(formality['style'])
    print(category['category'])


if __name__ == '__main__':
    asyncio.run(main())
