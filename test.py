from app.core.category_classificator import CategoryClassificator

import json
import asyncio



async def main() -> None:
    category_cls = CategoryClassificator()

    data = category_cls.classify(text='Привет, у меня есть 1к рублей я готов вам выделить их на рекламу')

    print(data)

    print(data['category'])


if __name__ == '__main__':

    asyncio.run(main())