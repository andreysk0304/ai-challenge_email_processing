import asyncio

from app.core.category_classificator import CategoryClassificator
from app.core.formality_classificator import FormalityClassificator


async def main() -> None:
    formality = FormalityClassificator()
    category = CategoryClassificator()

    text = "Требуется подтверждение действия для заказа №123457\nДля завершения операции изменения e-mail в вашем личном кабинете required подтверждение.\nКод подтверждения: 294817\nНикому не сообщайте этот код."


    formality = formality.classify(text=text)
    category = category.classify(text=text)

    print(formality)
    print(category)

    print(formality['style'])
    print(category['category'])


if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    asyncio.run(main())
