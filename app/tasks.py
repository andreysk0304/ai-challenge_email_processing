from app.celery_app import celery_app
from app.core.database.client import session_maker
from app.core.database.models import Emails
from app.core.category_classificator import CategoryClassificator

from sqlalchemy import select


category_classifier = CategoryClassificator()

@celery_app.task(name="classify_email", max_retries=3)
def classify_email_task(payload: dict):
    email_id = payload["email_id"]

    with session_maker() as session:
        email = session.execute(select(Emails).where(Emails.id == email_id))
        email  = email.scalars().first()

        category: dict = category_classifier.classify(text=email.subject)

        email.category = category['category']
        email.reason = category['reason']

        session.commit()