from celery import Celery

from app.utils.config import REDIS_URL

celery_app = Celery(
    "llm_workers",
    broker=REDIS_URL
)

celery_app.conf.update(
    task_acks_late=True,
    broker_transport_options={"visibility_timeout": 3600},
)