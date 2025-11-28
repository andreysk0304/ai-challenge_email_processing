FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 celeryuser && \
    chown -R celeryuser:celeryuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R celeryuser:celeryuser /app

USER celeryuser

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info"]