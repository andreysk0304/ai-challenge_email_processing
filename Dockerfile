FROM python:3.11-slim

WORKDIR /app

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 celeryuser && \
    chown -R celeryuser:celeryuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Устанавливаем права на файлы для пользователя celeryuser
RUN chown -R celeryuser:celeryuser /app

# Переключаемся на непривилегированного пользователя
USER celeryuser

CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info"]