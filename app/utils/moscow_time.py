from datetime import datetime

from app.utils.config import MOSCOW_TZ

def msk_now():
    """Текущее время по МСК"""
    return datetime.now(MOSCOW_TZ)