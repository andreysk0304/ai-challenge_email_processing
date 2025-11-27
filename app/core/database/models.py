from datetime import datetime

from app.utils.moscow_time import msk_now
from app.core.database.base import Base

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column


class Emails(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    subject: Mapped[str] = mapped_column(String(8192), nullable=True)
    status: Mapped[str] = mapped_column(String(128), default="new")
    category: Mapped[str] = mapped_column(String(128), nullable=True)
    reason: Mapped[str] = mapped_column(String(1024), nullable=True)
    deadline_time: Mapped[datetime] = mapped_column(DateTime(), nullable=True)
    formality: Mapped[str] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=msk_now)