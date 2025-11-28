from datetime import datetime

from app.utils.moscow_time import msk_now
from app.core.database.base import Base

from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column


class Emails(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    message_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), index=True, nullable=True)

    from_email: Mapped[str] = mapped_column(String(320), index=True, nullable=False)
    to_email: Mapped[str] = mapped_column(String(320), nullable=True)

    subject: Mapped[str] = mapped_column(String(500), nullable=True)
    raw_body: Mapped[str] = mapped_column(Text, nullable=True)
    cleaned_body: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(128), default="new")
    category: Mapped[str] = mapped_column(String(128), nullable=True)
    reason: Mapped[str] = mapped_column(String(1024), nullable=True)

    deadline_time: Mapped[datetime] = mapped_column(DateTime(), nullable=True)

    formality: Mapped[str] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=msk_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=msk_now)