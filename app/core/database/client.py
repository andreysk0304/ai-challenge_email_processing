from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.utils.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    future=True
)

session_maker = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)