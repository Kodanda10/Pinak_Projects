import datetime
import uuid
from typing import Optional  # Import Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    faiss_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    memory_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    redacted: Mapped[Optional[str]] = mapped_column(Text, default=None)

    def __repr__(self):
        return f"<Memory(id='{self.id}', content='{self.content[:20]}...')>"
