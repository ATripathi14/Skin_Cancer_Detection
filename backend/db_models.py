from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # Relationship to track user's prediction tasks
    prediction_tasks = relationship("PredictionTask", back_populates="owner", cascade="all, delete-orphan")

class PredictionTask(Base):
    __tablename__ = "prediction_tasks"

    task_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship back to the User
    owner = relationship("User", back_populates="prediction_tasks")
