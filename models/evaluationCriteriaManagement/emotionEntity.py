# not use in this version
from sqlalchemy import create_engine, Column, Integer, VARCHAR, DateTime
from sqlalchemy.orm import declarative_base

# Create the engine
engine = create_engine('sqlite:///monitoring-and-evaluation.db', echo=True)

# Create a base class for declarative models
Base = declarative_base()

# Define your models using the base class
class Emotion(Base):
    __tablename__ = 'Emotion'
    Timestamp = Column(DateTime, primary_key=True)
    Type = Column(VARCHAR(255), nullable=False)
    Amount = Column(Integer, nullable=False)
    Camera_ID = Column(Integer, nullable=False)