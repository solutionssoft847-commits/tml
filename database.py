from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")
# Handle Render's postgres:// vs postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

class InspectionRecord(Base):
    __tablename__ = "inspection_history"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String)
    defects = Column(JSON)
    processing_time = Column(Float)

class Stats(Base):
    __tablename__ = "system_stats"
    id = Column(Integer, primary_key=True)
    total_scans = Column(Integer, default=0)
    perfect_count = Column(Integer, default=0)
    defected_count = Column(Integer, default=0)

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)

def get_db():
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None
