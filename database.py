import os
import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, select, update, delete, desc

# Database Configuration
# Render uses postgres://, SQLAlchemy requires postgresql+asyncpg://
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./inspector.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Models
class InspectionHistoryRecord(Base):
    __tablename__ = 'inspection_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String)
    defects = Column(Integer)
    image = Column(Text)
    processing_time = Column(Float)

class SystemStatsRecord(Base):
    __tablename__ = 'system_stats'
    id = Column(String, primary_key=True) # 'stats'
    total_scans = Column(Integer, default=0)
    perfect_count = Column(Integer, default=0)
    defected_count = Column(Integer, default=0)

class CameraRecord(Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    type = Column(String)
    url = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class TemplateRecord(Base):
    __tablename__ = 'templates'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image = Column(Text) # Base64
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

async def init_db():
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSessionLocal() as session:
        # Initialize stats if not exists
        result = await session.execute(select(SystemStatsRecord).where(SystemStatsRecord.id == "stats"))
        stats = result.scalar_one_or_none()
        if not stats:
            stats = SystemStatsRecord(id="stats", total_scans=0, perfect_count=0, defected_count=0)
            session.add(stats)
            await session.commit()
    
    print(" PostgreSQL connected and initialized via SQLAlchemy")

class MongoDatabase: # Keeping the name to minimize changes in main.py, though it's now Postgres
    @staticmethod
    async def add_inspection_record(status, defects, image=None, processing_time=0):
        async with AsyncSessionLocal() as session:
            record = InspectionHistoryRecord(
                status=status,
                defects=defects,
                image=image,
                processing_time=processing_time,
                timestamp=datetime.now(timezone.utc)
            )
            session.add(record)
            
            # Update Stats
            result = await session.execute(select(SystemStatsRecord).where(SystemStatsRecord.id == "stats"))
            stats = result.scalar_one_or_none()
            if stats:
                stats.total_scans += 1
                if status.upper() in ["PERFECT", "PASS"]:
                    stats.perfect_count += 1
                elif status.upper() in ["DEFECTIVE", "FAIL"]:
                    stats.defected_count += 1
                session.add(stats)
            
            await session.commit()
            await session.refresh(record)
            
            return {
                "id": record.id,
                "timestamp": record.timestamp.isoformat() if record.timestamp else datetime.now(timezone.utc).isoformat(),
                "status": record.status,
                "defects": record.defects,
                "image": record.image,
                "processing_time": record.processing_time
            }

    @staticmethod
    async def get_stats():
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(SystemStatsRecord).where(SystemStatsRecord.id == "stats"))
            stats = result.scalar_one_or_none()
            if not stats:
                return {"total_scans": 0, "perfect_count": 0, "defected_count": 0}
            return {
                "total_scans": stats.total_scans,
                "perfect_count": stats.perfect_count,
                "defected_count": stats.defected_count
            }

    @staticmethod
    async def get_history(limit=50):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(InspectionHistoryRecord).order_by(desc(InspectionHistoryRecord.id)).limit(limit)
            )
            records = result.scalars().all()
            return [{
                "id": r.id,
                "_id": str(r.id), # Compatibility with frontend expecting _id
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "status": r.status,
                "defects": r.defects,
                "image": r.image,
                "processing_time": r.processing_time
            } for r in records]

    @staticmethod
    async def get_recent_scans(hours=24, limit=10):
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(InspectionHistoryRecord)
                .where(InspectionHistoryRecord.timestamp >= since)
                .order_by(desc(InspectionHistoryRecord.id))
                .limit(limit)
            )
            records = result.scalars().all()
            return [{
                "id": r.id,
                "_id": str(r.id),
                "timestamp": r.timestamp.timestamp() if r.timestamp else time.time(),
                "status": r.status,
                "defects": r.defects,
                "image": r.image,
                "processing_time": r.processing_time
            } for r in records]

    @staticmethod
    async def add_camera(name: str, type: str, url: str):
        async with AsyncSessionLocal() as session:
            camera = CameraRecord(
                name=name,
                type=type,
                url=url,
                created_at=datetime.now(timezone.utc)
            )
            session.add(camera)
            await session.commit()
            await session.refresh(camera)
            return {
                "id": camera.id,
                "_id": str(camera.id),
                "name": camera.name,
                "type": camera.type,
                "url": camera.url,
                "created_at": camera.created_at.isoformat() if camera.created_at else None
            }

    @staticmethod
    async def get_cameras():
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(CameraRecord).order_by(desc(CameraRecord.created_at)))
            cameras = result.scalars().all()
            return [{
                "id": c.id,
                "_id": str(c.id),
                "name": c.name,
                "type": c.type,
                "url": c.url,
                "created_at": c.created_at.isoformat() if c.created_at else None
            } for c in cameras]

    @staticmethod
    async def delete_camera(camera_id: int):
        async with AsyncSessionLocal() as session:
            await session.execute(delete(CameraRecord).where(CameraRecord.id == int(camera_id)))
            await session.commit()
            return True

    @staticmethod
    async def add_template(image_data: str):
        async with AsyncSessionLocal() as session:
            template = TemplateRecord(
                image=image_data,
                created_at=datetime.now(timezone.utc)
            )
            session.add(template)
            await session.commit()
            return True

    @staticmethod
    async def get_templates():
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(TemplateRecord).order_by(TemplateRecord.created_at))
            templates = result.scalars().all()
            return [t.image for t in templates]

    @staticmethod
    async def clear_templates():
        async with AsyncSessionLocal() as session:
            await session.execute(delete(TemplateRecord))
            await session.commit()
            return True
