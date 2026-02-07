from fastapi import FastAPI, UploadFile, File, Request, WebSocket, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
import io
from PIL import Image
import json
import uvicorn
from contextlib import asynccontextmanager
import time
from datetime import datetime
from inspector_engine import AdvancedBlockInspector
from database import init_db, get_db, InspectionRecord, Stats, SessionLocal

# Initialize Inspector
inspector = AdvancedBlockInspector(yolo_model_path='yolo26n-obb.pt')

# In-memory fallback for last_24h (not persisted, resets on restart)
last_24h_cache = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database on startup
    init_db()
    # Ensure Stats row exists
    db = next(get_db())
    if db:
        stats = db.query(Stats).first()
        if not stats:
            stats = Stats(id=1, total_scans=0, perfect_count=0, defected_count=0)
            db.add(stats)
            db.commit()
        db.close()
    yield

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws/stats")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

def get_stats_from_db(db: Session):
    """Helper to get stats dict from database"""
    if db is None:
        return {'total_scans': 0, 'perfect_count': 0, 'defected_count': 0, 'last_24h': []}
    
    stats = db.query(Stats).first()
    if not stats:
        return {'total_scans': 0, 'perfect_count': 0, 'defected_count': 0, 'last_24h': []}
    
    # Filter last_24h_cache
    now = time.time()
    global last_24h_cache
    last_24h_cache = [x for x in last_24h_cache if now - x['timestamp'] < 86400]
    
    return {
        'total_scans': stats.total_scans,
        'perfect_count': stats.perfect_count,
        'defected_count': stats.defected_count,
        'last_24h': last_24h_cache
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    return get_stats_from_db(db)

@app.post("/api/update_status")
async def update_status(request: Request, db: Session = Depends(get_db)):
    
    data = await request.json()
    status = data.get('status', 'UNKNOWN')
    latency_ms = data.get('latency_ms', 0)
    defective_count = data.get('defective_count', 0)
    
    # Update Stats in DB
    if db:
        stats = db.query(Stats).first()
        if stats:
            stats.total_scans += 1
            if status == 'PERFECT':
                stats.perfect_count += 1
            elif status == 'DEFECTIVE':
                stats.defected_count += 1
            db.commit()
    
    # Update last_24h cache
    global last_24h_cache
    last_24h_cache.append({
        'timestamp': time.time(),
        'status': status
    })
    
    # Add to history in DB
    if db:
        record = InspectionRecord(
            timestamp=datetime.utcnow(),
            status=status,
            defects=defective_count,
            processing_time=latency_ms
        )
        db.add(record)
        db.commit()
    
    # Broadcast via WebSocket
    await manager.broadcast({
        "type": "stats_update",
        "data": get_stats_from_db(db)
    })
    await manager.broadcast({
        "type": "live_status",
        "data": {"status": status, "latency_ms": latency_ms}
    })
    
    return {"message": "Status received", "status": status}

@app.get("/health")
def health_check():
    """Health check for Render deployment"""
    return {"status": "ok", "service": "inspector-api"}

@app.get("/api/history")
async def get_history(db: Session = Depends(get_db)):
    if db is None:
        return []
    records = db.query(InspectionRecord).order_by(InspectionRecord.id.desc()).limit(50).all()
    return [
        {
            'id': r.id,
            'timestamp': r.timestamp.isoformat() if r.timestamp else None,
            'status': r.status,
            'defects': r.defects,
            'processing_time': r.processing_time
        }
        for r in records
    ]

@app.post("/api/inspect_upload")
async def inspect_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image"})
    
    result = inspector.inspect_block(image)
    
    # Update Stats in DB
    if db:
        stats = db.query(Stats).first()
        if stats:
            stats.total_scans += 1
            if result.block_status == 'PERFECT':
                stats.perfect_count += 1
            else:
                stats.defected_count += 1
            db.commit()
    
    # Update last_24h cache
    global last_24h_cache
    last_24h_cache.append({
        'timestamp': time.time(),
        'status': result.block_status
    })
    
    # Broadcast updates via WebSocket
    await manager.broadcast({
        "type": "stats_update",
        "data": get_stats_from_db(db)
    })
    
    # Generate visualization
    if hasattr(inspector, 'last_saddles'):
        vis = inspector.visualize_results(image, inspector.last_saddles, result.saddle_results)
        _, buffer = cv2.imencode('.jpg', vis)
        import base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Add to history in DB
        hist_item = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': result.block_status,
            'defects': result.defective_saddles,
            'processing_time': result.processing_time_ms
        }
        
        if db:
            record = InspectionRecord(
                timestamp=datetime.utcnow(),
                status=result.block_status,
                defects=result.defective_saddles,
                processing_time=result.processing_time_ms
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            hist_item['id'] = record.id

        # Broadcast history update
        await manager.broadcast({
            "type": "history_update",
            "data": [hist_item]
        })
            
        return {
            "result": result.to_dict(),
            "image": f"data:image/jpeg;base64,{img_str}",
            "history_item": hist_item
        }
        
    return {"result": result.to_dict()}

@app.post("/api/add_reference")
async def add_reference(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image"})
        
    success = inspector.add_reference_image(image)
    count = inspector.reference_manager.get_reference_count()
    return {"success": success, "count": count}

# Camera handling
camera = None
current_camera_index = 0

def get_camera():
    global camera, current_camera_index
    if camera is None:
        camera = cv2.VideoCapture(current_camera_index)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

@app.get("/api/cameras")
async def get_available_cameras():
    available = []
    current_state = camera is not None
    
    for i in range(4):
        if i == current_camera_index and current_state:
            available.append({"id": i, "name": f"Camera {i} (Active)"})
            continue
            
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append({"id": i, "name": f"Camera {i}"})
            cap.release()
    
    return available

@app.post("/api/camera/select")
async def select_camera_endpoint(request: Request):
    global current_camera_index
    data = await request.json()
    index = int(data.get("index", 0))
    
    release_camera()
    current_camera_index = index
    return {"message": f"Switched to camera {index}", "current": index}

@app.post("/api/camera/stop")
async def stop_camera_endpoint():
    release_camera()
    return {"message": "Camera stopped"}

def generate_frames():
    cam = get_camera()
    if not cam.isOpened():
        cam.open(current_camera_index)
        
    while True:
        if not cam.isOpened():
            break
            
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Main entry point
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)