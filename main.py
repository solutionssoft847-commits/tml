from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import cv2
import numpy as np
import io
from PIL import Image
import json
import uvicorn
from contextlib import asynccontextmanager
import time
from inspector_engine import AdvancedBlockInspector

# Initialize Inspector
inspector = AdvancedBlockInspector(yolo_model_path='yolo26n-obb.pt')

# history storage (in-memory for now, as per request to keep it simple but functional)
inspection_history = []
system_stats = {
    'total_scans': 0,
    'perfect_count': 0,
    'defected_count': 0,
    'last_24h': [] # List of {timestamp, status}
}

app = FastAPI()

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
            # Just keep connection alive, maybe send periodic heartbeats if needed
            # For now, we rely on broadcast from other events
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    # ... existing get_stats logic ...
    now = time.time()
    recent = [x for x in system_stats['last_24h'] if now - x['timestamp'] < 86400]
    system_stats['last_24h'] = recent
    return system_stats

@app.get("/api/history")
async def get_history():
    return inspection_history

@app.post("/api/inspect_upload")
async def inspect_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image"})
    
    result = inspector.inspect_block(image)
    
    # Update Stats
    system_stats['total_scans'] += 1
    if result.block_status == 'PERFECT':
        system_stats['perfect_count'] += 1
    else:
        system_stats['defected_count'] += 1
        
    system_stats['last_24h'].append({
        'timestamp': time.time(),
        'status': result.block_status
    })
    
    # Broadcast updates via WebSocket
    await manager.broadcast({
        "type": "stats_update",
        "data": system_stats
    })
    
    # Generate visualization
    if hasattr(inspector, 'last_saddles'):
        vis = inspector.visualize_results(image, inspector.last_saddles, result.saddle_results)
        _, buffer = cv2.imencode('.jpg', vis)
        import base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Add to history
        hist_item = {
            'id': system_stats['total_scans'],
            'timestamp': datetime.fromtimestamp(time.time()).isoformat(),
            'status': result.block_status,
            'defects': result.defective_saddles,
            'processing_time': result.processing_time_ms
        }
        inspection_history.insert(0, hist_item)
        if len(inspection_history) > 50:
            inspection_history.pop()

        # Broadcast history update as well
        await manager.broadcast({
            "type": "history_update",
            "data": [hist_item] # Send the new item or full list? Let's send list or just new item. 
                                # Frontend expects list usually. Let's send event type.
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
    # Simple check for first 4 indexes
    available = []
    current_state = camera is not None
    
    # We temporarily release to check others (not ideal but works for simple setup)
    # OR we just try to open them without releasing current if possible.
    # Windows DSHOW allows multiple opens sometimes, but let's be safe:
    # Just Assume 0 and 1 are valid or user provided.
    # Better approach: Try to open indexes 0-3 that aren't current.
    
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
    # validation happens in get_camera()
    return {"message": f"Switched to camera {index}", "current": index}

@app.post("/api/camera/stop")
async def stop_camera_endpoint():
    release_camera()
    return {"message": "Camera stopped"}

def generate_frames():
    cam = get_camera()
    if not cam.isOpened():
        # Re-try opening
        cam.open(current_camera_index)
        
    while True:
        if not cam.isOpened():
            # Return a blank frame or break
            break
            
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01) # Small delay to prevent CPU hogging

@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Main entry point moved to standard uvicorn run
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
from datetime import datetime