from fastapi import FastAPI, UploadFile, File, Request, WebSocket, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image
import json
import uvicorn
from contextlib import asynccontextmanager
import time
import gc
from datetime import datetime, timedelta, timezone
from inspector_engine import AdvancedBlockInspector
from database import init_db, MongoDatabase
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from typing import Optional
import tempfile
import base64

# ==================== CONFIGURATION ====================
class Config:
    # Performance settings
    MAX_WORKERS = 4  # Thread pool size
    FRAME_QUEUE_SIZE = 5  # Max frames in processing queue
    RESULT_QUEUE_SIZE = 10
    
    # Camera settings
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1  
    
    # Paths
    TEMP_DIR = tempfile.gettempdir()  
    
    # HuggingFace
    HF_SPACE = os.environ.get('HF_SPACE')
    
    # Render detection
    IS_RENDER = os.environ.get('RENDER') is not None
    
    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8001",
        "https://*.onrender.com", 
        "*"  
    ]

config = Config()

# ==================== GLOBAL STATE ====================
inspector = None
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Multiprocessing queues
frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
result_queue = queue.Queue(maxsize=config.RESULT_QUEUE_SIZE)

# Thread control
processing_active = threading.Event()
processing_active.set()

# ==================== REMOTE INSPECTOR  ====================
class RemoteBlockInspector:
    
    def __init__(self, space_id):
        self.space_id = space_id
        self.client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from gradio_client import Client
            self.client = Client(self.space_id)
            print(f" Connected to Remote HF Inspector: {self.space_id}")
        except Exception as e:
            print(f"✗ Failed to connect to HF Space {self.space_id}: {e}")
            self.client = None
    
    async def inspect_block_async(self, image: np.ndarray):
        """Async inspection - non-blocking"""
        if not self.client:
            return None
        
        try:
            temp_path = os.path.join(config.TEMP_DIR, f"inspect_{int(time.time()*1000)}.jpg")
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None
            
            with open(temp_path, 'wb') as f:
                f.write(buffer.tobytes())
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                executor,
                self._sync_predict,
                temp_path
            )
            
            try:
                os.remove(temp_path)
            except:
                pass
            
            return result
        except Exception as e:
            print(f"✗ Remote inspection error: {e}")
            return None
    
    def _sync_predict(self, temp_path):
        from gradio_client import handle_file
        return self.client.predict(
            image=handle_file(temp_path),
            api_name="/predict"
        )
    
    def inspect_block(self, image: np.ndarray):
        return asyncio.run(self.inspect_block_async(image))

    # --- TEMPLATE MANAGEMENT (LOCAL FALLBACK) ---
    def _get_local(self):
        """Lazy load local inspector for template management features"""
        if not hasattr(self, '_local_inspector') or self._local_inspector is None:
            print(" 初始化本地检查器用于模板捕获 (Initialising local inspector for template capture)...")
            from inspector_engine import AdvancedBlockInspector
            self._local_inspector = AdvancedBlockInspector(yolo_model_path='yolo26n-obb.pt')
        return self._local_inspector

    def add_reference_image(self, image: np.ndarray):
        return self._get_local().add_reference_image(image)
    
    def get_reference_count(self):
        if not hasattr(self, '_local_inspector') or self._local_inspector is None:
            return 0
        return self._local_inspector.reference_manager.get_reference_count()

    def get_reference_images(self):
        if not hasattr(self, '_local_inspector') or self._local_inspector is None:
            return []
        return self._local_inspector.get_reference_images()

# ==================== INSPECTOR INITIALIZATION ====================
async def init_inspector():
    global inspector
    if config.HF_SPACE:
        inspector = RemoteBlockInspector(config.HF_SPACE)
    else:
        try:
            loop = asyncio.get_running_loop()
            inspector = await loop.run_in_executor(
                executor,
                lambda: AdvancedBlockInspector(yolo_model_path='yolo26n-obb.pt')
            )
            
            # Load persisted templates from database
            print(" Loading saved templates from database...")
            saved_templates = await MongoDatabase.get_templates()
            for i, img_b64 in enumerate(saved_templates):
                try:
                    img_data = base64.b64decode(img_b64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        success = await loop.run_in_executor(executor, inspector.add_reference_image, img)
                        if success:
                            print(f"  ✓ Template {i+1} loaded")
                        else:
                            print(f"  ✗ Failed to process saved template {i+1}")
                except Exception as e:
                    print(f"  ✗ Error loading template {i+1}: {e}")
                    
        except Exception as e:
            print(f"✗ Error initializing local inspector: {e}")
            raise e
    return inspector

def get_inspector():
    global inspector
    if inspector is None:
        raise RuntimeError("Inspector not initialized")
    return inspector

# ==================== MULTIPROCESSING PIPELINE ====================
# class FrameProcessor:
#     def __init__(self):
#         self.thread = None
#         self.running = False
#     
#     def start(self):
#         self.running = True
#         self.thread = threading.Thread(target=self._process_loop, daemon=True)
#         self.thread.start()
#         print(" Frame processor started")
#     
#     def stop(self):
#         self.running = False
#         if self.thread:
#             self.thread.join(timeout=2.0)
#     
#     def _process_loop(self):
#         while self.running and processing_active.is_set():
#             try:
#                 frame = frame_queue.get(timeout=0.1)
#                 start_time = time.time()
#                 insp = get_inspector()
#                 result = insp.inspect_live_frame(frame)
#                 
#                 if result is not None and result.block_status not in ['PENDING', 'WASTE_IMAGE']:
#                     processing_time = (time.time() - start_time) * 1000
#                     image_base64 = None
#                     if hasattr(insp, 'visualize_results'):
#                         try:
#                             vis = insp.visualize_results(frame, insp.last_saddles, result.saddle_results)
#                             _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                             img_str = base64.b64encode(buffer).decode('utf-8')
#                             image_base64 = f"data:image/jpeg;base64,{img_str}"
#                         except:
#                             pass
# 
#                     result_data = {
#                         'status': result.block_status,
#                         'latency_ms': processing_time,
#                         'defective_count': result.defective_saddles,
#                         'timestamp': time.time(),
#                         'result': result.to_dict(),
#                         'image': image_base64
#                     }
#                     try:
#                         result_queue.put_nowait(result_data)
#                     except queue.Full:
#                         try:
#                             result_queue.get_nowait()
#                             result_queue.put_nowait(result_data)
#                         except: pass
#                 frame_queue.task_done()
#             except queue.Empty: continue
#             except Exception as e:
#                 print(f"✗ Frame processing error: {e}")

# class AsyncResultBroadcaster:
#     def __init__(self, manager):
#         self.manager = manager
#         self.task = None
#         self.running = False
#     
#     def start(self):
#         self.running = True
#         self.task = asyncio.create_task(self._broadcast_loop())
#         print(" Async Result Broadcaster started")
#     
#     def stop(self):
#         self.running = False
#     
#     async def _broadcast_loop(self):
#         while self.running and processing_active.is_set():
#             try:
#                 result_data = await asyncio.to_thread(result_queue.get, timeout=0.1)
#                 record = await MongoDatabase.add_inspection_record(
#                     status=result_data['status'],
#                     defects=result_data['defective_count'],
#                     image=result_data.get('image'),
#                     processing_time=result_data['latency_ms']
#                 )
#                 result_data['id'] = record['id']
#                 await self._async_broadcast(result_data)
#                 result_queue.task_done()
#             except (queue.Empty, asyncio.TimeoutError):
#                 await asyncio.sleep(0.01)
#             except Exception as e:
#                 print(f"✗ Broadcast error: {e}")
#                 await asyncio.sleep(1)
#     
#     async def _async_broadcast(self, result_data):
#         try:
#             await self.manager.broadcast({
#                 "type": "live_status",
#                 "data": {
#                     "id": result_data.get('id'),
#                     "status": result_data['status'],
#                     "latency_ms": result_data['latency_ms'],
#                     "image": result_data.get('image')
#                 }
#             })
#             if result_data['status'] != 'PENDING':
#                 await self.manager.broadcast({
#                     "type": "history_update",
#                     "data": [{
#                         "id": result_data.get('id'),
#                         "timestamp": datetime.now(timezone.utc).isoformat(),
#                         "status": result_data['status'],
#                         "defects": result_data['defective_count'],
#                         "image": result_data.get('image'),
#                         "processing_time": result_data['latency_ms']
#                     }]
#                 })
#         except Exception as e:
#             print(f"✗ WebSocket broadcast error: {e}")

frame_processor = None
result_broadcaster = None

MAIN_LOOP = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_processor, result_broadcaster, MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    await init_db()
    await init_inspector()
    # frame_processor = FrameProcessor()
    # frame_processor.start()
    # result_broadcaster = AsyncResultBroadcaster(manager)
    # result_broadcaster.start()
    
    # Auto-start default camera (0)
    # try:
    #     print("Attempting to auto-start Camera 0...")
    #     await camera_manager.select_camera(0)
    #     # if camera_manager.is_camera_active():
    #     #     processing_active.set() # Trigger live inspection
    # except Exception as e:
    #     print(f"Auto-start camera failed: {e}")
        
    yield
    processing_active.clear()
    # if frame_processor: frame_processor.stop()
    # if result_broadcaster: result_broadcaster.stop()
    release_camera()
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock: self.active_connections.append(websocket)
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections: self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        async with self._lock:
            dead = []
            for connection in self.active_connections:
                try: await connection.send_json(message)
                except: dead.append(connection)
            for conn in dead: 
                if conn in self.active_connections: self.active_connections.remove(conn)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping": await websocket.send_json({"type": "pong"})
    except: pass
    finally: await manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/health")
async def health_check():
    db_ok = await MongoDatabase.check_connection()
    return {
        "status": "ok", 
        "inspector_ready": inspector is not None,
        "database_connected": db_ok,
        "environment": "Render/Cloud" if config.IS_RENDER else "Local/Edge"
    }

@app.get("/api/stats")
async def get_stats():
    stats = await MongoDatabase.get_stats()
    stats['last_24h'] = await MongoDatabase.get_recent_scans()
    return stats

@app.post("/api/update_status")
async def update_status(request: Request):
    data = await request.json()
    status = data.get('status', 'UNKNOWN')
    await MongoDatabase.add_inspection_record(status, data.get('defective_count', 0), None, data.get('latency_ms', 0))
    stats = await MongoDatabase.get_stats()
    stats['last_24h'] = await MongoDatabase.get_recent_scans()
    await manager.broadcast({"type": "stats_update", "data": stats})
    return {"message": "Status received"}

@app.get("/api/history")
async def get_history():
    return await MongoDatabase.get_history()

@app.post("/api/inspect_upload")
async def inspect_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: return JSONResponse(status_code=400, content={"message": "Invalid image"})
    insp = get_inspector()
    try:
        if isinstance(insp, RemoteBlockInspector):
            result = await asyncio.wait_for(insp.inspect_block_async(image), timeout=45.0)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, insp.inspect_block, image)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"message": "Inference timed out. Hugging Face Space might be sleeping or slow."})
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return JSONResponse(status_code=500, content={"message": f"Inference server error: {str(e)}"})

    if result is None:
        return JSONResponse(status_code=503, content={"message": "Inference server returned no data. Check HF Space status."})
    
    # ROBUST RESULT HANDLING (Gradio/Remote results can be [image_path, json_data])
    debug_result = result
    
    if isinstance(result, (tuple, list)):
        # Search for a dictionary or a JSON string in the list
        found_data = None
        for item in result:
            if isinstance(item, dict) and 'block_status' in item:
                found_data = item
                break
            if isinstance(item, str):
                try:
                    import json
                    parsed = json.loads(item)
                    if isinstance(parsed, dict) and 'block_status' in parsed:
                        found_data = parsed
                        break
                except:
                    continue
        
        # If we found a valid data bit, use it. Otherwise fallback to first item
        if found_data:
            result = found_data
        elif len(result) > 0:
            result = result[0]

    if isinstance(result, str):
        try:
            import json
            result = json.loads(result)
        except:
            if "/tmp/gradio" in result or result.endswith(('.webp', '.jpg', '.png')):
                print(f"  ⚠ Caught file path instead of data: {result}")
                # Try to find data in the OTHER parts of the debug_result if it was a list
                result = {"block_status": "ERROR", "message": "HF Space returned image path but no data found"}
            else:
                result = {"block_status": result if result else "UNKNOWN"}

    # Final conversion to dict
    if isinstance(result, dict):
        result_dict = result
    elif hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = {"block_status": "ERROR", "message": f"Invalid result type: {type(result)}"}
    
    block_status = result_dict.get('block_status', 'UNKNOWN')
    
    if block_status == 'WASTE_IMAGE':
        return JSONResponse(status_code=400, content={"message": "Rejected: Image too dark or likely waste", "result": result_dict})

    image_data = None
    if hasattr(insp, 'last_saddles'):
        vis = insp.visualize_results(image, insp.last_saddles, result.saddle_results)
        _, buffer = cv2.imencode('.jpg', vis)
        image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    
    record = await MongoDatabase.add_inspection_record(block_status, result_dict.get('defective_saddles', 0), image_data, result_dict.get('processing_time_ms', 0))
    stats = await MongoDatabase.get_stats()
    stats['last_24h'] = await MongoDatabase.get_recent_scans()
    await manager.broadcast({"type": "stats_update", "data": stats})
    
    hist_item = {
        'id': record['id'],
        'timestamp': record['timestamp'].isoformat() if isinstance(record['timestamp'], datetime) else record['timestamp'],
        'status': block_status,
        'defects': result_dict.get('defective_saddles', 0),
        'image': image_data,
        'processing_time': result_dict.get('processing_time_ms', 0)
    }
    await manager.broadcast({"type": "history_update", "data": [hist_item]})
    gc.collect()
    return {"result": result_dict, "image": image_data, "history_item": hist_item}

@app.post("/api/add_reference")
async def add_reference(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: return JSONResponse(status_code=400, content={"message": "Invalid image"})
    
    insp = get_inspector()
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(executor, insp.add_reference_image, image)
    
    if success:
        # Persist to database
        try:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            await MongoDatabase.add_template(img_b64)
            print(" ✓ Template persisted to database")
        except Exception as e:
            print(f" ⚠ Failed to persist template: {e}")
            
    return {"success": success, "count": insp.get_reference_count() if hasattr(insp, 'get_reference_count') else 0}

@app.get("/api/templates")
async def get_templates():
    insp = get_inspector()
    if hasattr(insp, 'get_reference_images'):
        return insp.get_reference_images()
    return []

from camera_manager import CameraManager
camera_manager = CameraManager(is_cloud=config.IS_RENDER, fps=config.CAMERA_FPS, buffer_size=config.CAMERA_BUFFER_SIZE)

@app.get("/api/cameras")
async def get_available_cameras():
    # Get physical USB cameras
    physical_cameras = await camera_manager.get_available_cameras()
    
    # Get saved cameras from DB
    saved_cameras = await MongoDatabase.get_cameras()
    
    # Format saved cameras
    formatted_saved = []
    for cam in saved_cameras:
        formatted_saved.append({
            "id": cam["url"], # Use URL as ID for selection
            "db_id": cam["id"], # Database ID for deletion
            "name": cam["name"],
            "type": cam["type"],
            "url": cam["url"],
            "is_saved": True
        })
        
    # Merge lists (avoid duplicates if any)
    # Priority: Saved cameras with custom names > Physical discovery
    return {
        "physical": physical_cameras,
        "saved": formatted_saved
    }

@app.post("/api/cameras/add")
async def add_camera_endpoint(request: Request):
    data = await request.json()
    name = data.get("name")
    url = data.get("url")
    cam_type = data.get("type", "ip")
    
    if not name or not url:
        return JSONResponse(status_code=400, content={"message": "Name and URL are required"})
        
    camera = await MongoDatabase.add_camera(name, cam_type, url)
    return {"message": "Camera saved successfully", "camera": camera}

@app.post("/api/camera/test")
async def test_camera_endpoint(request: Request):
    data = await request.json()
    url = data.get("url")
    if not url:
        return JSONResponse(status_code=400, content={"message": "URL is required"})
    
    # Run test in executor to avoid blocking main thread
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(executor, camera_manager.test_connection, url)
    
    if success:
        return {"message": "Connection successful", "status": "ok"}
    else:
        return JSONResponse(status_code=400, content={"message": "Failed to connect to camera", "status": "error"})

@app.delete("/api/cameras/{camera_id}")
async def delete_camera_endpoint(camera_id: int):
    success = await MongoDatabase.delete_camera(camera_id)
    if success:
        return {"message": "Camera deleted successfully"}
    return JSONResponse(status_code=404, content={"message": "Camera not found"})

@app.post("/api/camera/select")
async def select_camera_endpoint(request: Request):
    data = await request.json()
    camera_id = data.get("index") or data.get("url")
    if await camera_manager.select_camera(camera_id):
        # Broadcast status update
        await manager.broadcast({
            "type": "camera_status",
            "data": {"active": True, "id": camera_id}
        })
        return {"message": f"Switched to camera {camera_id}"}
    return JSONResponse(status_code=400, content={"message": "Failed to select camera"})

@app.post("/api/camera/stop")
async def stop_camera_endpoint():
    camera_manager.release_camera()
    # Broadcast status update
    await manager.broadcast({
        "type": "camera_status",
        "data": {"active": False, "id": None}
    })
    return {"message": "Camera stopped"}

def generate_frames():
    while True:
        # Get frame + capture flag
        frame, should_capture = camera_manager.get_frame()
        
        if frame is None:
            # If no camera, yield placeholder/black frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "NO SIGNAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue
            
        # If capture triggered by motion gate, send to inspector asynchronously
        # if should_capture:
        #     # Run inspection on the MAIN LOOP safely from this thread
        #     if MAIN_LOOP and not MAIN_LOOP.is_closed():
        #         asyncio.run_coroutine_threadsafe(
        #             async_process_and_broadcast(frame.copy()), 
        #             MAIN_LOOP
        #         )
        #     else:
        #         print("⚠ Main loop not available for inspection")

        # Encode for stream (JPEG Quality 95 as requested)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret: continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Limit stream FPS to save bandwidth (e.g. 15fps)
        time.sleep(0.06)

async def async_process_and_broadcast(frame):
    """Async helper to run inspection on main loop"""
    try:
        insp = get_inspector()
        
        result = None
        if isinstance(insp, RemoteBlockInspector):
            result = await insp.inspect_block_async(frame)
        else:
            # Run local inference in executor to avoid blocking main loop
            # But inspect_block is sync, so we need run_in_executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, insp.inspect_block, frame)
        
        result_dict = result if isinstance(result, dict) else result.to_dict()
        block_status = result_dict.get('block_status', 'UNKNOWN')
        
        # FILTER WASTE IMAGES
        if block_status == 'WASTE_IMAGE':
            # print(" [IGNORED] Waste image detected")
            return

        image_data = None
        if hasattr(insp, 'last_saddles'):
            vis = insp.visualize_results(frame, insp.last_saddles, result.saddle_results)
            _, buffer = cv2.imencode('.jpg', vis)
            image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        # Save record and broadcast updates
        await save_and_broadcast_inspection_result(block_status, result_dict, image_data)
        
    except Exception as e:
        print(f"Background inspection failed: {e}")

async def save_and_broadcast_inspection_result(block_status: str, result_dict: dict, image_data: Optional[str]):
    """Saves inspection record and broadcasts updates asynchronously."""
    record = await MongoDatabase.add_inspection_record(
        block_status,
        result_dict.get('defective_saddles', 0),
        image_data,
        result_dict.get('processing_time_ms', 0)
    )
    
    stats = await MongoDatabase.get_stats()
    stats['last_24h'] = await MongoDatabase.get_recent_scans()
    await manager.broadcast({"type": "stats_update", "data": stats})
    
    hist_item = {
        'id': record['id'],
        'timestamp': record['timestamp'].isoformat() if isinstance(record['timestamp'], datetime) else record['timestamp'],
        'status': block_status,
        'defects': result_dict.get('defective_saddles', 0),
        'image': image_data,
        'processing_time': result_dict.get('processing_time_ms', 0)
    }
    await manager.broadcast({"type": "history_update", "data": [hist_item]})
    gc.collect()

@app.get("/api/video_feed")
async def video_feed():
    """Video streaming route"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def release_camera():
    camera_manager.release_camera()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)