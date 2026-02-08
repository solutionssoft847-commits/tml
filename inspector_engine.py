import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import torch
import torch.onnx
import torchvision
from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import warnings

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class SaddleROI:
    """Saddle region of interest with rotation info"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    crop: np.ndarray
    center: Tuple[int, int]
    area: int
    angle: float  # Rotation angle from YOLO-OBB
    confidence: float  # Detection confidence
    
    def to_dict(self):
        return {
            'id': int(self.id),
            'bbox': [int(x) for x in self.bbox],
            'center': [int(self.center[0]), int(self.center[1])],
            'area': int(self.area),
            'angle': float(self.angle),
            'confidence': float(self.confidence)
        }


def export_resnet18_onnx():
    model = OptimizedFeatureExtractor().model
    model.eval()
    
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4
    
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18_features_fp16.onnx",
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=17
    )
    
    # Convert to FP16 for 2× speedup
    import onnx
    from onnxconverter_common import float16
    
    model_fp32 = onnx.load("resnet18_features_fp16.onnx")
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, "resnet18_features_fp16.onnx")

@dataclass
class SaddleInspectionResult:
    """Inspection result for one saddle"""
    saddle_id: int
    status: str
    similarity_score: float
    confidence: float
    feature_distance: float
    detected: bool  # Was it detected or interpolated?
    
    def to_dict(self):
        return {
            'saddle_id': int(self.saddle_id),
            'status': str(self.status),
            'similarity_score': float(self.similarity_score),
            'confidence': float(self.confidence),
            'feature_distance': float(self.feature_distance),
            'detected': bool(self.detected)
        }


@dataclass
class BlockInspectionResult:
    """Complete block inspection result"""
    block_status: str
    total_saddles: int
    detected_saddles: int
    defective_saddles: int
    saddle_results: List[SaddleInspectionResult]
    processing_time_ms: float
    alignment_status: str
    block_angle: float
    
    def to_dict(self):
        return {
            'block_status': str(self.block_status),
            'total_saddles': int(self.total_saddles),
            'detected_saddles': int(self.detected_saddles),
            'defective_saddles': int(self.defective_saddles),
            'saddle_results': [r.to_dict() for r in self.saddle_results],
            'processing_time_ms': float(self.processing_time_ms),
            'alignment_status': str(self.alignment_status),
            'block_angle': float(self.block_angle)
        }


class YOLOOBBDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.available = YOLO_AVAILABLE
        
        # Use provided path or default to v11/v8
        target_path = model_path if model_path else 'yolo26n-obb.pt'
        
        if self.available:
            try:
                self.model = YOLO(target_path)
                self.model_loaded = True
                print(f" ✓ YOLO-OBB loaded: {target_path}")
            except Exception as e:
                try:
                    # Fallback to alternative
                    fallback = 'yolo11n-obb.pt' if target_path == 'yolo26n-obb.pt' else 'yolo26n-obb.pt'
                    self.model = YOLO(fallback)
                    self.model_loaded = True
                    print(f" ✓ YOLO-OBB fallback loaded: {fallback}")
                except:
                    print(f" ✗ YOLO load failed: {e}")
                    self.model_loaded = False
                    self.available = False
        else:
            print(" ⚠ YOLO not available (check ultralytics installation)")


    def detect_saddles(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[SaddleROI]:
        """
        Detect saddles using YOLO-OBB
        
        Returns: List of SaddleROI with rotation angles
        """
        
        if not self.available or self.model is None:
            return []
        
        # Convert to grayscale for faster inference (your suggestion)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Run YOLO-OBB
        results = self.model(gray_3ch, conf=conf_threshold, verbose=False)
        
        saddles = []
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # Oriented bounding boxes
                for i, obb in enumerate(result.obb):
                    # Extract OBB data
                    xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy()  # 4 corner points
                    conf = float(obb.conf[0])
                    cls = int(obb.cls[0])
                    
                    # Calculate rotation angle
                    angle = self._calculate_angle(xyxyxyxy)
                    
                    # Get axis-aligned bounding box for cropping
                    x_coords = xyxyxyxy[:, 0]
                    y_coords = xyxyxyxy[:, 1]
                    
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Center
                    cx = int(np.mean(x_coords))
                    cy = int(np.mean(y_coords))
                    
                    # Crop
                    crop = image[y_min:y_max, x_min:x_max].copy()
                    
                    # Area
                    area = w * h
                    
                    saddles.append(SaddleROI(
                        id=i,
                        bbox=(x_min, y_min, w, h),
                        crop=crop,
                        center=(cx, cy),
                        area=area,
                        angle=angle,
                        confidence=conf
                    ))
            
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Fallback to regular boxes if OBB not available
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    
                    crop = image[y_min:y_max, x_min:x_max].copy()
                    area = w * h
                    
                    saddles.append(SaddleROI(
                        id=i,
                        bbox=(x_min, y_min, w, h),
                        crop=crop,
                        center=(cx, cy),
                        area=area,
                        angle=0.0,  # No angle info
                        confidence=conf
                    ))
        
        # Sort by position (top-left to bottom-right)
        saddles.sort(key=lambda s: (s.center[1], s.center[0]))
        
        # Reassign IDs
        for i, saddle in enumerate(saddles):
            saddle.id = i
        
        print(f"✓ YOLO detected: {len(saddles)} saddles")
        
        return saddles
    
    def _calculate_angle(self, xyxyxyxy: np.ndarray) -> float:
        """Calculate rotation angle from OBB corners"""
        
        # xyxyxyxy: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Calculate angle from first edge
        p1 = xyxyxyxy[0]
        p2 = xyxyxyxy[1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize to [-180, 180]
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        
        return float(angle)


class GeometricSaddleDetector:
    """
    Fallback geometric detector (from your original code)
    Used when YOLO is not available
    """
    
    def __init__(self):
        pass
    
    def detect_saddles(self, image: np.ndarray) -> List[SaddleROI]:
        """Detect using geometric segmentation"""
        
        print(f"[GeometricDetector] Input image shape: {image.shape}")
        
        h, w = image.shape[:2]
        
        # Detect circles
        outer_circle, inner_circle = self._detect_circles(image)
        
        if outer_circle is None:
            print("[GeometricDetector] ✗ Circle detection failed completely")
            return []
        
        ox, oy, or_ = outer_circle
        ix, iy, ir = inner_circle
        
        print(f"[GeometricDetector] Outer circle: center=({ox},{oy}), radius={or_}")
        print(f"[GeometricDetector] Inner circle: center=({ix},{iy}), radius={ir}")
        
        # Validate circles are reasonable
        if or_ < 50 or or_ > min(h, w):
            print(f"[GeometricDetector] ✗ Invalid outer radius: {or_}")
            return []
        
        if ir >= or_:
            print(f"[GeometricDetector] ✗ Inner radius >= outer radius")
            ir = int(or_ * 0.4)
            print(f"[GeometricDetector] Adjusted inner radius to {ir}")
        
        # Create annular mask
        annular_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(annular_mask, (ox, oy), or_, 255, -1)
        cv2.circle(annular_mask, (ix, iy), ir, 0, -1)
        
        # Check if mask has content
        mask_pixels = np.sum(annular_mask > 0)
        print(f"[GeometricDetector] Annular mask pixels: {mask_pixels}")
        
        if mask_pixels < 1000:
            print(f"[GeometricDetector] ✗ Mask too small")
            return []
        
        # 4 quadrants
        quadrants = [
            (0, 90, "Top-Right"),
            (90, 180, "Top-Left"),
            (180, 270, "Bottom-Left"),
            (270, 360, "Bottom-Right")
        ]
        
        saddles = []
        
        for idx, (start_angle, end_angle, name) in enumerate(quadrants):
            sector_mask = self._create_sector_mask(
                (h, w), (ox, oy), or_, ir, start_angle, end_angle
            )
            
            saddle_mask = cv2.bitwise_and(annular_mask, sector_mask)
            
            ys, xs = np.where(saddle_mask > 0)
            
            if len(xs) == 0 or len(ys) == 0:
                print(f"[GeometricDetector] ⚠ Quadrant {idx} ({name}): No pixels")
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # Validate bounding box
            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                print(f"[GeometricDetector] ⚠ Quadrant {idx} ({name}): Box too small")
                continue
            
            crop = image[y_min:y_max+1, x_min:x_max+1].copy()
            mask_crop = saddle_mask[y_min:y_max+1, x_min:x_max+1].copy()
            crop_masked = cv2.bitwise_and(crop, crop, mask=mask_crop)
            
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            area = np.sum(saddle_mask > 0)
            
            saddles.append(SaddleROI(
                id=idx,
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                crop=crop_masked,
                center=(cx, cy),
                area=area,
                angle=0.0,
                confidence=1.0
            ))
            
            print(f"[GeometricDetector] ✓ Saddle {idx} ({name}): center=({cx},{cy}), area={area}")
        
        print(f"[GeometricDetector] Total saddles detected: {len(saddles)}")
        
        return saddles
    
    def _detect_circles(self, image: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Detect outer and inner circles - OPTIMIZED (no loop).
        Uses single HoughCircles call + contour fallback for speed.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Single optimized parameter set (balanced for most cases)
        minRadius = int(min(h, w) * 0.25)
        maxRadius = int(min(h, w) * 0.65)
        
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=80,
            param1=80,
            param2=25,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
        
        if circles is not None and len(circles[0]) >= 1:
            circles = np.round(circles[0, :]).astype(int)
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            outer = tuple(circles[0])
            inner_radius = int(outer[2] * 0.4)
            inner = (outer[0], outer[1], inner_radius)
            print(f"  [OK] Detected: outer={outer}, inner={inner}")
            return outer, inner
        
        # Contour-based fallback (faster than trying multiple HoughCircles params)
        print("  [WARN] HoughCircles failed - using contour fallback")
        edges = cv2.Canny(gray_blur, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            center_x, center_y = int(cx), int(cy)
            outer_radius = int(radius * 0.9)
            inner_radius = int(outer_radius * 0.4)
            print(f"  Using contour estimate: center=({center_x},{center_y}), radius={outer_radius}")
            return ((center_x, center_y, outer_radius), (center_x, center_y, inner_radius))
        
        # Last resort: image center
        center_x, center_y = w // 2, h // 2
        outer_radius = min(w, h) // 2 - 50
        inner_radius = int(outer_radius * 0.4)
        print(f"  Using dimension-based estimate: center=({center_x},{center_y}), radius={outer_radius}")
        return ((center_x, center_y, outer_radius), (center_x, center_y, inner_radius))
    
    def _create_sector_mask(self, shape, center, outer_r, inner_r, start_angle, end_angle):
        """Create sector mask"""
        
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        y, x = np.ogrid[:h, :w]
        dx = x - center[0]
        dy = y - center[1]
        
        angles = np.rad2deg(np.arctan2(dy, dx)) % 360
        radius = np.sqrt(dx**2 + dy**2)
        
        if start_angle < end_angle:
            angle_mask = (angles >= start_angle) & (angles < end_angle)
        else:
            angle_mask = (angles >= start_angle) | (angles < end_angle)
        
        radius_mask = (radius >= inner_r) & (radius <= outer_r)
        sector = angle_mask & radius_mask
        mask[sector] = 255
        
        return mask


class AlignmentCorrector:
    """
    Affine transformation for rotation correction
    
    Your suggestion: Use YOLO-OBB angle θ to rotate image to 0°
    This ensures saddles are always in same orientation as reference
    """
    
    @staticmethod
    def correct_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by -angle to normalize orientation
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (from YOLO-OBB)
        
        Returns:
            Rotated image
        """
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix for -angle
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        
        return rotated
    
    @staticmethod
    def correct_saddle(saddle: SaddleROI) -> np.ndarray:
        """Rotate individual saddle to 0° orientation"""
        
        if abs(saddle.angle) < 1.0:  # Already aligned
            return saddle.crop
        
        return AlignmentCorrector.correct_rotation(saddle.crop, saddle.angle)


class CNNFeatureExtractor:
    """
    ResNet-18 ONNX feature extraction for deep similarity
    
    Optimizations:
    1. Uses ONNX Runtime instead of PyTorch (faster CPU inference)
    2. ResNet-18 instead of ResNet-50 (11M vs 25M params)
    3. Supports batch processing for multiple saddles at once
    4. 512-dim embeddings (vs 2048-dim for ResNet-50)
    """
    
    def __init__(self, onnx_path: str = "resnet18_features.onnx"):
        self.onnx_path = onnx_path
        self.session = None
        self.feature_dim = 512  # ResNet-18 feature dimension
        
        # Try to load ONNX model
        try:
            import onnxruntime as ort
            
            # Use CPU with optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # Parallel within ops
            
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            print(f"[OK] ONNX ResNet-18 loaded: {onnx_path}")
            
        except Exception as e:
            print(f"[WARN] ONNX load failed: {e}, falling back to PyTorch")
            self._init_pytorch_fallback()
    
    def _init_pytorch_fallback(self):
        """Fallback to PyTorch if ONNX not available"""
        from torchvision.models import resnet18, ResNet18_Weights
        
        self.device = torch.device('cpu')
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        self.session = None  # Mark as using PyTorch
        print("[OK] PyTorch ResNet-18 fallback initialized")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image for inference"""
        # Handle edge cases
        if image is None or image.size == 0:
            return np.zeros((3, 224, 224), dtype=np.float32)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Normalize (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        return image
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 512-dim feature embedding for a single image
        
        Returns: L2-normalized feature vector
        """
        batch = self._preprocess_image(image)[np.newaxis, ...]  # Add batch dim
        features = self.extract_batch(batch)
        return features[0]
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract features for a batch of images (optimized for 4 saddles)
        
        Args:
            images: List of images (raw numpy arrays) or pre-processed batch tensor
        
        Returns: (N, 512) array of L2-normalized feature vectors
        """
        # If already a tensor, use directly
        if isinstance(images, np.ndarray) and len(images.shape) == 4:
            batch = images.astype(np.float32)
        else:
            # Preprocess each image and stack
            preprocessed = [self._preprocess_image(img) for img in images]
            batch = np.stack(preprocessed, axis=0).astype(np.float32)
        
        if self.session is not None:
            # ONNX Runtime inference
            outputs = self.session.run(None, {'input': batch})
            features = outputs[0]  # Shape: (N, 512, 1, 1)
            features = features.reshape(features.shape[0], -1)  # (N, 512)
        else:
            # PyTorch fallback
            import torch
            batch_tensor = torch.from_numpy(batch).to(self.device)
            with torch.no_grad():
                features = self.model(batch_tensor)
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
        
        # L2 normalize each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Cosine similarity between two feature vectors
        
        Formula: cos(theta) = (A . B) / (||A|| ||B||)
        
        Returns: Similarity in [0, 1]
        """
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
        )
        
        return float(np.clip(similarity, 0, 1))


class GeometricConstraintValidator:
    """
    Validate saddle positions using geometric constraints
    
    Your suggestion: Use rigid body constraints to infer missing saddles
    """
    
    def __init__(self, expected_count: int = 4):
        self.expected_count = expected_count
        self.reference_positions = None  # Will be set from golden template
    
    def set_reference_positions(self, saddles: List[SaddleROI]):
        """Learn saddle positions from reference image"""
        
        if len(saddles) != self.expected_count:
            print(f"⚠ Reference has {len(saddles)} saddles, expected {self.expected_count}")
            return False
        
        # Store relative positions
        centers = [s.center for s in saddles]
        
        # Calculate centroid
        centroid = (
            int(np.mean([c[0] for c in centers])),
            int(np.mean([c[1] for c in centers]))
        )
        
        # Store relative positions from centroid
        self.reference_positions = [
            (c[0] - centroid[0], c[1] - centroid[1]) for c in centers
        ]
        
        # Store pairwise distances
        self.reference_distances = {}
        for i in range(len(saddles)):
            for j in range(i+1, len(saddles)):
                dist = np.linalg.norm(
                    np.array(saddles[i].center) - np.array(saddles[j].center)
                )
                self.reference_distances[(i, j)] = dist
        
        print(f"✓ Reference geometry learned: {len(saddles)} saddles")
        return True
    
    def infer_missing_saddles(
        self, 
        detected_saddles: List[SaddleROI],
        image: np.ndarray
    ) -> List[SaddleROI]:
        """
        Infer positions of missing saddles using geometric constraints
        
        Your suggestion: Force crop at calculated coordinates
        """
        
        if len(detected_saddles) >= self.expected_count:
            return detected_saddles  # All found
        
        if self.reference_positions is None:
            print("⚠ No reference geometry - cannot infer")
            return detected_saddles
        
        if len(detected_saddles) < 2:
            print("⚠ Need at least 2 saddles to infer")
            return detected_saddles
        
        # Calculate current centroid
        centers = [s.center for s in detected_saddles]
        centroid = (
            int(np.mean([c[0] for c in centers])),
            int(np.mean([c[1] for c in centers]))
        )
        
        # Find scale factor (compare distances)
        scale = self._estimate_scale(detected_saddles)
        
        # Predict missing positions
        detected_ids = {s.id for s in detected_saddles}
        all_saddles = list(detected_saddles)
        
        for expected_id in range(self.expected_count):
            if expected_id not in detected_ids:
                # Calculate expected position
                rel_x, rel_y = self.reference_positions[expected_id]
                
                expected_x = int(centroid[0] + rel_x * scale)
                expected_y = int(centroid[1] + rel_y * scale)
                
                # Force crop at expected position
                crop_size = 100  # Estimate
                
                x_min = max(0, expected_x - crop_size // 2)
                y_min = max(0, expected_y - crop_size // 2)
                x_max = min(image.shape[1], expected_x + crop_size // 2)
                y_max = min(image.shape[0], expected_y + crop_size // 2)
                
                crop = image[y_min:y_max, x_min:x_max].copy()
                
                # Create inferred saddle
                inferred = SaddleROI(
                    id=expected_id,
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    crop=crop,
                    center=(expected_x, expected_y),
                    area=(x_max - x_min) * (y_max - y_min),
                    angle=0.0,
                    confidence=0.0  # Mark as inferred
                )
                
                all_saddles.append(inferred)
                
                print(f"  ⚙ Inferred Saddle {expected_id} at ({expected_x}, {expected_y})")
        
        # Sort by ID
        all_saddles.sort(key=lambda s: s.id)
        
        return all_saddles
    
    def _estimate_scale(self, saddles: List[SaddleROI]) -> float:
        """Estimate scale factor from detected saddles"""
        
        if len(saddles) < 2 or not self.reference_distances:
            return 1.0
        
        # Compare first two saddles
        if len(saddles) >= 2:
            s0, s1 = saddles[0], saddles[1]
            current_dist = np.linalg.norm(
                np.array(s0.center) - np.array(s1.center)
            )
            
            key = (min(s0.id, s1.id), max(s0.id, s1.id))
            if key in self.reference_distances:
                ref_dist = self.reference_distances[key]
                scale = current_dist / ref_dist
                return scale
        
        return 1.0
    
    def validate_geometry(self, saddles: List[SaddleROI], tolerance: float = 0.05) -> Tuple[bool, str]:
        """
        Validate that detected saddles form the expected geometric pattern.
        
        Checks:
        1. Saddles form a straight line (collinearity check)
        2. Distances between adjacent saddles are within ±tolerance of golden distances
        
        Args:
            saddles: List of detected saddles (should be 4)
            tolerance: Allowed deviation from golden distance (default 5%)
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        if len(saddles) != self.expected_count:
            return False, f"Expected {self.expected_count} saddles, got {len(saddles)}"
        
        if self.reference_distances is None or not self.reference_distances:
            return True, "No reference geometry to validate against"
        
        # Sort saddles by ID to ensure consistent ordering
        sorted_saddles = sorted(saddles, key=lambda s: s.id)
        centers = [np.array(s.center) for s in sorted_saddles]
        
        # Check 1: Collinearity - all points should lie on a line
        # Use cross product to check if points are collinear
        if len(centers) >= 3:
            v1 = centers[1] - centers[0]  # Vector from saddle 0 to 1
            for i in range(2, len(centers)):
                v2 = centers[i] - centers[0]  # Vector from saddle 0 to i
                # Cross product in 2D: v1.x * v2.y - v1.y * v2.x
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                line_length = np.linalg.norm(v1) * np.linalg.norm(v2)
                # Normalize by line length to get deviation ratio
                if line_length > 0:
                    deviation = cross / line_length
                    if deviation > 0.1:  # 10% deviation from line
                        return False, f"Saddles not collinear (deviation: {deviation:.2f})"
        
        # Check 2: Distance validation - compare adjacent saddle distances
        for i in range(len(sorted_saddles) - 1):
            s0, s1 = sorted_saddles[i], sorted_saddles[i + 1]
            current_dist = np.linalg.norm(centers[i + 1] - centers[i])
            
            key = (min(s0.id, s1.id), max(s0.id, s1.id))
            if key in self.reference_distances:
                ref_dist = self.reference_distances[key]
                if ref_dist > 0:
                    deviation = abs(current_dist - ref_dist) / ref_dist
                    if deviation > tolerance:
                        return False, f"Distance S{s0.id}-S{s1.id} off by {deviation*100:.1f}% (max: {tolerance*100}%)"
        
        return True, "Geometry valid"


class MultiReferenceManager:
    """
    Manage multiple golden template images
    
    Your suggestion: Allow multiple reference images for better matching
    """
    
    def __init__(self, feature_extractor: CNNFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.references = []  # List of (image, features_per_saddle)
        
    def add_reference(self, image: np.ndarray, saddles: List[SaddleROI]) -> bool:
        """Add a golden template"""
        
        if len(saddles) != 4:
            print(f"⚠ Reference must have 4 saddles, got {len(saddles)}")
            return False
        
        # Extract features for each saddle
        features_list = []
        
        for saddle in saddles:
            # Align before feature extraction
            aligned_crop = AlignmentCorrector.correct_saddle(saddle)
            features = self.feature_extractor.extract_features(aligned_crop)
            features_list.append(features)
        
        # Store
        self.references.append({
            'image': image,
            'saddles': saddles,
            'features': features_list
        })
        
        print(f"✓ Reference {len(self.references)} added")
        return True
    
    def match_saddle(self, saddle: SaddleROI) -> Tuple[float, float]:
        """
        Match saddle against all references, return best match
        
        Returns: (best_similarity, avg_distance)
        """
        
        if not self.references:
            return 0.0, 999.0
        
        # Align saddle
        aligned_crop = AlignmentCorrector.correct_saddle(saddle)
        
        # Extract features
        test_features = self.feature_extractor.extract_features(aligned_crop)
        
        # Compare against all references for this saddle ID
        similarities = []
        distances = []
        
        for ref in self.references:
            if saddle.id < len(ref['features']):
                ref_features = ref['features'][saddle.id]
                
                sim = self.feature_extractor.compute_similarity(test_features, ref_features)
                dist = np.linalg.norm(test_features - ref_features)
                
                similarities.append(sim)
                distances.append(dist)
        
        # Return best match
        if similarities:
            best_sim = max(similarities)
            avg_dist = np.mean(distances)
            return best_sim, avg_dist
        
        return 0.0, 999.0
    
    def get_reference_count(self) -> int:
        """Number of reference images"""
        return len(self.references)


class AdvancedBlockInspector:
    """
    Advanced Engine Block Inspector - OPTIMIZED
    
    Features:
    1. YOLO-OBB detection with rotation angles
    2. Affine transformation for alignment
    3. Multiple reference images (10-15 recommended)
    4. Geometric constraint validation
    5. ResNet-18 ONNX feature matching (batch processing)
    6. Frame skipping for live video (5 consecutive failures triggers geometric fallback)
    """
    
    def __init__(self, yolo_model_path: Optional[str] = None, onnx_path: str = "resnet18_features.onnx"):
        # Detectors
        self.yolo_detector = YOLOOBBDetector(yolo_model_path)
        self.geometric_detector = GeometricSaddleDetector()
        
        # Feature extraction (ONNX-based)
        self.cnn_extractor = CNNFeatureExtractor(onnx_path=onnx_path)
        
        # Multi-reference manager
        self.reference_manager = MultiReferenceManager(self.cnn_extractor)
        
        # Geometric validator
        self.geo_validator = GeometricConstraintValidator(expected_count=4)
        
        # Frame skipping state for live video
        self.consecutive_failures = 0
        self.max_skip_frames = 5  # Trigger geometric fallback after 5 consecutive failures
        
        # Sliding window for result consensus (5 frames)
        self.result_window = []  # Stores last N results: 'PERFECT', 'DEFECTIVE', or None
        self.window_size = 5
        self.geometry_tolerance = 0.05  # ±5% distance tolerance
        
        # Config
        self.config = {
            'expected_saddles': 4,
            'similarity_threshold': 0.88,
            'use_yolo': self.yolo_detector.model_loaded,
            'use_geometric_fallback': True,
            'use_alignment_correction': True,
            'infer_missing_saddles': True,
            'yolo_confidence': 0.25,
            'use_geometry_validation': True,  # Enable geometric constraint validation
            'use_sliding_window': True,  # Enable 5-frame consensus
        }
        
        print("[OK] Advanced Inspector initialized (ONNX optimized)")
        print(f"  YOLO-OBB: {'Enabled' if self.config['use_yolo'] else 'Disabled'}")
        print(f"  Frame skip threshold: {self.max_skip_frames} frames")
        print(f"  Sliding window: {self.window_size} frames")
    
    def add_reference_image(self, image: np.ndarray) -> bool:
        """Add a golden template image"""
        
        try:
            # Detect saddles
            if self.config['use_yolo']:
                saddles = self.yolo_detector.detect_saddles(
                    image, conf_threshold=self.config['yolo_confidence']
                )
            else:
                saddles = []
            
            if len(saddles) < self.config['expected_saddles'] and self.config['use_geometric_fallback']:
                print("  Falling back to geometric detection...")
                saddles = self.geometric_detector.detect_saddles(image)
            
            if len(saddles) != self.config['expected_saddles']:
                print(f"✗ Reference failed: {len(saddles)}/{self.config['expected_saddles']} saddles")
                return False
            
            # Add to reference manager
            success = self.reference_manager.add_reference(image, saddles)
            
            # Update geometric validator (only once)
            if success and self.reference_manager.get_reference_count() == 1:
                self.geo_validator.set_reference_positions(saddles)
            
            return success
            
        except Exception as e:
            print(f"✗ Reference error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def inspect_block(self, image: np.ndarray) -> BlockInspectionResult:
        """
        Perform complete inspection with all enhancements
        """
        
        start_time = time.time()
        
        # Step 1: Detect saddles
        if self.config['use_yolo']:
            saddles = self.yolo_detector.detect_saddles(
                image, conf_threshold=self.config['yolo_confidence']
            )
        else:
            saddles = []
        
        if len(saddles) < self.config['expected_saddles'] and self.config['use_geometric_fallback']:
            print("  Falling back to geometric detection...")
            saddles = self.geometric_detector.detect_saddles(image)
        
        detected_count = len(saddles)
        
        # Step 2: Infer missing saddles if enabled
        if len(saddles) < self.config['expected_saddles'] and self.config['infer_missing_saddles']:
            print(f"  Inferring {self.config['expected_saddles'] - len(saddles)} missing saddles...")
            saddles = self.geo_validator.infer_missing_saddles(saddles, image)
        
        # Step 3: Validate count
        if len(saddles) != self.config['expected_saddles']:
            alignment_status = f"ERROR: {len(saddles)}/{self.config['expected_saddles']} saddles"
            return self._error_result(alignment_status, 0.0, saddles, detected_count)
        
        alignment_status = "OK"
        
        # Step 4: Calculate average block angle
        block_angle = np.mean([s.angle for s in saddles])
        
        # Step 5: Feature comparison with alignment correction
        saddle_results = []
        
        for saddle in saddles:
            # Match against all references
            similarity, distance = self.reference_manager.match_saddle(saddle)
            
            # Determine status
            status = 'PERFECT' if similarity >= self.config['similarity_threshold'] else 'DEFECTIVE'
            
            # Check if detected or inferred
            detected = saddle.confidence > 0.0
            
            saddle_results.append(SaddleInspectionResult(
                saddle_id=saddle.id,
                status=status,
                similarity_score=similarity,
                confidence=similarity,
                feature_distance=distance,
                detected=detected
            ))
        
        # Step 6: Block decision
        defective_count = sum(1 for r in saddle_results if r.status == 'DEFECTIVE')
        block_status = 'DEFECTIVE' if defective_count > 0 else 'PERFECT'
        
        processing_time = (time.time() - start_time) * 1000
        
        # Store for visualization
        self.last_saddles = saddles
        self.last_image = image
        
        return BlockInspectionResult(
            block_status=block_status,
            total_saddles=self.config['expected_saddles'],
            detected_saddles=detected_count,
            defective_saddles=defective_count,
            saddle_results=saddle_results,
            processing_time_ms=processing_time,
            alignment_status=alignment_status,
            block_angle=block_angle
        )
    
    def inspect_live_frame(self, image: np.ndarray) -> Optional[BlockInspectionResult]:
        """
        Inspect a live video frame with optimizations:
        1. Frame skipping if YOLO fails (up to 5 consecutive failures)
        2. Geometry validation (collinearity + ±5% distance tolerance)
        3. Sliding window consensus (5/5 agreement required for final result)
        
        Returns:
            BlockInspectionResult or None (if frame skipped or awaiting consensus)
        """
        start_time = time.time()
        
        # Step 1: Try YOLO detection
        saddles = []
        if self.config['use_yolo']:
            saddles = self.yolo_detector.detect_saddles(
                image, conf_threshold=self.config['yolo_confidence']
            )
        
        # Step 2: Frame skipping logic
        if len(saddles) < self.config['expected_saddles']:
            self.consecutive_failures += 1
            
            if self.consecutive_failures < self.max_skip_frames:
                return None  # Skip frame
            else:
                print(f"  [FALLBACK] {self.consecutive_failures} failures - geometric detection")
                saddles = self.geometric_detector.detect_saddles(image)
                self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
        
        detected_count = len(saddles)
        
        # Step 3: Infer missing saddles
        if len(saddles) < self.config['expected_saddles'] and self.config['infer_missing_saddles']:
            saddles = self.geo_validator.infer_missing_saddles(saddles, image)
        
        # Step 4: Validate count
        if len(saddles) != self.config['expected_saddles']:
            self._add_to_window(None)  # Invalid frame
            return self._error_result(f"ERROR: {len(saddles)}/{self.config['expected_saddles']} saddles", 0.0, saddles, detected_count)
        
        # Step 5: GEOMETRY VALIDATION - reject frames with invalid saddle geometry
        if self.config.get('use_geometry_validation', True):
            is_valid, reason = self.geo_validator.validate_geometry(saddles, self.geometry_tolerance)
            if not is_valid:
                self._add_to_window(None)  # Geometry invalid
                return self._error_result(f"GEOMETRY: {reason}", 0.0, saddles, detected_count)
        
        alignment_status = "OK"
        block_angle = np.mean([s.angle for s in saddles])
        
        # Step 6: BATCH feature extraction
        aligned_crops = [AlignmentCorrector.correct_saddle(s) for s in saddles]
        batch_features = self.cnn_extractor.extract_batch(aligned_crops)
        
        # Step 7: Compare each saddle against references
        saddle_results = []
        for i, saddle in enumerate(saddles):
            test_features = batch_features[i]
            best_sim = 0.0
            avg_dist = 999.0
            
            for ref in self.reference_manager.references:
                if saddle.id < len(ref['features']):
                    ref_features = ref['features'][saddle.id]
                    sim = self.cnn_extractor.compute_similarity(test_features, ref_features)
                    dist = np.linalg.norm(test_features - ref_features)
                    if sim > best_sim:
                        best_sim = sim
                        avg_dist = dist
            
            status = 'PERFECT' if best_sim >= self.config['similarity_threshold'] else 'DEFECTIVE'
            saddle_results.append(SaddleInspectionResult(
                saddle_id=saddle.id,
                status=status,
                similarity_score=best_sim,
                confidence=best_sim,
                feature_distance=avg_dist,
                detected=saddle.confidence > 0.0
            ))
        
        # Step 8: Block decision (single frame)
        defective_count = sum(1 for r in saddle_results if r.status == 'DEFECTIVE')
        frame_status = 'DEFECTIVE' if defective_count > 0 else 'PERFECT'
        
        # Step 9: SLIDING WINDOW CONSENSUS
        self._add_to_window(frame_status)
        
        if self.config.get('use_sliding_window', True):
            consensus = self._get_consensus()
            if consensus is None:
                # Not enough consistent frames - return result but mark as "PENDING"
                block_status = 'PENDING'
            else:
                block_status = consensus
        else:
            block_status = frame_status
        
        processing_time = (time.time() - start_time) * 1000
        
        self.last_saddles = saddles
        self.last_image = image
        
        return BlockInspectionResult(
            block_status=block_status,
            total_saddles=self.config['expected_saddles'],
            detected_saddles=detected_count,
            defective_saddles=defective_count,
            saddle_results=saddle_results,
            processing_time_ms=processing_time,
            alignment_status=alignment_status,
            block_angle=block_angle
        )
    
    def _add_to_window(self, status: Optional[str]):
        """Add result to sliding window"""
        self.result_window.append(status)
        if len(self.result_window) > self.window_size:
            self.result_window.pop(0)
    
    def _get_consensus(self) -> Optional[str]:
        """
        Get consensus from sliding window.
        
        Rules:
        - 5/5 PERFECT -> PERFECT
        - 5/5 DEFECTIVE -> DEFECTIVE
        - Mixed or contains None -> None (no consensus)
        """
        if len(self.result_window) < self.window_size:
            return None  # Not enough data
        
        # Filter out None values
        valid_results = [r for r in self.result_window if r is not None]
        
        if len(valid_results) < self.window_size:
            return None  # Contains invalid frames
        
        # Check for unanimous decision
        if all(r == 'PERFECT' for r in valid_results):
            return 'PERFECT'
        elif all(r == 'DEFECTIVE' for r in valid_results):
            return 'DEFECTIVE'
        else:
            return None  # Mixed results - no consensus
    
    def reset_live_state(self):
        """Reset live video state (call when switching to new block)"""
        self.consecutive_failures = 0
        self.result_window = []
    
    def _error_result(self, message: str, angle: float, saddles: List, detected_count: int) -> BlockInspectionResult:
        """Create error result"""
        
        self.last_saddles = saddles
        
        return BlockInspectionResult(
            block_status='ERROR',
            total_saddles=self.config['expected_saddles'],
            detected_saddles=detected_count,
            defective_saddles=0,
            saddle_results=[],
            processing_time_ms=0.0,
            alignment_status=message,
            block_angle=angle
        )
    
    def visualize_results(
        self,
        original_image: np.ndarray,
        saddles: List[SaddleROI],
        results: List[SaddleInspectionResult]
    ) -> np.ndarray:
        """Create detailed visualization"""
        
        vis = original_image.copy()
        
        for saddle, result in zip(saddles, results):
            # Color based on status and detection
            if not result.detected:
                color = (255, 165, 0)  # Orange for inferred
            elif result.status == 'PERFECT':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            x, y, w, h = saddle.bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
            
            # Draw center with rotation indicator
            cx, cy = saddle.center
            cv2.circle(vis, (cx, cy), 10, color, -1)
            
            # Rotation indicator
            if abs(saddle.angle) > 1.0:
                rad = np.deg2rad(saddle.angle)
                end_x = int(cx + 30 * np.cos(rad))
                end_y = int(cy + 30 * np.sin(rad))
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), color, 3)
            
            # Label
            detection_marker = "✓" if result.detected else "⚙"
            label = f"{detection_marker} S{saddle.id}: {result.status}"
            sim_text = f"{result.similarity_score:.3f}"
            angle_text = f"{saddle.angle:.1f}°"
            
            # Background
            cv2.rectangle(vis, (cx-70, cy-70), (cx+70, cy+40), color, -1)
            
            # Text
            cv2.putText(vis, label, (cx-65, cy-45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, sim_text, (cx-65, cy-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, angle_text, (cx-65, cy+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, f"C:{saddle.confidence:.2f}", (cx-65, cy+28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis
