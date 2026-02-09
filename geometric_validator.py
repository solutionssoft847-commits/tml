
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

# Lazy import for RANSAC (avoids startup delay)
_ransac_lock = threading.Lock()
_RANSACRegressor = None

def _get_ransac():
    global _RANSACRegressor
    if _RANSACRegressor is None:
        with _ransac_lock:
            if _RANSACRegressor is None:
                from sklearn.linear_model import RANSACRegressor
                _RANSACRegressor = RANSACRegressor
    return _RANSACRegressor

warnings.filterwarnings('ignore')


@dataclass
class PhysicalDimensions:
    """Physical dimensions of saddles in millimeters"""
    height_min: float = 55.0
    height_max: float = 95.0
    width_min: float = 40.0
    width_max: float = 75.0
    spacing_min: float = 150.0
    spacing_max: float = 250.0
    spacing_avg: float = 200.0
    aspect_ratio_min: float = 1.3
    aspect_ratio_max: float = 1.5
    
    def validate_dimensions(self, height_px: float, width_px: float, 
                          spacing_px: float, px_to_mm: float) -> Tuple[bool, str]:
        height_mm = height_px * px_to_mm
        width_mm = width_px * px_to_mm
        spacing_mm = spacing_px * px_to_mm
        
        if not (self.height_min <= height_mm <= self.height_max):
            return False, f"Height {height_mm:.1f}mm out of range"
        if not (self.width_min <= width_mm <= self.width_max):
            return False, f"Width {width_mm:.1f}mm out of range"
        if spacing_px > 0 and not (self.spacing_min <= spacing_mm <= self.spacing_max):
            return False, f"Spacing {spacing_mm:.1f}mm out of range"
        
        aspect_ratio = height_px / width_px if width_px > 0 else 0
        if not (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max):
            return False, f"Aspect ratio {aspect_ratio:.2f} invalid"
        return True, "Dimensions valid"


class PixelToMMCalibrator:
    """Camera calibration for pixel-to-mm conversion"""
    
    def __init__(self, physical_dims: PhysicalDimensions):
        self.physical_dims = physical_dims
        self.px_to_mm_ratio: Optional[float] = None
        self.mm_to_px_ratio: Optional[float] = None
        self.calibrated = False
    
    def calibrate_from_saddles(self, saddles: List) -> Tuple[bool, str]:
        """Calibrate using spacing between two most confident saddles"""
        if len(saddles) < 2:
            return False, "Need at least 2 saddles"
        
        sorted_saddles = sorted(saddles, key=lambda s: s.confidence, reverse=True)
        s1, s2 = sorted_saddles[0], sorted_saddles[1]
        
        spacing_px = np.linalg.norm(np.array(s1.center) - np.array(s2.center))
        if spacing_px < 10:
            return False, f"Spacing too small: {spacing_px:.1f}px"
        
        spacing_mm = self.physical_dims.spacing_avg
        self.mm_to_px_ratio = spacing_px / spacing_mm
        self.px_to_mm_ratio = spacing_mm / spacing_px
        
        avg_height_mm = (self.physical_dims.height_min + self.physical_dims.height_max) / 2
        implied_height_px = avg_height_mm * self.mm_to_px_ratio
        
        if implied_height_px < 10 or implied_height_px > 500:
            return False, f"Scale anomaly: height {implied_height_px:.1f}px"
        
        self.calibrated = True
        return True, "Calibration successful"
    
    def px_to_mm(self, pixels: float) -> float:
        return pixels * self.px_to_mm_ratio if self.calibrated else 0.0
    
    def mm_to_px(self, millimeters: float) -> float:
        return millimeters * self.mm_to_px_ratio if self.calibrated else 0.0


class RANSACLineDetector:
    """Use RANSAC to robustly fit a line through saddle centers - FAST"""
    
    def __init__(self, residual_threshold: float = 10.0):
        self.residual_threshold = residual_threshold
        self.line_params: Optional[Tuple[float, float]] = None
        self.inliers: Optional[np.ndarray] = None
    
    def fit_line(self, points: np.ndarray) -> Tuple[bool, str]:
        if len(points) < 2:
            return False, "Need at least 2 points"
        
        try:
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            
            RANSACRegressor = _get_ransac()  # Lazy import
            ransac = RANSACRegressor(residual_threshold=self.residual_threshold,
                                     random_state=42, max_trials=50)  # Reduced trials for speed
            ransac.fit(X, y)
            
            self.line_params = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
            self.inliers = ransac.inlier_mask_
            
            outlier_count = len(points) - np.sum(self.inliers)
            if outlier_count > 0:
                return False, f"RANSAC: {outlier_count} outliers"
            return True, "All points are inliers"
        except Exception as e:
            return False, f"RANSAC failed: {e}"
    
    def get_distance_from_line(self, point: Tuple[float, float]) -> float:
        if self.line_params is None:
            return 999.0
        slope, intercept = self.line_params
        x, y = point
        return abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)


class SaddleMeshProjector:
    """Project a 1x4 grid (mesh) onto the image"""
    
    def __init__(self, physical_dims: PhysicalDimensions, calibrator: PixelToMMCalibrator):
        self.physical_dims = physical_dims
        self.calibrator = calibrator
        self.mesh_centers: Optional[List[Tuple[int, int]]] = None
        self.mesh_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    
    def create_mesh_from_detections(self, saddles: List) -> Tuple[bool, str]:
        if len(saddles) < 2 or not self.calibrator.calibrated:
            return False, "Need 2+ saddles and calibration"
        
        centers = np.array([s.center for s in saddles])
        spacing_px = self.calibrator.mm_to_px(self.physical_dims.spacing_avg)
        
        x_coords = centers[:, 0]
        start_point = centers[np.argmin(x_coords)]
        end_point = centers[np.argmax(x_coords)]
        
        direction = end_point - start_point
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1:
            return False, "Start and end too close"
        
        direction_unit = direction / direction_norm
        
        self.mesh_centers = []
        for i in range(4):
            point = start_point + direction_unit * (i * spacing_px)
            self.mesh_centers.append((int(point[0]), int(point[1])))
        
        height_mm = (self.physical_dims.height_min + self.physical_dims.height_max) / 2
        width_mm = (self.physical_dims.width_min + self.physical_dims.width_max) / 2
        height_px = int(self.calibrator.mm_to_px(height_mm))
        width_px = int(self.calibrator.mm_to_px(width_mm))
        
        self.mesh_bboxes = []
        for cx, cy in self.mesh_centers:
            self.mesh_bboxes.append((cx - width_px // 2, cy - height_px // 2, width_px, height_px))
        
        return True, "Mesh created"
    
    def force_crop_at_mesh(self, image: np.ndarray, mesh_id: int) -> Optional[np.ndarray]:
        if self.mesh_bboxes is None or mesh_id >= len(self.mesh_bboxes):
            return None
        
        x, y, w, h = self.mesh_bboxes[mesh_id]
        h_img, w_img = image.shape[:2]
        
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        x_end = max(x + 1, min(x + w, w_img))
        y_end = max(y + 1, min(y + h, h_img))
        
        return image[y:y_end, x:x_end].copy()


class OptimizedGeometricValidator:
    """
    Optimized Geometric Constraint Validator
    
    Validates:
    1. RANSAC collinearity
    2. Physical dimensions (55-95mm × 40-75mm)
    3. Semicircular surface + middle arc (via SaddleStructureValidator)
    4. Mesh-based inference
    5. Automatic calibration
    """
    
    def __init__(self, expected_count: int = 4):
        self.expected_count = expected_count
        self.physical_dims = PhysicalDimensions()
        self.calibrator = PixelToMMCalibrator(self.physical_dims)
        self.ransac = RANSACLineDetector(residual_threshold=15.0)
        self.mesh_projector = SaddleMeshProjector(self.physical_dims, self.calibrator)
        self.reference_positions = None
        self.reference_distances = {}
        
        # Optional structure validator
        self.structure_validator = None
        try:
            from saddle_structure_validator import SaddleStructureValidator
            self.structure_validator = SaddleStructureValidator(
                arc_center_tolerance=0.05,
                min_arc_length_ratio=0.6,
                min_structure_confidence=0.65
            )
        except ImportError:
            pass
    
    def set_reference_positions(self, saddles: List) -> bool:
        if len(saddles) != self.expected_count:
            return False
        
        success, _ = self.calibrator.calibrate_from_saddles(saddles)
        if not success:
            return False
        
        self.mesh_projector.create_mesh_from_detections(saddles)
        
        centers = [s.center for s in saddles]
        centroid = (int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers])))
        self.reference_positions = [(c[0] - centroid[0], c[1] - centroid[1]) for c in centers]
        
        self.reference_distances = {}
        for i in range(len(saddles)):
            for j in range(i+1, len(saddles)):
                dist = np.linalg.norm(np.array(saddles[i].center) - np.array(saddles[j].center))
                self.reference_distances[(i, j)] = {'px': dist, 'mm': self.calibrator.px_to_mm(dist)}
        
        return True
    
    def validate_and_refine(self, saddles: List, image: np.ndarray) -> Tuple[List, Dict]:
        """
        PRODUCTION-LEVEL validation with parallel processing for instant results.
        Structure validation, RANSAC, and mesh creation run concurrently.
        """
        report = {'initial_count': len(saddles), 'warnings': [], 'inferred_count': 0}
        
        # Early calibration (fast - ~1ms)
        if not self.calibrator.calibrated and len(saddles) >= 2:
            self.calibrator.calibrate_from_saddles(saddles)
        
        # PARALLEL: Structure validation + RANSAC + Mesh creation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Task 1: Parallel structure validation for each saddle
            if self.structure_validator and saddles:
                def validate_structure(saddle):
                    if hasattr(saddle, 'crop') and saddle.crop is not None and saddle.crop.size > 0:
                        return saddle, self.structure_validator.validate_structure(saddle.crop)
                    return saddle, None
                
                for saddle in saddles:
                    futures[executor.submit(validate_structure, saddle)] = 'structure'
            
            # Task 2: RANSAC line fitting (runs parallel)
            if len(saddles) >= 2:
                centers = np.array([s.center for s in saddles])
                futures[executor.submit(self.ransac.fit_line, centers)] = 'ransac'
            
            # Task 3: Mesh projection (runs parallel)
            if len(saddles) >= 2 and self.calibrator.calibrated:
                futures[executor.submit(self.mesh_projector.create_mesh_from_detections, saddles)] = 'mesh'
            
            # Collect results as they complete (non-blocking)
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    result = future.result()
                    if task_type == 'structure' and result[1] is not None:
                        saddle, structure = result
                        if hasattr(saddle, '__dict__'):
                            saddle.structure = structure
                except Exception:
                    pass  # Continue on failure - don't block
        
        # Refine bboxes for aspect ratio (fast - in-place)
        refined_saddles = self._refine_bboxes(saddles, image)
        
        # Infer missing saddles using mesh
        if len(refined_saddles) < self.expected_count and self.mesh_projector.mesh_bboxes:
            refined_saddles = self._infer_missing(refined_saddles, image, report)
        
        refined_saddles.sort(key=lambda s: s.id)
        report['final_count'] = len(refined_saddles)
        return refined_saddles, report
    
    def _refine_bboxes(self, saddles: List, image: np.ndarray) -> List:
        """Refine bounding boxes to match physical aspect ratio"""
        for saddle in saddles:
            x, y, w, h = saddle.bbox
            aspect_ratio = h / w if w > 0 else 0
            
            if not (self.physical_dims.aspect_ratio_min <= aspect_ratio <= self.physical_dims.aspect_ratio_max):
                target_aspect = (self.physical_dims.aspect_ratio_min + self.physical_dims.aspect_ratio_max) / 2
                cx, cy = saddle.center
                
                if self.calibrator.calibrated:
                    avg_height = (self.physical_dims.height_min + self.physical_dims.height_max) / 2
                    avg_width = (self.physical_dims.width_min + self.physical_dims.width_max) / 2
                    new_h = int(self.calibrator.mm_to_px(avg_height))
                    new_w = int(self.calibrator.mm_to_px(avg_width))
                else:
                    new_h, new_w = h, int(h / target_aspect)
                
                new_x = max(0, min(cx - new_w // 2, image.shape[1] - new_w))
                new_y = max(0, min(cy - new_h // 2, image.shape[0] - new_h))
                
                saddle.bbox = (new_x, new_y, new_w, new_h)
                saddle.crop = image[new_y:new_y+new_h, new_x:new_x+new_w].copy()
                saddle.area = new_w * new_h
        
        return saddles
    
    def _infer_missing(self, saddles: List, image: np.ndarray, report: Dict) -> List:
        """Infer missing saddles using mesh projection"""
        from inspector_engine import SaddleROI  # Import here to avoid circular
        
        detected_ids = {s.id for s in saddles}
        
        for expected_id in range(self.expected_count):
            if expected_id not in detected_ids:
                crop = self.mesh_projector.force_crop_at_mesh(image, expected_id)
                if crop is not None and crop.size > 0:
                    cx, cy = self.mesh_projector.mesh_centers[expected_id]
                    x, y, w, h = self.mesh_projector.mesh_bboxes[expected_id]
                    
                    inferred = SaddleROI(
                        id=expected_id, bbox=(x, y, w, h), crop=crop,
                        center=(cx, cy), area=w * h, angle=0.0, confidence=0.0
                    )
                    
                    if self.structure_validator:
                        inferred.structure = self.structure_validator.validate_structure(crop)
                    
                    saddles.append(inferred)
                    report['inferred_count'] += 1
        
        return saddles
    
    def validate_geometry(self, saddles: List, tolerance: float = 0.10) -> Tuple[bool, str]:
        """Validate geometry with RANSAC and physical constraints"""
        if len(saddles) != self.expected_count:
            return False, f"Expected {self.expected_count}, got {len(saddles)}"
        
        # RANSAC check
        centers = np.array([s.center for s in saddles])
        success, msg = self.ransac.fit_line(centers)
        if not success:
            return False, msg
        
        for i, saddle in enumerate(saddles):
            if self.ransac.inliers is not None and not self.ransac.inliers[i]:
                return False, f"S{saddle.id} is RANSAC outlier"
            if self.ransac.get_distance_from_line(saddle.center) > 20.0:
                return False, f"S{saddle.id} too far from line"
        
        # Physical dimensions check
        if self.calibrator.calibrated:
            for saddle in saddles:
                x, y, w, h = saddle.bbox
                height_mm = self.calibrator.px_to_mm(h)
                width_mm = self.calibrator.px_to_mm(w)
                
                if not (self.physical_dims.height_min <= height_mm <= self.physical_dims.height_max):
                    return False, f"S{saddle.id} height {height_mm:.1f}mm invalid"
                if not (self.physical_dims.width_min <= width_mm <= self.physical_dims.width_max):
                    return False, f"S{saddle.id} width {width_mm:.1f}mm invalid"
        
        # Spacing check
        sorted_saddles = sorted(saddles, key=lambda s: s.id)
        for i in range(len(sorted_saddles) - 1):
            s0, s1 = sorted_saddles[i], sorted_saddles[i + 1]
            dist_px = np.linalg.norm(np.array(s1.center) - np.array(s0.center))
            
            if self.calibrator.calibrated:
                dist_mm = self.calibrator.px_to_mm(dist_px)
                if not (self.physical_dims.spacing_min <= dist_mm <= self.physical_dims.spacing_max):
                    return False, f"Spacing S{s0.id}-S{s1.id}: {dist_mm:.1f}mm invalid"
        
        return True, "All geometry checks passed"
