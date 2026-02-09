"""
Enhanced Saddle Structure Validator
====================================

Validates saddles based on physical structure:
1. Semicircular top surface
2. Vertical arc dividing the semicircle exactly in the middle
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SaddleStructure:
    """Detected saddle structure components"""
    has_semicircle: bool
    semicircle_center: Optional[Tuple[int, int]]
    semicircle_radius: Optional[int]
    has_middle_arc: bool
    arc_center_x: Optional[int]
    arc_alignment_score: float
    structure_confidence: float
    
    def is_valid_saddle(self, min_confidence: float = 0.7) -> bool:
        return (self.has_semicircle and 
                self.has_middle_arc and 
                self.structure_confidence >= min_confidence)


class SaddleStructureValidator:
    """
    Validates saddle structure using geometric constraints:
    1. Semicircular Surface Detection (Hough Circle + contour fallback)
    2. Middle Arc Detection (Sobel + Hough Lines + intensity profile)
    3. Alignment Validation (arc-semicircle center ±5% tolerance)
    """
    
    def __init__(self, 
                 arc_center_tolerance: float = 0.05,
                 min_arc_length_ratio: float = 0.6,
                 min_structure_confidence: float = 0.7):
        self.arc_center_tolerance = arc_center_tolerance
        self.min_arc_length_ratio = min_arc_length_ratio
        self.min_structure_confidence = min_structure_confidence
    
    def validate_structure(self, crop: np.ndarray) -> SaddleStructure:
        """Main validation function - returns SaddleStructure"""
        if crop is None or crop.size == 0:
            return self._invalid_structure()
        
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return self._invalid_structure()
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        
        # Step 1: Detect semicircular surface
        semi_detected, semi_center, semi_radius = self._detect_semicircular_surface(gray)
        
        # Step 2: Detect middle arc
        arc_detected, arc_center_x, arc_angle, arc_length = self._detect_middle_arc(gray)
        
        # Step 3: Validate alignment
        alignment_score = 0.0
        if semi_detected and arc_detected and semi_center is not None:
            expected_center_x = semi_center[0]
            center_deviation = abs(arc_center_x - expected_center_x)
            max_deviation = w * self.arc_center_tolerance
            
            alignment_score = max(0.0, 1.0 - (center_deviation / max_deviation)) if center_deviation <= max_deviation else 0.0
            
            if arc_length / h < self.min_arc_length_ratio:
                alignment_score *= 0.5
            if arc_angle is not None and abs(90 - abs(arc_angle)) > 15:
                alignment_score *= 0.5
        
        # Step 4: Calculate confidence
        if semi_detected and arc_detected:
            structure_confidence = alignment_score
        elif semi_detected:
            structure_confidence = 0.3
        elif arc_detected:
            structure_confidence = 0.2
        else:
            structure_confidence = 0.0
        
        return SaddleStructure(
            has_semicircle=semi_detected,
            semicircle_center=semi_center,
            semicircle_radius=semi_radius,
            has_middle_arc=arc_detected,
            arc_center_x=arc_center_x,
            arc_alignment_score=alignment_score,
            structure_confidence=structure_confidence
        )
    
    def _detect_semicircular_surface(self, gray: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], Optional[int]]:
        """Detect semicircular surface on top portion of saddle"""
        h, w = gray.shape
        upper_region = gray[:int(h * 0.6), :]
        
        blurred = cv2.GaussianBlur(upper_region, (5, 5), 1.5)
        edges = cv2.Canny(blurred, 30, 100)
        
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=w // 2,
            param1=50, param2=30, minRadius=int(w * 0.3), maxRadius=int(w * 0.8)
        )
        
        if circles is None or len(circles[0]) == 0:
            return self._detect_semicircle_from_contours(upper_region, w, h)
        
        circles = np.round(circles[0, :]).astype(int)
        cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        
        if abs(cx - w // 2) > w * 0.2 or cy > h * 0.5 or r < w * 0.3 or r > w * 0.8:
            return False, None, None
        
        return True, (cx, cy), r
    
    def _detect_semicircle_from_contours(self, upper_region: np.ndarray, full_w: int, full_h: int):
        """Fallback: Detect semicircle using contours"""
        contours, _ = cv2.findContours(cv2.Canny(upper_region, 30, 100), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None
        
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 5:
            return False, None, None
        
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, radius = int(cx), int(cy), int(radius)
        
        if abs(cx - full_w // 2) > full_w * 0.2:
            return False, None, None
        return True, (cx, cy), radius
    
    def _detect_middle_arc(self, gray: np.ndarray) -> Tuple[bool, Optional[int], Optional[float], float]:
        """Detect vertical arc dividing the saddle in the middle"""
        h, w = gray.shape
        
        # Method 1: Vertical edge detection
        arc_x, arc_length = self._detect_arc_from_edges(gray)
        if arc_x is not None:
            return True, arc_x, 90.0, arc_length
        
        # Method 2: Hough Line detection
        arc_detected, arc_x, angle, length = self._detect_arc_from_lines(gray)
        if arc_detected:
            return True, arc_x, angle, length
        
        # Method 3: Intensity profile
        arc_x = self._detect_arc_from_intensity(gray)
        if arc_x is not None:
            return True, arc_x, 90.0, h * 0.8
        
        return False, None, None, 0.0
    
    def _detect_arc_from_edges(self, gray: np.ndarray) -> Tuple[Optional[int], float]:
        """Detect arc using vertical edge detection (Sobel X)"""
        h, w = gray.shape
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        
        center_start, center_end = int(w * 0.3), int(w * 0.7)
        center_region = sobelx[:, center_start:center_end]
        vertical_sums = np.sum(center_region, axis=0)
        
        if len(vertical_sums) == 0:
            return None, 0.0
        
        peak_idx = np.argmax(vertical_sums)
        if vertical_sums[peak_idx] < np.mean(vertical_sums) + np.std(vertical_sums):
            return None, 0.0
        
        arc_x = center_start + peak_idx
        column = sobelx[:, arc_x]
        arc_length = float(np.sum(column > np.percentile(column, 70)))
        return arc_x, arc_length
    
    def _detect_arc_from_lines(self, gray: np.ndarray) -> Tuple[bool, Optional[int], Optional[float], float]:
        """Detect arc using Hough Line Transform"""
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=int(h * 0.3),
                                minLineLength=int(h * 0.4), maxLineGap=int(h * 0.2))
        
        if lines is None:
            return False, None, None, 0.0
        
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = 90.0 if x2 == x1 else abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if abs(angle - 90) < 15:
                line_center_x = (x1 + x2) / 2
                if abs(line_center_x - w / 2) < w * 0.3:
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    vertical_lines.append((line_center_x, angle, length))
        
        if not vertical_lines:
            return False, None, None, 0.0
        
        best = max(vertical_lines, key=lambda x: x[2])
        return True, int(best[0]), best[1], best[2]
    
    def _detect_arc_from_intensity(self, gray: np.ndarray) -> Optional[int]:
        """Detect arc using intensity profile (dark line in middle)"""
        h, w = gray.shape
        center_start, center_end = int(w * 0.3), int(w * 0.7)
        center_region = gray[:, center_start:center_end]
        column_means = np.mean(center_region, axis=0)
        
        if len(column_means) == 0:
            return None
        
        darkest_idx = np.argmin(column_means)
        darkest_value = column_means[darkest_idx]
        
        if 0 < darkest_idx < len(column_means) - 1:
            avg_neighbor = (column_means[darkest_idx - 1] + column_means[darkest_idx + 1]) / 2
            if darkest_value < avg_neighbor * 0.9:
                return center_start + darkest_idx
        return None
    
    def _invalid_structure(self) -> SaddleStructure:
        return SaddleStructure(False, None, None, False, None, 0.0, 0.0)
