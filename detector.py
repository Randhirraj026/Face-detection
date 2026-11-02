"""
MTCNN-based face detector for CCTV scenarios
- Better at detecting small, blurry, low-light faces
- Returns face coordinates and confidence scores
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)


class MTCNNDetector:
    """
    MTCNN face detector optimized for CCTV
    """

    def __init__(self, min_face_size=20):
        """
        Args:
            min_face_size: Minimum face size in pixels (lower = detects smaller faces)
        """
        self.detector = MTCNN()
        self.min_face_size = min_face_size

    def detect_faces(self, image):
        """
        Detect faces in image
        
        Args:
            image: BGR image (from cv2.imread)
            
        Returns:
            List of dicts with keys: 'box', 'confidence', 'keypoints'
            box format: [x, y, width, height]
            Returns [] if no faces detected
        """
        if image is None or image.size == 0:
            logger.warning("Invalid input image for detection")
            return []

        # Convert BGR to RGB for MTCNN
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            detections = self.detector.detect_faces(rgb)
            
            # Filter by minimum face size and confidence
            valid_detections = []
            for det in detections:
                box = det['box']
                x, y, w, h = box
                
                # Check if face is large enough
                if w >= self.min_face_size and h >= self.min_face_size:
                    # Check confidence threshold
                    if det['confidence'] >= 0.9:
                        valid_detections.append(det)
            
            return valid_detections
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []

    def extract_face_roi(self, image, detection, margin=10):
        """
        Extract face region with optional margin
        
        Args:
            image: BGR image
            detection: detection dict from detect_faces()
            margin: pixels to expand crop (handles near-edge faces)
            
        Returns:
            Cropped face image (BGR), or None if extraction fails
        """
        if image is None or detection is None:
            return None

        try:
            box = detection['box']
            x, y, w, h = box

            # Add margin and clip to image boundaries
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                logger.warning("Empty ROI after extraction")
                return None
                
            return face_roi
            
        except Exception as e:
            logger.error(f"Face ROI extraction error: {e}")
            return None

    def preprocess_for_cctv(self, image):
        """
        Preprocess image for better CCTV detection (low-light, blur)
        - CLAHE: Contrast Limited Adaptive Histogram Equalization
        
        Args:
            image: BGR image
            
        Returns:
            Preprocessed BGR image
        """
        if image is None:
            return image

        try:
            # Convert to grayscale for CLAHE
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (improves low-light performance)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Blend original and enhanced (avoids over-processing)
            result = cv2.addWeighted(image, 0.6, enhanced, 0.4, 0)
            
            return result
            
        except Exception as e:
            logger.warning(f"CCTV preprocessing error: {e}")
            return image

    def upscale_small_faces(self, image, detection, target_size=200):
        """
        Upscale frame if detected face is too small
        
        Args:
            image: BGR image
            detection: detection dict
            target_size: target face width (pixels)
            
        Returns:
            Upscaled image if face is small, original image otherwise
        """
        try:
            box = detection['box']
            face_width = box[2]
            
            if face_width < target_size:
                scale = target_size / face_width
                new_h = int(image.shape[0] * scale)
                new_w = int(image.shape[1] * scale)
                
                upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Upscaled face from {face_width} to ~{target_size}px")
                return upscaled, scale
                
            return image, 1.0
            
        except Exception as e:
            logger.warning(f"Upscaling error: {e}")
            return image, 1.0
