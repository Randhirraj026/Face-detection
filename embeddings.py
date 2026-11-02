"""
FaceNet embedding generation for CCTV face recognition
- Generates 128-D embeddings using FaceNet model
- Includes face alignment and normalization
- Handles errors gracefully
"""

import cv2
import numpy as np
from keras_facenet import FaceNet
import logging

logger = logging.getLogger(__name__)


class FaceNetEmbedder:
    """
    FaceNet embedding generator
    """

    def __init__(self):
        """Initialize FaceNet model"""
        try:
            self.model = FaceNet()
            self.required_size = (160, 160)  # FaceNet requires 160x160
            logger.info("FaceNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {e}")
            raise

    def align_face(self, face_image):
        """
        Align and normalize face image for FaceNet
        
        Args:
            face_image: BGR face image
            
        Returns:
            Aligned and resized face (160x160, normalized), or None if fails
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Invalid face image for alignment")
            return None

        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to FaceNet required size (160x160)
            face_aligned = cv2.resize(face_rgb, self.required_size, interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [-1, 1] (FaceNet standard)
            face_aligned = face_aligned.astype(np.float32)
            mean, std = face_aligned.mean(), face_aligned.std()
            face_aligned = (face_aligned - mean) / (std + 1e-6)
            
            return face_aligned
            
        except Exception as e:
            logger.error(f"Face alignment error: {e}")
            return None

    def generate_embedding(self, face_image):
        """
        Generate FaceNet embedding for a face
        
        Args:
            face_image: BGR face image
            
        Returns:
            128-D embedding vector (numpy array), or None if fails
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Invalid face image for embedding")
            return None

        try:
            # Align and normalize
            aligned_face = self.align_face(face_image)
            
            if aligned_face is None:
                return None
            
            # Add batch dimension for model
            face_batch = np.expand_dims(aligned_face, axis=0)
            
            # Generate embedding
            embedding = self.model.embeddings(face_batch)
            
            # Return 1-D embedding
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def batch_generate_embeddings(self, face_images):
        """
        Generate embeddings for multiple faces at once (faster)
        
        Args:
            face_images: List of BGR face images
            
        Returns:
            List of embeddings (same length as input, None for failed images)
        """
        embeddings = []
        
        for face_img in face_images:
            embedding = self.generate_embedding(face_img)
            embeddings.append(embedding)
        
        return embeddings

    def get_embedding_size(self):
        """Return embedding vector size"""
        return 128

    def extract_and_embed(self, image, face_detection):
        """
        End-to-end: extract face ROI from image and generate embedding
        
        Args:
            image: BGR image
            face_detection: Detection dict with 'box' key
            
        Returns:
            (face_roi, embedding) tuple, or (None, None) if fails
        """
        if image is None or face_detection is None:
            return None, None

        try:
            box = face_detection['box']
            x, y, w, h = box
            
            # Extract face ROI (with small margin)
            margin = 5
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                logger.warning("Empty ROI for embedding")
                return None, None
            
            # Generate embedding
            embedding = self.generate_embedding(face_roi)
            
            if embedding is None:
                logger.warning("Embedding generation failed")
                return face_roi, None
            
            return face_roi, embedding
            
        except Exception as e:
            logger.error(f"Extract and embed error: {e}")
            return None, None
