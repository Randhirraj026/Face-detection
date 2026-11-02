"""
Face recognition and matching logic
- Compares embeddings using cosine similarity
- Handles threshold-based matching
- Returns confidence scores
"""

import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition engine using embedding comparison
    """

    def __init__(self, known_embeddings, known_names, threshold=0.6):
        """
        Args:
            known_embeddings: List of reference embeddings (numpy arrays)
            known_names: List of corresponding person names
            threshold: Cosine similarity threshold for recognition (0.5-0.7 typical)
        """
        if len(known_embeddings) != len(known_names):
            raise ValueError("Embeddings and names must have same length")
        
        self.known_embeddings = np.array(known_embeddings)
        self.known_names = np.array(known_names)
        self.threshold = threshold
        
        logger.info(f"Recognizer initialized with {len(known_embeddings)} reference embeddings")

    def recognize_face(self, embedding, return_distance=False):
        """
        Recognize a face by comparing its embedding
        
        Args:
            embedding: Face embedding (128-D vector)
            return_distance: If True, return (name, confidence) else just name
            
        Returns:
            If return_distance=False: Person name or "Unknown"
            If return_distance=True: (name, confidence_score)
        """
        if embedding is None:
            return ("Unknown", 0.0) if return_distance else "Unknown"
        
        if len(self.known_embeddings) == 0:
            logger.warning("No reference embeddings loaded")
            return ("Unknown", 0.0) if return_distance else "Unknown"

        try:
            # Compute cosine similarity with all reference embeddings
            # Cosine similarity = 1 - cosine_distance
            distances = []
            for ref_embedding in self.known_embeddings:
                dist = cosine(embedding, ref_embedding)
                similarity = 1 - dist
                distances.append(similarity)
            
            distances = np.array(distances)
            
            # Find best match
            best_idx = np.argmax(distances)
            best_similarity = distances[best_idx]
            
            # Check if above threshold
            if best_similarity >= self.threshold:
                name = self.known_names[best_idx]
                confidence = float(best_similarity)
                logger.debug(f"Match: {name} (confidence: {confidence:.4f})")
            else:
                name = "Unknown"
                confidence = float(best_similarity)
                logger.debug(f"No match (best score: {confidence:.4f}, threshold: {self.threshold})")
            
            if return_distance:
                return (name, confidence)
            else:
                return name
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return ("Unknown", 0.0) if return_distance else "Unknown"

    def batch_recognize(self, embeddings, return_distances=False):
        """
        Recognize multiple faces at once
        
        Args:
            embeddings: List of embeddings
            return_distances: If True, return tuples (name, confidence)
            
        Returns:
            List of names or list of (name, confidence) tuples
        """
        results = []
        for embedding in embeddings:
            result = self.recognize_face(embedding, return_distance=return_distances)
            results.append(result)
        
        return results

    def set_threshold(self, threshold):
        """Update recognition threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")

    def get_stats(self):
        """Get recognizer statistics"""
        return {
            "num_known_embeddings": len(self.known_embeddings),
            "num_unique_persons": len(np.unique(self.known_names)),
            "threshold": self.threshold
        }


class FaceDistanceMeasurer:
    """
    Utility for measuring distances between embeddings
    """

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """
        Compute cosine similarity between two embeddings
        Returns value in [0, 1] where 1 = identical
        """
        try:
            dist = cosine(emb1, emb2)
            return 1 - dist
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return 0.0

    @staticmethod
    def cosine_distance(emb1, emb2):
        """
        Compute cosine distance (dissimilarity)
        Returns value in [0, 1] where 0 = identical
        """
        try:
            return cosine(emb1, emb2)
        except Exception as e:
            logger.error(f"Distance computation error: {e}")
            return 1.0

    @staticmethod
    def euclidean_distance(emb1, emb2):
        """Compute Euclidean distance between embeddings"""
        try:
            return np.sqrt(np.sum((emb1 - emb2) ** 2))
        except Exception as e:
            logger.error(f"Euclidean distance error: {e}")
            return float('inf')

    @staticmethod
    def l2_normalize(embedding):
        """L2 normalize an embedding"""
        return embedding / np.linalg.norm(embedding)
