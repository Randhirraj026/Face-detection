"""
CCTV-optimized dataset processing
- Scans dataset folder (employee_name/image.jpg structure)
- Detects faces using MTCNN
- Generates FaceNet embeddings
- Caches embeddings to pickle for faster loading
- Provides detailed logging of skipped/failed images
"""

import os
import cv2
import pickle
import numpy as np
import logging
from detector import MTCNNDetector
from embeddings import FaceNetEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Process employee dataset and generate embeddings
    """

    def __init__(self, dataset_path, output_file="embeddings_cache.pkl"):
        """
        Args:
            dataset_path: Root dataset folder (contains employee_name folders)
            output_file: Where to save embeddings cache
        """
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.detector = MTCNNDetector(min_face_size=20)
        self.embedder = FaceNetEmbedder()
        
        self.stats = {
            "total_images": 0,
            "images_with_faces": 0,
            "embeddings_generated": 0,
            "skipped_images": 0,
            "failed_employees": 0,
            "successful_employees": 0
        }

    def process_employee_folder(self, emp_name, emp_folder):
        """
        Process all images for one employee
        
        Args:
            emp_name: Employee name/ID
            emp_folder: Path to employee folder
            
        Returns:
            (embeddings_list, names_list) for this employee
        """
        embeddings = []
        names = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing employee: {emp_name}")
        logger.info(f"{'='*60}")
        
        if not os.path.isdir(emp_folder):
            logger.warning(f"Not a directory: {emp_folder}")
            self.stats["failed_employees"] += 1
            return embeddings, names

        image_files = [f for f in os.listdir(emp_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            logger.warning(f"No images found in {emp_folder}")
            self.stats["failed_employees"] += 1
            return embeddings, names

        logger.info(f"Found {len(image_files)} images")

        for img_name in image_files:
            img_path = os.path.join(emp_folder, img_name)
            self.stats["total_images"] += 1

            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"[SKIP] Corrupted image: {img_name}")
                    self.stats["skipped_images"] += 1
                    continue

                # Preprocess for CCTV (improves low-light detection)
                image_preprocessed = self.detector.preprocess_for_cctv(image)

                # Detect faces
                detections = self.detector.detect_faces(image_preprocessed)

                if not detections:
                    logger.warning(f"[SKIP] No face detected: {img_name}")
                    self.stats["skipped_images"] += 1
                    continue

                if len(detections) > 1:
                    logger.warning(f"[WARN] Multiple faces ({len(detections)}) in {img_name}, using best confidence")

                # Use first/best detection
                detection = detections[0]
                
                # Extract face ROI
                face_roi = self.detector.extract_face_roi(image, detection, margin=10)

                if face_roi is None or face_roi.size == 0:
                    logger.warning(f"[SKIP] Failed to extract face ROI: {img_name}")
                    self.stats["skipped_images"] += 1
                    continue

                # Generate embedding
                embedding = self.embedder.generate_embedding(face_roi)

                if embedding is None:
                    logger.warning(f"[SKIP] Embedding generation failed: {img_name}")
                    self.stats["skipped_images"] += 1
                    continue

                # Success
                embeddings.append(embedding)
                names.append(emp_name)
                self.stats["embeddings_generated"] += 1
                
                confidence = detection['confidence']
                logger.info(f"[OK] {img_name} - embedding generated (confidence: {confidence:.4f})")

            except Exception as e:
                logger.error(f"[ERROR] Processing {img_name}: {str(e)}")
                self.stats["skipped_images"] += 1
                continue

        if embeddings:
            self.stats["successful_employees"] += 1
            logger.info(f"Generated {len(embeddings)} embeddings for {emp_name}")
        else:
            logger.error(f"FAILED to generate any embeddings for {emp_name}")
            self.stats["failed_employees"] += 1

        return embeddings, names

    def process_dataset(self):
        """
        Process entire dataset
        
        Returns:
            (all_embeddings, all_names)
        """
        all_embeddings = []
        all_names = []

        logger.info(f"\nStarting dataset processing: {self.dataset_path}\n")

        for emp_name in sorted(os.listdir(self.dataset_path)):
            emp_folder = os.path.join(self.dataset_path, emp_name)

            # Skip non-directories
            if not os.path.isdir(emp_folder):
                continue

            # Skip special folders
            if emp_name.lower() in ['unknown', 'temp', '.git', '__pycache__']:
                logger.debug(f"Skipping special folder: {emp_name}")
                continue

            # Process employee
            embeddings, names = self.process_employee_folder(emp_name, emp_folder)
            
            all_embeddings.extend(embeddings)
            all_names.extend(names)

        return all_embeddings, all_names

    def save_embeddings(self, embeddings, names):
        """
        Save embeddings to pickle cache
        
        Args:
            embeddings: List of embedding vectors
            names: List of corresponding names
        """
        if not embeddings:
            logger.error("No embeddings to save")
            return False

        try:
            data = {
                "embeddings": np.array(embeddings),
                "names": np.array(names)
            }
            
            with open(self.output_file, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"\n✓ Saved {len(embeddings)} embeddings to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False

    def print_stats(self):
        """Print processing statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total images scanned: {self.stats['total_images']}")
        logger.info(f"Images with detected faces: {self.stats['images_with_faces']}")
        logger.info(f"Embeddings generated: {self.stats['embeddings_generated']}")
        logger.info(f"Skipped/Failed images: {self.stats['skipped_images']}")
        logger.info(f"Successful employees: {self.stats['successful_employees']}")
        logger.info(f"Failed employees: {self.stats['failed_employees']}")
        logger.info(f"{'='*60}\n")


def main():
    """
    Main dataset processing script
    """
    DATASET_PATH = "dataset"
    OUTPUT_FILE = "embeddings_cache.pkl"

    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path not found: {DATASET_PATH}")
        return

    processor = DatasetProcessor(DATASET_PATH, OUTPUT_FILE)
    
    # Process dataset
    embeddings, names = processor.process_dataset()
    
    # Save to cache
    if embeddings:
        processor.save_embeddings(embeddings, names)
    
    # Print stats
    processor.print_stats()


if __name__ == "__main__":
    main()
