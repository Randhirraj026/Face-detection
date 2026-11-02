"""
Improved CCTV face recognition system
- Uses MTCNN for detection
- Uses FaceNet for embeddings
- Real-time recognition from video or RTSP stream
- Saves unknown faces for review
- Includes frame processing optimization (skip frames)
- Confidence-based thresholding
"""

import cv2
import os
import pickle
import numpy as np
import time
import logging
from datetime import datetime
from detector import MTCNNDetector
from embeddings import FaceNetEmbedder
from recognition import FaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CCTVFaceRecognitionSystem:
    """
    Real-time CCTV face recognition system
    """

    def __init__(self, video_source, embeddings_cache="embeddings_cache.pkl",
                 unknown_save_path="dataset/Unknown", 
                 recognition_threshold=0.6,
                 frame_skip=2,
                 save_unknown_interval=3.0):
        """
        Args:
            video_source: Video file path, RTSP URL, or camera index (0 for webcam)
            embeddings_cache: Path to saved embeddings pickle file
            unknown_save_path: Where to save unknown faces
            recognition_threshold: Cosine similarity threshold (0.5-0.7)
            frame_skip: Process every Nth frame (2 = skip 1, process 1)
            save_unknown_interval: Minimum seconds between saving unknown faces
        """
        self.video_source = video_source
        self.unknown_save_path = unknown_save_path
        self.recognition_threshold = recognition_threshold
        self.frame_skip = frame_skip
        self.save_unknown_interval = save_unknown_interval
        
        # Create unknown folder
        os.makedirs(unknown_save_path, exist_ok=True)
        
        # Initialize components
        self.detector = MTCNNDetector(min_face_size=20)
        self.embedder = FaceNetEmbedder()
        
        # Load embeddings
        self.recognizer = self._load_embeddings(embeddings_cache)
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "unknown_faces": 0,
            "start_time": time.time()
        }
        
        self.unknown_last_save_time = {}

    def _load_embeddings(self, cache_file):
        """
        Load pre-computed embeddings from cache
        
        Args:
            cache_file: Path to embeddings pickle file
            
        Returns:
            FaceRecognizer instance
        """
        if not os.path.exists(cache_file):
            logger.error(f"Embeddings cache not found: {cache_file}")
            logger.info("Run process_dataset.py first to generate embeddings")
            raise FileNotFoundError(f"Missing embeddings cache: {cache_file}")

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            embeddings = data["embeddings"]
            names = data["names"]
            
            logger.info(f"Loaded {len(embeddings)} embeddings from cache")
            
            recognizer = FaceRecognizer(embeddings, names, threshold=self.recognition_threshold)
            
            stats = recognizer.get_stats()
            logger.info(f"  - Unique persons: {stats['num_unique_persons']}")
            logger.info(f"  - Recognition threshold: {stats['threshold']}")
            
            return recognizer
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def process_frame(self, frame):
        """
        Process a single frame for face recognition
        
        Args:
            frame: BGR image frame
            
        Returns:
            Frame with bounding boxes and names drawn
        """
        if frame is None or frame.size == 0:
            return frame

        self.stats["frames_processed"] += 1

        try:
            # Preprocess for CCTV (improves detection)
            frame_processed = self.detector.preprocess_for_cctv(frame)

            # Detect faces
            detections = self.detector.detect_faces(frame_processed)

            if not detections:
                logger.debug("No faces detected in frame")
                return frame

            logger.debug(f"Detected {len(detections)} face(s)")
            self.stats["faces_detected"] += len(detections)

            # Process each detected face
            for detection in detections:
                try:
                    # Extract face ROI
                    face_roi = self.detector.extract_face_roi(frame, detection, margin=10)

                    if face_roi is None or face_roi.size == 0:
                        logger.warning("Failed to extract face ROI")
                        continue

                    # Generate embedding
                    embedding = self.embedder.generate_embedding(face_roi)

                    if embedding is None:
                        logger.warning("Failed to generate embedding")
                        continue

                    # Recognize face
                    person_name, confidence = self.recognizer.recognize_face(
                        embedding, 
                        return_distance=True
                    )

                    if person_name != "Unknown":
                        self.stats["faces_recognized"] += 1
                        logger.info(f"Recognized: {person_name} (confidence: {confidence:.4f})")
                    else:
                        self.stats["unknown_faces"] += 1
                        logger.info(f"Unknown face (best score: {confidence:.4f})")
                        
                        # Save unknown face
                        self._save_unknown_face(face_roi)

                    # Draw on frame
                    box = detection['box']
                    x, y, w, h = box
                    
                    # Color: green for known, red for unknown
                    color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw name and confidence
                    label = f"{person_name} ({confidence:.2f})"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(frame, 
                                (x, y - 30),
                                (x + text_size[0] + 5, y),
                                color, -1)
                    
                    # Text
                    cv2.putText(frame, label,
                               (x + 2, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 2)

                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

        return frame

    def _save_unknown_face(self, face_roi):
        """
        Save unknown face image for manual review
        
        Args:
            face_roi: Face image (BGR)
        """
        try:
            current_time = time.time()
            
            # Rate limit: don't save unknown faces too frequently
            if current_time - self.unknown_last_save_time.get("last_save", 0) < self.save_unknown_interval:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unknown_count = len([f for f in os.listdir(self.unknown_save_path) 
                               if f.startswith("unknown_")])
            
            filename = os.path.join(self.unknown_save_path, 
                                   f"unknown_{unknown_count}_{timestamp}.jpg")
            
            cv2.imwrite(filename, face_roi)
            
            logger.info(f"Saved unknown face: {filename}")
            self.unknown_last_save_time["last_save"] = current_time
            
        except Exception as e:
            logger.error(f"Failed to save unknown face: {e}")

    def run_live(self):
        """
        Run live CCTV recognition from video file or stream
        """
        logger.info(f"\nStarting CCTV recognition system")
        logger.info(f"Video source: {self.video_source}")
        logger.info(f"Recognition threshold: {self.recognition_threshold}")
        logger.info(f"Frame skip: {self.frame_skip} (process every {self.frame_skip}th frame)")
        logger.info("Press 'q' to quit, 's' to save current frame\n")

        # Open video source
        try:
            cap = cv2.VideoCapture(self.video_source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_source}")
                return False

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties - FPS: {fps}, Resolution: {width}x{height}")

        except Exception as e:
            logger.error(f"Failed to open video: {e}")
            return False

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("End of video reached")
                    break

                frame_count += 1

                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    cv2.imshow("CCTV Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Resize for faster processing
                frame_resized = cv2.resize(frame, (960, 540))

                # Process frame
                frame_processed = self.process_frame(frame_resized)

                # Add stats overlay
                self._draw_stats(frame_processed)

                # Display
                cv2.imshow("CCTV Face Recognition", frame_processed)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"frame_capture_{timestamp}.jpg"
                    cv2.imwrite(save_path, frame_processed)
                    logger.info(f"Frame saved: {save_path}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_stats()

        return True

    def _draw_stats(self, frame):
        """
        Draw system statistics on frame
        
        Args:
            frame: Frame to draw on (modified in place)
        """
        try:
            elapsed = time.time() - self.stats["start_time"]
            fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0

            stats_text = [
                f"Frames: {self.stats['frames_processed']}",
                f"FPS: {fps:.1f}",
                f"Faces detected: {self.stats['faces_detected']}",
                f"Recognized: {self.stats['faces_recognized']}",
                f"Unknown: {self.stats['unknown_faces']}"
            ]

            y_offset = 30
            for text in stats_text:
                cv2.putText(frame, text,
                           (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)
                y_offset += 25

        except Exception as e:
            logger.error(f"Stats drawing error: {e}")

    def _print_final_stats(self):
        """Print final statistics"""
        elapsed = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0

        logger.info(f"\n{'='*60}")
        logger.info("FINAL STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Total frames processed: {self.stats['frames_processed']}")
        logger.info(f"Average FPS: {fps:.1f}")
        logger.info(f"Total faces detected: {self.stats['faces_detected']}")
        logger.info(f"Faces recognized: {self.stats['faces_recognized']}")
        logger.info(f"Unknown faces: {self.stats['unknown_faces']}")
        logger.info(f"Unknown faces saved to: {self.unknown_save_path}")
        logger.info(f"{'='*60}\n")


def main():
    """
    Main entry point
    """
    # Configuration
    VIDEO_SOURCE = "video/test.mp4"  # Change to RTSP URL or 0 for webcam
    EMBEDDINGS_CACHE = "embeddings_cache.pkl"
    UNKNOWN_SAVE_PATH = "dataset/Unknown"
    RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold
    FRAME_SKIP = 2  # Process every 2nd frame (0.5x speed improvement)
    SAVE_UNKNOWN_INTERVAL = 3.0  # Seconds between saving unknown faces

    try:
        system = CCTVFaceRecognitionSystem(
            video_source=VIDEO_SOURCE,
            embeddings_cache=EMBEDDINGS_CACHE,
            unknown_save_path=UNKNOWN_SAVE_PATH,
            recognition_threshold=RECOGNITION_THRESHOLD,
            frame_skip=FRAME_SKIP,
            save_unknown_interval=SAVE_UNKNOWN_INTERVAL
        )

        system.run_live()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
