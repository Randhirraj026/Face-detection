"""
Configuration file for CCTV Face Recognition System
Modify these settings to match your setup and requirements
"""

import os

# ============================================================================
# VIDEO SOURCE CONFIGURATION
# ============================================================================

# For local video file:
# VIDEO_SOURCE = "video/test.mp4"

# For RTSP camera stream (example):
# VIDEO_SOURCE = "rtsp://192.168.1.100:554/stream"

# For webcam:
VIDEO_SOURCE = 0

# ============================================================================
# DATASET PATHS
# ============================================================================

DATASET_PATH = "dataset"                           # Employee images folder
EMBEDDINGS_CACHE = "embeddings_cache.pkl"          # Cached embeddings file
UNKNOWN_SAVE_PATH = os.path.join(DATASET_PATH, "Unknown")  # Unknown faces folder

# ============================================================================
# RECOGNITION PARAMETERS
# ============================================================================

# Cosine similarity threshold for face recognition
# Range: 0.0 - 1.0
# Lower values = more lenient (accept more matches, fewer unknowns)
# Higher values = stricter (fewer matches, more unknowns)
# Typical range: 0.55 - 0.70
# Default: 0.6
RECOGNITION_THRESHOLD = 0.6

# Minimum face size for detection (pixels)
# Lower values detect smaller faces but slower
# Typical: 10-40 pixels
MIN_FACE_SIZE = 20

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

# Frame skipping: Process every Nth frame
# 1 = process all frames (slower)
# 2 = process every 2nd frame (2x faster)
# 3 = process every 3rd frame (3x faster)
# Default: 2
FRAME_SKIP = 2

# Video resolution for processing
# Smaller = faster but may miss small faces
# (960, 540) = good balance between speed and accuracy
VIDEO_RESIZE = (960, 540)

# Minimum interval between saving unknown faces (seconds)
# Prevents spam of identical unknown faces
SAVE_UNKNOWN_INTERVAL = 3.0

# ============================================================================
# LOGGING
# ============================================================================

# Logging level: DEBUG, INFO, WARNING, ERROR
# DEBUG = most verbose
# ERROR = least verbose
LOG_LEVEL = "INFO"

# ============================================================================
# OPTIONAL: GPU ACCELERATION
# ============================================================================

# Set to True if CUDA/GPU is available
USE_GPU = False

# ============================================================================
# OPTIONAL: FACE TRACKING
# ============================================================================

# Enable face tracking to reduce reprocessing same face
# (Feature for future implementation)
ENABLE_FACE_TRACKING = False
FACE_TRACKING_MAX_AGE = 10  # frames

# ============================================================================
# OPTIONAL: MULTI-STREAM PROCESSING
# ============================================================================

# Process multiple video sources
# (Feature for future implementation)
ENABLE_MULTI_STREAM = False
STREAM_URLS = [
    "rtsp://camera1:554/stream",
    "rtsp://camera2:554/stream"
]

# ============================================================================
# DEFAULTS FOR COMMAND LINE
# ============================================================================

# Use these if not specified as arguments
DEFAULT_CONFIDENCE_THRESHOLD = 0.9  # MTCNN detection confidence
DEFAULT_MARGIN_PIXELS = 10          # Pixels to expand face crop
