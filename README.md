# Face Recognition Model

This project recognizes faces from employee reference photos and a live camera
feed. It uses **DeepFace with the Facenet512 model** in the recommended webcam
program. It is face recognition, not general full-body human detection.

## Run the laptop webcam

The project currently has a Python 3.10 virtual environment named `cctv_env`.
In PowerShell:

```powershell
Set-Location 'D:\human detection model'
.\cctv_env\Scripts\Activate.ps1
python live_detect.py
```

Press **q** to quit. The default webcam is camera index `0`, configured by
`VIDEO_SOURCE` near the top of `live_detect.py`. If the laptop camera does not
open, try index `1` and check that another application is not using it.

To create a new environment instead:

```powershell
Set-Location 'D:\human detection model'
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install deepface opencv-python numpy
python live_detect.py
```

Run these commands from the project directory because the script uses paths
relative to it.

## Add employee photos

Create one folder per employee in `dataset`. The folder name is the identity
shown by the program:

```text
dataset/
├── KA001/
│   ├── photo1.jpg
│   └── photo2.jpg
└── KA002/
    └── photo1.jpg
```

Use clear photos with the employee's face visible. Supported image formats
are JPG, JPEG, PNG, and BMP. The program creates `dataset/Unknown/` and saves
crops of unmatched faces there.

On the first run, `live_detect.py` creates
`deepface_facenet512_cache.pkl` from the employee photos. Later runs reuse
that cache. If you change the reference photos, delete this cache file and run
the program again to rebuild it.

## How recognition works

1. Load cached employee face embeddings, or create them from `dataset/`.
2. Read frames from the webcam using OpenCV.
3. Analyze every third frame at 960 x 540 resolution with DeepFace's RetinaFace
   detector and Facenet512 embedding model.
4. Compare each detected face embedding to the employee embeddings using
   cosine similarity.
5. Draw recognized faces in green and unknown faces in red; save unknown face
   crops under `dataset/Unknown/`.

The similarity threshold is `0.45` (`SIMILARITY_THRESHOLD` in
`live_detect.py`). A higher threshold is stricter; a lower threshold accepts
more matches but may increase false matches. Check recognition results before
relying on them.

## Project files

| File | Purpose |
|---|---|
| `live_detect.py` | Recommended DeepFace webcam/video face recognition program |
| `live_detect_improved.py` | Alternative MTCNN/FaceNet live recognition program |
| `process_dataset.py` | Builds the alternative pipeline's embedding cache |
| `detector.py` | MTCNN face detection and image preprocessing |
| `embeddings.py` | FaceNet embedding generation |
| `recognition.py` | Matching logic for the MTCNN/FaceNet pipeline |
| `config.py` | Settings documented for the alternative pipeline; live scripts do not import it |
| `requirements_improved.txt` | Dependencies for the alternative MTCNN/FaceNet pipeline |

## Alternative pipeline

The MTCNN/FaceNet scripts are a separate implementation and use a different
cache. To run them:

```powershell
python -m pip install -r requirements_improved.txt
python process_dataset.py
python live_detect_improved.py
```

This creates `embeddings_cache.pkl`; it is not interchangeable with
`deepface_facenet512_cache.pkl`. In `live_detect_improved.py`, set
`VIDEO_SOURCE = 0` inside `main()` to use the default webcam.

## Git and privacy

`.gitignore` excludes the Python environments, employee dataset, recognition
caches, videos, and large model/data binaries. Employee face photos and
recognition outputs can contain sensitive personal data; do not upload them
to a public repository.

The folder must be initialized as a Git repository and connected to a remote
before you can push it. Once configured, review the files before staging:

```powershell
git status --short
git add .gitignore README.md *.py requirements_improved.txt
git status --short
```

Only commit and push after confirming the staged files contain no private
images, credentials, or other data you do not intend to publish.
