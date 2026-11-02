import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import cv2
import numpy as np
from deepface import DeepFace

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_SOURCE = 0  # Laptop's default webcam; try 1 if another camera is selected.
DATASET_PATH = "dataset"
UNKNOWN_PATH = os.path.join(DATASET_PATH, "Unknown")
CACHE_PATH = "deepface_facenet512_cache.pkl"

MODEL_NAME = "Facenet512"
DATASET_DETECTOR = "opencv"
LIVE_DETECTOR = "retinaface"
NORMALIZATION = "base"

# Raise this if wrong names appear. Lower it if known employees stay Unknown.
SIMILARITY_THRESHOLD = 0.45
FRAME_SKIP = 3
DISPLAY_SIZE = (960, 540)
SAVE_INTERVAL = 2

os.makedirs(UNKNOWN_PATH, exist_ok=True)


def employee_name_from_path(image_path):
    return os.path.basename(os.path.dirname(image_path))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def build_embedding_cache():
    embeddings = []
    names = []

    print("[INFO] Building DeepFace employee cache...")
    for person_name in sorted(os.listdir(DATASET_PATH)):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder) or person_name.lower() == "unknown":
            continue

        print(f"[INFO] Processing {person_name}")
        for image_name in os.listdir(person_folder):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            image_path = os.path.join(person_folder, image_name)
            try:
                reps = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    detector_backend=DATASET_DETECTOR,
                    enforce_detection=False,
                    align=True,
                    normalization=NORMALIZATION,
                )
            except Exception as exc:
                print(f"[WARN] Skipped {image_path}: {exc}")
                continue

            if not reps:
                print(f"[WARN] No face embedding: {image_path}")
                continue

            embeddings.append(reps[0]["embedding"])
            names.append(person_name)

    if not embeddings:
        raise RuntimeError("No employee embeddings were generated from dataset.")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names}, f)

    print(f"[DONE] Saved {len(embeddings)} embeddings for {len(set(names))} employees.")
    return embeddings, names


def load_embedding_cache():
    if not os.path.exists(CACHE_PATH):
        return build_embedding_cache()

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    names = data["names"]

    if len(embeddings) == 0:
        return build_embedding_cache()

    print(f"[INFO] Loaded {len(embeddings)} embeddings for {len(set(names))} employees.")
    return embeddings, names


def recognize_embedding(embedding, known_embeddings, known_names):
    scores = [cosine_similarity(known, embedding) for known in known_embeddings]
    best_index = int(np.argmax(scores))
    best_score = scores[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return known_names[best_index], best_score

    return "Unknown", best_score


def save_unknown_face(frame, area, unknown_count):
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    crop = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    if crop.size == 0:
        return unknown_count

    filename = os.path.join(UNKNOWN_PATH, f"unknown_{unknown_count}.jpg")
    cv2.imwrite(filename, crop)
    return unknown_count + 1


def main():
    known_embeddings, known_names = load_embedding_cache()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {VIDEO_SOURCE}")

    frame_count = 0
    unknown_count = 0
    last_save_time = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, DISPLAY_SIZE)

        if frame_count % FRAME_SKIP == 0:
            last_results = []

            try:
                reps = DeepFace.represent(
                    img_path=frame,
                    model_name=MODEL_NAME,
                    detector_backend=LIVE_DETECTOR,
                    enforce_detection=False,
                    align=True,
                    normalization=NORMALIZATION,
                )
            except Exception as exc:
                print(f"[WARN] Live frame skipped: {exc}")
                reps = []

            for rep in reps:
                area = rep.get("facial_area", {})
                if area.get("w", 0) < 8 or area.get("h", 0) < 8:
                    continue

                name, score = recognize_embedding(rep["embedding"], known_embeddings, known_names)
                last_results.append((area, name, score))

                if name == "Unknown":
                    current_time = time.time()
                    if current_time - last_save_time > SAVE_INTERVAL:
                        unknown_count = save_unknown_face(frame, area, unknown_count)
                        last_save_time = current_time

        for area, name, score in last_results:
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} {score:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Employee Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
