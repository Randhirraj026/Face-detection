import os
import cv2
import numpy as np
import pickle
import face_recognition

DATASET_PATH = "dataset"
OUTPUT_FILE = "encodings.pkl"

MODEL_PROTOTXT = "models/deploy.prototxt"
MODEL_WEIGHTS = "models/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)


def detect_face_dnn(image):
    """Detect face even if small/blurry using OpenCV DNN"""
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    if detections.shape[2] == 0:
        return None

    # pick best confidence face
    best_box = None
    best_conf = 0

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > 0.5 and conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            best_conf = conf

    if best_box is None:
        return None

    (x1, y1, x2, y2) = best_box

    # expand box slightly (IMPORTANT for CCTV blur)
    x1 = max(0, x1 - 20)
    y1 = max(0, y1 - 20)
    x2 = min(w, x2 + 20)
    y2 = min(h, y2 + 20)

    return image[y1:y2, x1:x2]


def load_dataset():
    encodings = []
    labels = []

    print("[INFO] Starting CCTV-optimized retrain...")

    for emp_id in os.listdir(DATASET_PATH):
        emp_path = os.path.join(DATASET_PATH, emp_id)

        if not os.path.isdir(emp_path):
            continue

        print(f"[INFO] Processing: {emp_id}")

        for img_name in os.listdir(emp_path):
            img_path = os.path.join(emp_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # 🔥 STEP 1: detect face using DNN (better than dlib)
            face_img = detect_face_dnn(image)

            if face_img is None or face_img.size == 0:
                print(f"[WARN] No face detected (DNN fallback failed): {img_path}")
                continue

            # 🔥 STEP 2: convert for encoding
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 🔥 STEP 3: encoding
            encs = face_recognition.face_encodings(rgb)

            if len(encs) == 0:
                print(f"[WARN] Encoding failed: {img_path}")
                continue

            encodings.append(encs[0])
            labels.append(emp_id)

    return encodings, labels


def save(encodings, labels):
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "labels": labels}, f)

    print(f"[DONE] Saved {len(encodings)} faces")


if __name__ == "__main__":
    enc, lbl = load_dataset()
    save(enc, lbl)