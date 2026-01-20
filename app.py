import os
import uuid
import cv2
import numpy as np
from collections import Counter, defaultdict

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from fer.fer import FER
from sklearn.cluster import KMeans


# App Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

app = Flask(__name__)
app.secret_key = "change-this-secret"



YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov8n.pt") 

person_detector = YOLO(YOLO_WEIGHTS if os.path.exists(YOLO_WEIGHTS) else "yolov8n.pt")

gender_classifier = pipeline("image-classification", model="rizvandwiki/gender-classification")
age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
# emotion_detector = FER(mtcnn=True)  



def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_persons(frame, model):
    results = model(frame, verbose=False)
    persons = []
    for box in results[0].boxes:
        if int(box.cls[0]) == 0: 
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append((x1, y1, x2, y2))
    return persons


def detect_gender(face_img):
    try:
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        gpred = gender_classifier(img)
        return f"{gpred[0]['label']} ({gpred[0]['score']:.2f})"
    except Exception:
        return "Unknown"


def detect_age(face_img):
    try:
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        apred = age_classifier(img)
        return f"{apred[0]['label']} ({apred[0]['score']:.2f})"
    except Exception:
        return "Unknown"


def dominant_colors(image, k=3):
    """Return multiple dominant BGR colors in image using KMeans."""
    if image is None or image.size == 0:
        return [(0, 0, 0)]
    img = cv2.resize(image, (50, 50))
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


def color_name_from_bgr(bgr):
    b, g, r = bgr
    if r > 150 and g < 100 and b < 100:
        return "Red"
    elif g > 150 and r < 100 and b < 100:
        return "Green"
    elif b > 150 and r < 100 and g < 100:
        return "Blue"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 60 and g < 60 and b < 60:
        return "Black"
    elif r > 150 and g > 150 and b < 80:
        return "Yellow"
    else:
        return "Other"


def get_dress_colors(crop, num_colors=2):
    if crop is None or crop.size == 0:
        return ["Unknown"]
    colors = dominant_colors(crop, k=num_colors)
    color_names = [color_name_from_bgr(c) for c in colors]
    return list(set(color_names))


def categorize_dress_type(y1, y2, x1, x2, torso_crop):
    h = y2 - y1
    w = x2 - x1
    aspect_ratio = h / (w + 1e-5)
    colors = get_dress_colors(torso_crop)
    if aspect_ratio > 2.5 and len(colors) > 1:
        return "Eastern"
    return "Western"


def categorize_specific_dress(y1, y2, x1, x2, torso_crop, full_crop):
    h = y2 - y1
    w = x2 - x1
    aspect_ratio = h / (w + 1e-5)
    colors = get_dress_colors(torso_crop, num_colors=2)

    # Eastern Types
    if aspect_ratio > 2.7 and "White" in colors:
        return "Thobe"
    if aspect_ratio > 2.6 and "Black" in colors:
        return "Abaya"
    if aspect_ratio > 2.4 and "Green" in colors:
        return "Shalwar Kameez"
    if aspect_ratio > 2.5 and len(colors) > 1:
        return "Kurta"
    if aspect_ratio > 2.8 and "Red" in colors:
        return "Saree"

    # Western Types
    if aspect_ratio < 2.0 and "Blue" in colors:
        return "Jeans + T-shirt"
    if aspect_ratio < 2.5 and "Black" in colors:
        return "Coat/Pants"
    if aspect_ratio < 2.0 and "White" in colors:
        return "Western T-shirt"

    return "Other"


def parse_label_prefix(s: str) -> str:
    # "Male (0.92)" -> "Male"
    if not s or s == "Unknown":
        return "Unknown"
    return s.split("(")[0].strip()


def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = 0
    if fps and frame_count:
        duration = frame_count / fps

    return {
        "fps": round(float(fps), 3) if fps else 0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
    }


def fast_dominant_color_names(bgr_img, num_colors=2):
    """
    Very fast dominant color estimator (no KMeans).
    """
    if bgr_img is None or bgr_img.size == 0:
        return ["Unknown"]

    small = cv2.resize(bgr_img, (32, 32), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)

    step = max(1, len(pixels) // 400)
    sample = pixels[::step]

    med = np.median(sample, axis=0).astype(int)
    colors = [tuple(med)]

    mean = np.mean(sample, axis=0).astype(int)
    colors.append(tuple(mean))

    names = [color_name_from_bgr(c) for c in colors][:num_colors]
    return list(dict.fromkeys(names)) 


def process_video(video_path: str, out_path: str,
                  analyze_every_n_frames: int = 3, 
                  attrs_every_n_frames: int = 15,      
                  yolo_imgsz: int = 640):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    stats = {
        "persons_total": 0,
        "gender": Counter(),
        "age": Counter(),
        "dress_type": Counter(),
        "specific_dress": Counter(),
        "colors": Counter(),
    }

 
    attr_cache = {}
    processed = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = frame

        
        if frame_idx % analyze_every_n_frames != 0:
            out.write(annotated)
            continue


        results = person_detector.predict(frame, imgsz=yolo_imgsz, verbose=False)
        persons = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                persons.append((x1, y1, x2, y2))

        stats["persons_total"] += len(persons)

        for (x1, y1, x2, y2) in persons:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(width - 1, x2); y2 = min(height - 1, y2)

            h = y2 - y1
            if h <= 0:
                continue

            face_y2 = y1 + int(h * 0.35)
            torso_y1 = y1 + int(h * 0.25)
            torso_y2 = y1 + int(h * 0.75)

            face_crop = frame[y1:face_y2, x1:x2]
            torso_crop = frame[torso_y1:torso_y2, x1:x2]

  
            key = (x1//10, y1//10, x2//10, y2//10)

            gender = "Unknown"
            age = "Unknown"

            do_attrs = (frame_idx % attrs_every_n_frames == 0)

            if not do_attrs and key in attr_cache:
                gender, age = attr_cache[key]
            else:
                if face_crop is not None and face_crop.size > 0 and face_crop.shape[0] >= 30 and face_crop.shape[1] >= 30:
                    gender = detect_gender(face_crop)
                    age = detect_age(face_crop)

                attr_cache[key] = (gender, age)

            colors = fast_dominant_color_names(torso_crop, num_colors=2)

            dress_type = categorize_dress_type(y1, y2, x1, x2, torso_crop)
            specific_dress = categorize_specific_dress(y1, y2, x1, x2, torso_crop, frame[y1:y2, x1:x2])

            stats["gender"][parse_label_prefix(gender)] += 1
            stats["age"][parse_label_prefix(age)] += 1
            stats["dress_type"][dress_type] += 1
            stats["specific_dress"][specific_dress] += 1
            for c in colors:
                stats["colors"][c] += 1

            # draw
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_texts = [
                f"Gender: {gender}",
                f"Age: {age}",
                f"Colors: {', '.join(colors)}",
                f"Dress Type: {dress_type}",
                f"Specific: {specific_dress}",
            ]
            for i, line in enumerate(label_texts):
                cv2.putText(
                    annotated,
                    line,
                    (x1, max(20, y1 - 10 - (i * 18))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        out.write(annotated)
        processed += 1

    cap.release()
    out.release()

    return {
        "persons_total": stats["persons_total"],
        "gender": dict(stats["gender"]),
        "age": dict(stats["age"]),
        "dress_type": dict(stats["dress_type"]),
        "specific_dress": dict(stats["specific_dress"]),
        "colors": dict(stats["colors"]),
        "frames_processed": processed,
    }



@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    if "video" not in request.files:
        flash("No file field found.")
        return redirect(url_for("index"))

    file = request.files["video"]
    if not file or file.filename == "":
        flash("Please choose a video file.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload mp4/mov/avi/mkv/webm.")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    vid_id = str(uuid.uuid4())
    in_name = f"{vid_id}.{ext}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    file.save(in_path)

    out_name = f"{vid_id}_output.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    meta = get_video_metadata(in_path)
    stats = process_video(in_path, out_path,
                      analyze_every_n_frames=3, 
                      attrs_every_n_frames=15,      
                      yolo_imgsz=640)


    return render_template(
        "result.html",
        input_filename=in_name,
        output_filename=out_name,
        meta=meta,
        stats=stats,
    )


@app.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
