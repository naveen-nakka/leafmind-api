
import os
import json
import subprocess

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# --------------------------------------------------
# LEAFMIND AI: 39-CLASS MODEL
# --------------------------------------------------

MODEL_PATH = "/tmp/leafmind_vgg16_39.keras"
CLASSES_PATH = "/tmp/class_names.json"

# Google Drive file IDs from your links.
# First link = model; second link = class_names.json.
MODEL_FILE_ID = "1lvTMXYpmVXtHKkyspuPVjXUwC1PTWL3v"
CLASSES_FILE_ID = "1uE1rR42oEvWoZaMHUtN5Gl8YeOZEgkvy"


def download_drive_file(file_id, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"File already downloaded: {destination}")
        return

    result = subprocess.run(
    [
        "python", "-m", "gdown",
        "--id", file_id,
        "-O", destination,
        "--fuzzy"
    ],
    capture_output=True,
    text=True,
    timeout=300
)

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not download Google Drive file {file_id}. "
            f"Check sharing permissions. Details: {result.stderr[-1500:]}"
        )

    if not os.path.exists(destination) or os.path.getsize(destination) == 0:
        raise RuntimeError(f"Downloaded file is missing or empty: {destination}")


print("Downloading LeafMind AI model files...")
download_drive_file(MODEL_FILE_ID, MODEL_PATH)
download_drive_file(CLASSES_FILE_ID, CLASSES_PATH)

print("Loading VGG16 model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if not isinstance(class_names, list) or len(class_names) != 39:
    raise RuntimeError(
        "class_names.json must contain a list of exactly 39 class names."
    )

if model.output_shape[-1] != len(class_names):
    raise RuntimeError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but class_names.json has {len(class_names)} classes."
    )

print("LeafMind AI model loaded with 39 classes.")

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="LeafMind AI API",
    description="Plant disease screening using a 39-class VGG16 model",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your website domain after testing
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_prevention_advice(disease):
    name = disease.lower()

    if "healthy" in name:
        return (
            "The model classified the leaf as healthy. Continue regular "
            "watering, monitor the plant, and maintain good garden hygiene."
        )

    if any(word in name for word in ["virus", "mosaic"]):
        return (
            "Inspect the plant and nearby plants. Some plant viruses are spread "
            "by insects or contaminated tools. Confirm the disease with an "
            "agricultural expert before choosing treatment."
        )

    if any(word in name for word in [
        "blight", "spot", "rot", "mold", "mildew", "rust"
    ]):
        return (
            "Avoid unnecessary leaf wetness, improve air circulation, keep "
            "tools clean, and safely remove severely affected plant material "
            "when appropriate. Confirm the disease before treatment."
        )

    return (
        "Inspect the plant and nearby leaves, maintain suitable watering and "
        "air circulation, and consult an agricultural expert to confirm the "
        "disease and choose appropriate treatment."
    )


@app.get("/")
def home():
    return {
        "app": "LeafMind AI",
        "status": "running",
        "model_classes": len(class_names)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(class_names)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        probabilities = model.predict(image_array, verbose=0)[0]

        top_indices = np.argsort(probabilities)[::-1][:3]
        best_index = int(top_indices[0])
        disease = class_names[best_index]
        confidence = float(probabilities[best_index]) * 100

        top3 = [
            {
                "class": class_names[int(i)],
                "confidence_percent": round(
                    float(probabilities[i]) * 100, 2
                )
            }
            for i in top_indices
        ]

        return {
            "predicted_class": disease,
            "confidence_percent": round(confidence, 2),
            "top3_predictions": top3,
            "prevention_advice": get_prevention_advice(disease),
            "disclaimer": "AI screening only — not a confirmed diagnosis."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the image: {str(e)}"
        )
