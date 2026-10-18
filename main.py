
import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LeafMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model file is stored beside this Python file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "plant_disease_vgg16.keras"
)

CLASS_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
]

ADVICE = {
    "Tomato Bacterial Spot":
        "Possible bacterial spot. Avoid overhead watering and remove infected debris.",
    "Tomato Early Blight":
        "Possible early blight. Improve airflow and remove affected leaves carefully.",
    "Tomato Healthy":
        "The model classified this leaf as healthy. Continue monitoring the plant.",
}

model = None
load_error = None

try:
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )
    print("VGG16 model loaded successfully.")
except Exception as exc:
    load_error = str(exc)
    print("MODEL LOAD ERROR:", repr(exc))


@app.get("/")
def home():
    return {
        "app": "LeafMind AI",
        "model_loaded": model is not None,
        "message": "Tomato disease prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "error": load_error
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model could not load. Check server logs."
        )

    try:
        raw = await file.read()
        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

        image = image.resize((224, 224))

        # Keep pixels in the 0–255 range.
        # The saved model performs its own rescaling.
        array = np.asarray(
            image, dtype=np.float32
        )[None, ...]

        probabilities = model.predict(
            array, verbose=0
        )[0]

        if len(probabilities) != len(CLASS_NAMES):
            raise HTTPException(
                status_code=500,
                detail="Unexpected model output."
            )

        idx = int(np.argmax(probabilities))
        label = CLASS_NAMES[idx]

        return {
            "predicted_class": label,
            "confidence_percent": round(
                float(probabilities[idx]) * 100, 2
            ),
            "prevention_advice": ADVICE[label],
            "class_probabilities": {
                CLASS_NAMES[i]: round(
                    float(probabilities[i]) * 100, 2
                )
                for i in range(len(CLASS_NAMES))
            },
            "disclaimer":
                "AI screening only; not a confirmed diagnosis."
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
