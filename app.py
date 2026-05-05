# =========================================
# Tomato Disease Prediction API (FastAPI)
# =========================================

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import json

# ==============================
# Initialize FastAPI App
# ==============================

app = FastAPI(title="Tomato Disease Prediction API")

# ==============================
# Load Model (SavedModel)
# ==============================

MODEL_PATH = "disease_model"

model = tf.saved_model.load(MODEL_PATH)

# Get inference function
infer = model.signatures["serving_default"]

# ==============================
# Load Class Labels (LIST)
# ==============================

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Map index → label
labels = {i: label for i, label in enumerate(class_labels)}

# ==============================
# Image Preprocessing
# ==============================

IMG_SIZE = 224

def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file")

    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize (same as training)
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

# ==============================
# Routes
# ==============================

@app.get("/")
def home():
    return {"message": "Tomato Disease Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        # Preprocess image
        img = preprocess_image(contents)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(img)

        # Run inference
        outputs = infer(input_tensor)

        # Extract prediction array
        predictions = list(outputs.values())[0].numpy()

        # Get result
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        return {
            "disease": predicted_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# ==============================
# Run Server
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)