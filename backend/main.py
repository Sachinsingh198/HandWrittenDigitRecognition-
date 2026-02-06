from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('model/cnn_model.h5')

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')

    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)

    return {
        "digit": int(predicted_digit),
        "confidence": float(confidence)
    }
