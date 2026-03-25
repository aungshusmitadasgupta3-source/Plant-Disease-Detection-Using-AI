import numpy as np
from PIL import Image
from api.model_loader import model, class_names

IMG_SIZE = 224

def preprocess_image(file):
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print("Image preprocessing error:", e)
        return None


def predict_disease(file):
    img = preprocess_image(file)

    if img is None:
        return None  # <-- THIS was missing before

    preds = model.predict(img)

    idx = np.argmax(preds)
    confidence = float(np.max(preds))

    label = class_names[idx]

    return label, confidence   # <-- MUST RETURN THIS