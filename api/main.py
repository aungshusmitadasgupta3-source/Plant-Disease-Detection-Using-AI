from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from api.predict import predict_disease
from api.remedies import get_remedy
app = FastAPI(title="Plant Disease Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def normalize_label(label):
    label = label.replace("___", "_")
    label = label.replace("__", "_")
    return label
def split_label(label):
    parts = label.split("_")
    crop = parts[0]
    disease = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
    return crop, disease
@app.get("/health")
def health():
    return {"status": "OK 🚀"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        result = predict_disease(file.file)
        if result is None:
            return {"error": "Invalid image or preprocessing failed"}
        label, confidence = result
        remedy = get_remedy(label)
        return {
            "crop": label.split("___")[0],
            "disease": label.split("___")[-1],
            "confidence": round(confidence * 100, 2),
            "remedy": remedy
        }
    except Exception as e:
        return {"error": str(e)}