import io
import os
import sys
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import json

# Ensure project modules are importable
CURRENT_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

# Import model definition and image size from training code
from models.train import CNN
from datasets.process_data import size

# Labels used during training
CLASS_LABELS: List[str] = json.load(open("models/labels.json"))

def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(REPO_ROOT, "models", "model.pth"))


app = FastAPI(title="Animals10 Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
device = torch.device("cpu")
transform = get_inference_transform()


@app.on_event("startup")
def load_model_on_startup() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    cnn = CNN(input_dim=size)
    cnn.load_state_dict(state_dict)
    cnn.eval()
    cnn.to(device)
    model = cnn


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Animals10 Classifier API is running"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/labels")
def get_labels() -> Dict[str, List[str]]:
    return {"labels": CLASS_LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the uploaded image.")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probabilities.argmax())
    top_label = CLASS_LABELS[top_idx]

    scores = {CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))}

    # Return top-3 for convenience
    top3_idx = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
    top3 = [
        {"label": CLASS_LABELS[i], "score": float(probabilities[i])}
        for i in top3_idx
    ]

    return {
        "predicted_label": top_label,
        "scores": scores,
        "top3": top3,
    }


# Allow running via: python model.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("code.deployment.api.model:app", host="0.0.0.0", port=8000, reload=False)


