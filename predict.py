from fastapi import APIRouter, UploadFile, File
from app.services import cnn_model

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence = cnn_model.predict(image_bytes)
    return {"predicted_class": predicted_class, "confidence": confidence}
