import json
import io
import pathlib
import numpy as np
from PIL import Image
import keras

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
IMG_SIZE = (224, 224)   # must match the size used in training

_model = None
_class_names = None


def load_model():
    global _model, _class_names
    _model = keras.models.load_model(MODEL_DIR / "clothing_model.keras")
    with open(MODEL_DIR / "class_names.json") as f:
        _class_names = json.load(f)
    print(f"CNN loaded with {len(_class_names)} classes.")


def predict_topk(image_bytes: bytes, k: int = 3):
    if _model is None:
        raise RuntimeError("Model not loaded.")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
    p = _model.predict(arr, verbose=0)[0]
    idx = np.argsort(p)[::-1][:k]
    return [(_class_names[i], float(p[i])) for i in idx]


def predict(image_bytes: bytes) -> tuple[str, float]:
    return predict_topk(image_bytes, 1)[0]