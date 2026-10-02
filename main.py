import pathlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.services import cnn_model, rag
from app.routers import predict, chat, style


@asynccontextmanager
async def lifespan(app: FastAPI):
    cnn_model.load_model()
    rag.build_vector_store()
    yield


app = FastAPI(title="Fashion Advisor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(style.router)


@app.get("/health")
def health():
    return {"status": "running"}


# Serve the website from the same server (needed for camera access).
FRONTEND = pathlib.Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")