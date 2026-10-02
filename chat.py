from fastapi import APIRouter
from pydantic import BaseModel
from app.services import rag, llm

router = APIRouter()

_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    predicted_class: str | None = None
    confidence: float | None = None


@router.post("/chat")
async def chat(req: ChatRequest):
    session = _sessions.setdefault(req.session_id, {"history": []})

    if req.predicted_class:
        session["predicted_class"] = req.predicted_class
        session["confidence"] = req.confidence

    if "predicted_class" not in session:
        return {"reply": "Please upload a photo of the clothing item first."}

    context = rag.retrieve_context(session["predicted_class"], req.message)

    reply = llm.generate_reply(
        predicted_class=session["predicted_class"],
        confidence=session["confidence"],
        context=context,
        user_message=req.message,
        history=session["history"],
    )

    session["history"].append({"role": "user", "content": req.message})
    session["history"].append({"role": "assistant", "content": reply})

    return {"reply": reply}
