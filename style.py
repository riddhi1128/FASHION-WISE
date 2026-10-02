import io
import urllib.parse
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from app.services import cnn_model, rag, llm

router = APIRouter()

PALETTE = {
    "black": (20, 20, 20), "white": (245, 245, 245), "grey": (128, 128, 128),
    "navy": (25, 35, 80), "blue": (40, 90, 200), "light blue": (150, 190, 235),
    "red": (200, 30, 40), "maroon": (110, 20, 35), "pink": (240, 150, 180),
    "orange": (240, 130, 30), "yellow": (240, 210, 50), "beige": (215, 195, 160),
    "brown": (100, 65, 40), "green": (40, 140, 70), "olive": (110, 110, 50),
    "purple": (120, 60, 150), "teal": (30, 130, 130),
}
NEUTRAL = {"black", "white", "grey", "navy", "beige"}
GROUP = {"shirt": "top", "longsleeve": "top", "t-shirt": "top", "pants": "bottom",
         "shorts": "bottom", "skirt": "bottom", "dress": "dress",
         "outwear": "outwear", "shoes": "shoes", "hat": "hat"}
COMPLEMENT = {"top": ["pants", "outwear", "shoes"], "bottom": ["t-shirt", "outwear", "shoes"],
              "dress": ["outwear", "shoes", "hat"], "outwear": ["t-shirt", "pants", "shoes"],
              "shoes": ["pants", "t-shirt"], "hat": ["t-shirt", "outwear"]}
SEARCH = {"longsleeve": "full sleeve top", "outwear": "jacket", "t-shirt": "t-shirt"}


def name_color(rgb):
    return min(PALETTE, key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n], rgb)))


def dominant_colors(img, n=2):
    w, h = img.size
    img = img.crop((int(w * .25), int(h * .25), int(w * .75), int(h * .75))).resize((48, 48))
    q = img.quantize(colors=4)
    pal = q.getpalette()[:12]
    out = []
    for _, i in sorted(q.getcolors(), reverse=True):
        nm = name_color(tuple(pal[i * 3:i * 3 + 3]))
        if nm not in out:
            out.append(nm)
    return out[:n]


def shop_links(color, category, gender):
    words = [gender if gender != "any" else "", color, SEARCH.get(category, category)]
    q = " ".join(w for w in words if w)
    e = urllib.parse.quote_plus(q)
    return {"Amazon.in": f"https://www.amazon.in/s?k={e}",
            "Flipkart": f"https://www.flipkart.com/search?q={e}",
            "Myntra": "https://www.myntra.com/" + "-".join(q.split()),
            "Ajio": f"https://www.ajio.com/search/?text={e}"}


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    top = cnn_model.predict_topk(data, 3)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return {"top": [{"category": c, "confidence": p} for c, p in top],
            "colors": dominant_colors(img)}


class Item(BaseModel):
    id: str
    category: str
    color: str


class StyleReq(BaseModel):
    mode: str                      # "item" or "inspo"
    category: str | None = None
    color: str | None = None
    wanted: list[str] = []
    colors: list[str] = []
    wardrobe: list[Item] = []
    gender: str = "any"
    message: str = ""


def pick(slot, wardrobe, prefer, used, avoid):
    cands = [w for w in wardrobe if w.id not in used and GROUP.get(w.category) == GROUP[slot]]
    if not cands:
        return None

    def score(w):
        s = 2 if w.category == slot else 0
        s += 2 if w.color in prefer else (1 if w.color in NEUTRAL else 0)
        if avoid and w.color == avoid and avoid not in NEUTRAL:
            s -= 2
        return s
    return max(cands, key=score)


@router.post("/style")
async def style(req: StyleReq):
    if req.mode == "item":
        base = req.category or "t-shirt"
        slots, prefer, avoid = COMPLEMENT.get(GROUP.get(base, "top"), []), [], req.color
    else:
        slots = [s for s in req.wanted if s in GROUP] or ["t-shirt"]
        base, prefer, avoid = slots[0], req.colors, None

    used, matches, missing = set(), [], []
    for s in slots:
        w = pick(s, req.wardrobe, prefer, used, avoid)
        if w:
            used.add(w.id)
            matches.append({"slot": s, "id": w.id, "category": w.category, "color": w.color})
        else:
            missing.append(s)

    hint = prefer[0] if prefer else ""
    shop = [{"label": f"{hint} {m}".strip(), "links": shop_links(hint, m, req.gender)} for m in missing]
    if req.mode == "item":
        shop.insert(0, {"label": f"similar {req.color} {req.category}",
                        "links": shop_links(req.color or "", base, req.gender)})

    have = ", ".join(f"{m['color']} {m['category']} (for {m['slot']})" for m in matches) or "nothing suitable"
    need = ", ".join(missing) or "nothing"
    goal = (f"The user has a {req.color} {req.category}. Suggest how to style it." if req.mode == "item"
            else f"The user wants to recreate an outfit with: {', '.join(slots)}. "
                 f"Inspiration colors: {', '.join(req.colors)}.")
    prompt = (f"{goal} Wardrobe pieces to use: {have}. Missing, to buy: {need}. "
              f"Give a short outfit plan using ONLY the listed wardrobe pieces, and say what to buy "
              f"for the missing ones. {req.message}")
    try:
        reply = llm.generate_reply(base, 0.9, rag.retrieve_context(base, "styling tips"), prompt, [])
    except Exception:
        reply = "The stylist model is not responding. Check that Ollama is running."
    return {"matches": matches, "missing": missing, "shop": shop, "reply": reply}