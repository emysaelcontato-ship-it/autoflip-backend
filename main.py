import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

# OpenAI SDK novo
from openai import OpenAI

# Supabase client
from supabase import create_client, Client

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not (OPENAI_API_KEY and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
    print("Missing envs. Please set OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI()

# CORS: permita seu domínio do Softr e localhost
origins = [
    "http://localhost:3000",
    "https://*.softr.app",
    "https://*.softr.io",
    # adicione seu domínio Softr exato quando publicar
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se quiser restringir, troque pelas origins acima
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeInput(BaseModel):
    user_email: str
    lot_url: Optional[str] = None
    auctioneer: Optional[str] = None
    car_title: Optional[str] = None
    year: Optional[str] = None
    km: Optional[str] = None
    condition_notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class AnalyzeOutput(BaseModel):
    score: float = Field(..., description="0-100")
    recommended_bid: float
    margin: float
    risk_level: str
    reasoning: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "autoflip-backend"}

@app.post("/users/upsert")
def upsert_user(payload: Dict[str, Any]):
    """
    payload: { "email": "...", "name": "..." }
    """
    email = payload.get("email")
    if not email:
        raise HTTPException(400, "email is required")

    data, error = supabase.table("users").upsert({"email": email, "name": payload.get("name")}).execute()
    if error:
        raise HTTPException(500, f"supabase error: {error}")
    return {"ok": True}

@app.post("/analyze", response_model=AnalyzeOutput)
def analyze(body: AnalyzeInput):
    if not body.user_email:
        raise HTTPException(400, "user_email is required")

    # 1) Gera uma análise com OpenAI (prompt simples para começar)
    messages = [
        {"role": "system", "content": "Você é um avaliador de oportunidades de lucro em leilões de carros no Brasil. Responda em JSON."},
        {"role": "user", "content": f"""
Lote: {body.lot_url}
Leiloeiro: {body.auctioneer}
Veículo: {body.car_title}
Ano: {body.year}
KM: {body.km}
Condições/Observações: {body.condition_notes}

Tarefa:
- Dê uma nota de 0 a 100 (score) para potencial de lucro.
- Informe lance máximo recomendado (em R$) e margem estimada (em %).
- Classifique risco em LOW/MEDIUM/HIGH.
- Inclua breve justificativa.
Responda somente JSON com as chaves: score, recommended_bid, margin, risk_level, reasoning.
        """}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {str(e)}")

    # 2) Tenta parsear o JSON (fallback simples se vier texto)
    import json
    try:
        data = json.loads(content)
    except Exception:
        data = {
            "score": 60,
            "recommended_bid": 30000,
            "margin": 15,
            "risk_level": "MEDIUM",
            "reasoning": content[:500],
        }

    # normaliza tipos
    score = float(data.get("score", 60))
    recommended_bid = float(data.get("recommended_bid", 30000))
    margin = float(data.get("margin", 15))
    risk_level = str(data.get("risk_level", "MEDIUM"))
    reasoning = str(data.get("reasoning", ""))

    # 3) Salva no Supabase
    insert_payload = {
        "user_email": body.user_email,
        "lot_url": body.lot_url,
        "auctioneer": body.auctioneer,
        "car_title": body.car_title,
        "raw_input": {
            "year": body.year,
            "km": body.km,
            "condition_notes": body.condition_notes,
            "extra": body.extra,
            "ai_reasoning": reasoning
        },
        "score": score,
        "recommended_bid": recommended_bid,
        "margin": margin,
        "risk_level": risk_level,
    }
    _, error = supabase.table("analyses").insert(insert_payload).execute()
    if error:
        raise HTTPException(500, f"supabase insert error: {error}")

    return AnalyzeOutput(
        score=score,
        recommended_bid=recommended_bid,
        margin=margin,
        risk_level=risk_level,
        reasoning=reasoning
    )
