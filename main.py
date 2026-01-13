import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from openai import OpenAI
from pydantic import BaseModel

# --------- LOGGING ---------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trust-layer")

# --------- CONFIG ---------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # fast & cheap
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------- RATE LIMITER ---------

limiter = Limiter(key_func=get_remote_address)

# --------- CACHE ---------

_analysis_cache: dict[str, tuple[datetime, dict]] = {}
CACHE_MAX_SIZE = 200
CACHE_TTL_HOURS = 48


def get_cache_key(url: str) -> str:
    return hashlib.sha256(url.lower().strip().encode()).hexdigest()[:16]


def get_cached_analysis(url: str) -> dict | None:
    key = get_cache_key(url)
    if key in _analysis_cache:
        timestamp, data = _analysis_cache[key]
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            logger.info(f"⚡ Cache HIT")
            return data
        del _analysis_cache[key]
    return None


def set_cached_analysis(url: str, data: dict) -> None:
    if len(_analysis_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_analysis_cache, key=lambda k: _analysis_cache[k][0])
        del _analysis_cache[oldest_key]
    _analysis_cache[get_cache_key(url)] = (datetime.now(), data)


# --------- APP ---------

app = FastAPI(
    title="Trust Layer Backend",
    description="Privacy policy analyzer - OpenAI Edition",
    version="0.6.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- MODELS ---------


class RiskCategory(BaseModel):
    id: str
    label: str
    level: str
    note: str


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    overall_risk: str
    overall_score: float
    categories: list[RiskCategory]
    bullets: list[str]
    recommendation: str
    cached: bool = False


# --------- HELPERS ---------


async def fetch_policy_html(url: str) -> str:
    """Fetch HTML with reliable timeout."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=True,
        headers={"User-Agent": "TrustLayer/0.6"},
        http2=True,
    ) as http_client:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return resp.text


def extract_visible_text(html: str) -> str:
    """Fast text extraction."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    return cleaned[:10000]


def build_prompt(policy_text: str) -> str:
    """Optimized prompt for speed."""
    return f"""Analyze this privacy policy for user privacy risks. Return ONLY valid JSON:

{{
  "overall_risk": "low"|"medium"|"high",
  "overall_score": 1-10 (10=safest),
  "categories": [
    {{"id": "data_sharing", "label": "Data sharing", "level": "low"|"medium"|"high", "note": "brief"}},
    {{"id": "tracking", "label": "Tracking", "level": "...", "note": "brief"}},
    {{"id": "ai_training", "label": "AI training", "level": "...", "note": "brief"}},
    {{"id": "retention", "label": "Data retention", "level": "...", "note": "brief"}},
    {{"id": "security", "label": "Security", "level": "...", "note": "brief"}},
    {{"id": "rights_control", "label": "User rights", "level": "...", "note": "brief"}}
  ],
  "bullets": ["risk1", "risk2", "risk3"],
  "recommendation": "one sentence advice"
}}

Policy:
\"\"\"{policy_text}\"\"\""""


async def call_openai_async(prompt: str) -> dict:
    """Async OpenAI call - non-blocking."""
    loop = asyncio.get_event_loop()
    
    def sync_call():
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    raw = await loop.run_in_executor(None, sync_call)
    
    # Parse JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if "```json" in raw:
            start = raw.find("```json") + 7
            end = raw.find("```", start)
            return json.loads(raw[start:end].strip())
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def parse_analysis(data: dict, cached: bool = False) -> AnalyzeResponse:
    """Parse response into model."""
    categories = []
    for c in data.get("categories", []) or []:
        try:
            categories.append(RiskCategory(
                id=str(c.get("id", "")),
                label=str(c.get("label", "")),
                level=str(c.get("level", "medium")).lower(),
                note=str(c.get("note", "")),
            ))
        except:
            continue

    try:
        score = float(data.get("overall_score", 5.0))
    except:
        score = 5.0

    return AnalyzeResponse(
        overall_risk=str(data.get("overall_risk", "medium")).lower(),
        overall_score=score,
        categories=categories,
        bullets=list(data.get("bullets", []))[:5],
        recommendation=str(data.get("recommendation", "")),
        cached=cached,
    )


# --------- ROUTES ---------


@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "trust-layer-backend",
        "model": OPENAI_MODEL,
        "cache_size": len(_analysis_cache),
        "version": "0.6.0",
    }


@app.post("/analyze-policy", response_model=AnalyzeResponse)
@limiter.limit(RATE_LIMIT)
async def analyze_policy(request: Request, req: AnalyzeRequest):
    """Analyze a privacy policy URL."""
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Cache hit = instant response
    cached_data = get_cached_analysis(req.url)
    if cached_data:
        return parse_analysis(cached_data, cached=True)

    try:
        html = await fetch_policy_html(req.url)
        text = extract_visible_text(html)
        
        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Not enough policy text found")

        prompt = build_prompt(text)
        data = await call_openai_async(prompt)
        
        set_cached_analysis(req.url, data)
        return parse_analysis(data, cached=False)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from AI: {e}") from e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/cache/stats")
async def cache_stats():
    return {"size": len(_analysis_cache), "max": CACHE_MAX_SIZE, "ttl_hours": CACHE_TTL_HOURS}


@app.delete("/cache/clear")
async def clear_cache():
    global _analysis_cache
    count = len(_analysis_cache)
    _analysis_cache = {}
    return {"cleared": count}
