import asyncio
import hashlib
import json
import logging
import os
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import urlparse

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

# Pre-cached domain lookup: domain -> analysis data
_precached_domains: dict[str, dict] = {}
_precached_meta: dict[str, dict] = {}  # domain -> {privacy_url, score, risk}


def get_cache_key(url: str) -> str:
    return hashlib.sha256(url.lower().strip().encode()).hexdigest()[:16]


def extract_domain(url: str) -> str:
    """Extract root domain from any URL (e.g., mail.google.com -> google.com)."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        hostname = parsed.hostname or ""
        parts = hostname.lower().split(".")
        # Handle common TLDs like .co.uk, .co.in, .co.jp
        if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "edu", "gov"):
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname
    except Exception:
        return ""


def get_precached_analysis(url: str) -> dict | None:
    """Check if the URL's domain matches a pre-cached entry."""
    domain = extract_domain(url)
    if domain in _precached_domains:
        logger.info(f"⚡ Pre-cache HIT for domain: {domain}")
        return _precached_domains[domain]
    return None


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


def load_precached_policies() -> int:
    """Load pre-cached privacy analysis from JSON file."""
    json_path = pathlib.Path(__file__).parent / "precached_policies.json"
    if not json_path.exists():
        logger.warning("⚠ precached_policies.json not found")
        return 0

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        domains = data.get("domains", {})
        count = 0

        for domain, entry in domains.items():
            analysis = entry.get("analysis", {})
            privacy_url = entry.get("privacy_url", "")

            # Store under primary domain
            _precached_domains[domain] = analysis
            _precached_meta[domain] = {
                "domain": domain,
                "privacy_url": privacy_url,
                "overall_score": analysis.get("overall_score", 0),
                "overall_risk": analysis.get("overall_risk", "unknown"),
            }
            count += 1

            # Store under alias domains
            for alias in entry.get("aliases", []):
                _precached_domains[alias] = analysis
                _precached_meta[alias] = {
                    "domain": alias,
                    "privacy_url": privacy_url,
                    "overall_score": analysis.get("overall_score", 0),
                    "overall_risk": analysis.get("overall_risk", "unknown"),
                    "alias_of": domain,
                }

        logger.info(f"✅ Loaded {count} pre-cached privacy policies ({len(_precached_domains)} domains total)")
        return count
    except Exception as e:
        logger.error(f"❌ Failed to load precached policies: {e}")
        return 0


# --------- APP ---------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load pre-cached policies on startup."""
    count = load_precached_policies()
    logger.info(f"🚀 Server ready with {count} pre-cached sites")
    yield


app = FastAPI(
    title="Trust Layer Backend",
    description="Privacy policy analyzer - OpenAI Edition",
    version="0.7.0",
    lifespan=lifespan,
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
    precached: bool = False


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


def parse_analysis(data: dict, cached: bool = False, precached: bool = False) -> AnalyzeResponse:
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
        precached=precached,
    )


# --------- ROUTES ---------


@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "trust-layer-backend",
        "model": OPENAI_MODEL,
        "cache_size": len(_analysis_cache),
        "precached_domains": len(_precached_domains),
        "version": "0.7.0",
    }


@app.post("/analyze-policy", response_model=AnalyzeResponse)
@limiter.limit(RATE_LIMIT)
async def analyze_policy(request: Request, req: AnalyzeRequest):
    """Analyze a privacy policy URL."""
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # 1. Pre-cached domain check = instant, survives cold starts
    precached_data = get_precached_analysis(req.url)
    if precached_data:
        return parse_analysis(precached_data, cached=True, precached=True)

    # 2. Runtime cache check = instant for repeat requests
    cached_data = get_cached_analysis(req.url)
    if cached_data:
        return parse_analysis(cached_data, cached=True)

    # 3. Live analysis via OpenAI
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


@app.get("/precached")
async def precached_sites():
    """List all pre-cached domains and their scores."""
    # Deduplicate: only show primary domains (skip aliases)
    sites = []
    seen = set()
    for domain, meta in _precached_meta.items():
        primary = meta.get("alias_of", domain)
        if primary not in seen:
            seen.add(primary)
            primary_meta = _precached_meta.get(primary, meta)
            sites.append({
                "domain": primary,
                "privacy_url": primary_meta.get("privacy_url", ""),
                "overall_score": primary_meta.get("overall_score", 0),
                "overall_risk": primary_meta.get("overall_risk", "unknown"),
            })
    return {
        "count": len(sites),
        "sites": sorted(sites, key=lambda s: s["domain"]),
    }


@app.get("/cache/stats")
async def cache_stats():
    return {
        "runtime_cache_size": len(_analysis_cache),
        "precached_domains": len(_precached_domains),
        "max": CACHE_MAX_SIZE,
        "ttl_hours": CACHE_TTL_HOURS,
    }


@app.delete("/cache/clear")
async def clear_cache():
    global _analysis_cache
    count = len(_analysis_cache)
    _analysis_cache = {}
    return {"cleared": count, "note": "Pre-cached entries are not affected"}
