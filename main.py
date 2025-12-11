import json
import os

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# --------- CONFIG ---------

# Load .env for local development
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # 4.1-mini by default

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # extension can call from chrome-extension://
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Pydantic MODELS ---------


class RiskCategory(BaseModel):
    id: str
    label: str
    level: str  # "low" | "medium" | "high"
    note: str


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    overall_risk: str       # "low" | "medium" | "high"
    overall_score: float    # 1–10, 1 bad, 10 good
    categories: list[RiskCategory]
    bullets: list[str]
    recommendation: str


# --------- HELPERS ---------


async def fetch_policy_html(url: str) -> str:
    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "TrustLayer/0.2"},
    ) as client_http:
        resp = await client_http.get(url)
        resp.raise_for_status()
        return resp.text


def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    # Hard cap so we don't send insane tokens
    return cleaned[:12000]


def build_prompt(policy_text: str) -> str:
    return f"""
You are a privacy policy risk analyzer helping a normal internet user.

Read the privacy policy text below and respond ONLY with a valid JSON object
with this exact structure:

{{
  "overall_risk": "low" | "medium" | "high",
  "overall_score": number from 1 to 10,
  "categories": [
    {{
      "id": "data_sharing",
      "label": "Data sharing",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }},
    {{
      "id": "tracking",
      "label": "Tracking & profiling",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }},
    {{
      "id": "ai_training",
      "label": "AI & model training",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }},
    {{
      "id": "retention",
      "label": "Data retention",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }},
    {{
      "id": "security",
      "label": "Security & breaches",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }},
    {{
      "id": "rights_control",
      "label": "User control & rights",
      "level": "low" | "medium" | "high",
      "note": "short explanation"
    }}
  ],
  "bullets": [
    "very short risk or protection in plain language (aimed at the user)",
    "..."
  ],
  "recommendation": "one short sentence of advice to the USER (e.g. 'Okay for normal use, avoid sharing very sensitive data')."
}}

Meaning of overall_score:
- 1 = very risky for the user (bad)
- 10 = very safe for the user (good)
- higher scores always mean better privacy for the USER.

Rules:
- Speak to the USER, not the company.
- Use lowercase for overall_risk and for each category.level.
- overall_score must be a number (float or int).
- Each bullet must be <= 18 words.
- Do not include any extra text, comments, or explanations outside the JSON.

Policy text:
\"\"\"{policy_text}\"\"\"
"""


def call_llm(prompt: str) -> AnalyzeResponse:
    # Use the Responses API so we can just read .output_text
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    raw = resp.output_text

    # Try to parse JSON, with a fallback that trims extra text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])

    raw_categories = data.get("categories", []) or []
    categories: list[RiskCategory] = []
    for c in raw_categories:
        try:
            categories.append(
                RiskCategory(
                    id=str(c.get("id", "")),
                    label=str(c.get("label", "")),
                    level=str(c.get("level", "")).lower(),
                    note=str(c.get("note", "")),
                )
            )
        except Exception:
            continue

    return AnalyzeResponse(
        overall_risk=str(data.get("overall_risk", "unknown")).lower(),
        overall_score=float(data.get("overall_score", 1.0)),
        categories=categories,
        bullets=list(data.get("bullets", [])),
        recommendation=str(data.get("recommendation", "")),
    )


# --------- ROUTE ---------


@app.post("/analyze-policy", response_model=AnalyzeResponse)
async def analyze_policy(req: AnalyzeRequest):
    if not req.url.startswith("http"):
        raise HTTPException(
            status_code=400, detail="URL must start with http or https"
        )

    try:
        html = await fetch_policy_html(req.url)
        text = extract_visible_text(html)
        if not text:
            raise HTTPException(
                status_code=400, detail="Could not extract text from policy"
            )

        prompt = build_prompt(text)
        analysis = call_llm(prompt)
        return analysis

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch policy URL: {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Model returned invalid JSON: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e
