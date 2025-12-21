import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Clinical Article Digest", layout="wide")
st.title("Clinical Article Digest (Excel → Prioritized List)")
st.caption("Upload an Excel file → prioritize clinically actionable papers → expandable abstracts.")

# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLUMNS = [
    "DOI",
    "Journal",
    "Published Date",
    "Article Title",
    "Abstract",
]

OPTIONAL_COLUMNS = [
    "Article URL",
    "Publisher",
    "Authors",
]

def safe_str(x: Any) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def parse_date(x: Any) -> Optional[pd.Timestamp]:
    # Handles datetime, pandas timestamp, or string like "2025-11-11"
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None

def month_year_from_date(ts: Optional[pd.Timestamp]) -> Tuple[str, str]:
    if ts is None:
        return ("", "")
    return (ts.strftime("%B"), ts.strftime("%Y"))

def doi_to_url(doi: str) -> str:
    doi = doi.strip()
    if not doi:
        return ""
    # Standard DOI resolver
    return f"https://doi.org/{doi}"

def validate_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def build_articles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        doi = safe_str(row.get("DOI", ""))
        journal = safe_str(row.get("Journal", ""))
        title = safe_str(row.get("Article Title", ""))
        abstract = safe_str(row.get("Abstract", ""))
        pub_date_raw = row.get("Published Date", "")
        pub_ts = parse_date(pub_date_raw)
        month, year = month_year_from_date(pub_ts)

        article_url = safe_str(row.get("Article URL", ""))
        doi_url = doi_to_url(doi)
        full_link = doi_url if doi_url else article_url  # fallback if no DOI

        # Skip empty rows
        if not (title or abstract or doi or article_url):
            continue

        articles.append(
            {
                "doi": doi,
                "doi_url": doi_url,
                "article_url": article_url,
                "link": full_link,
                "journal": journal,
                "month": month,
                "year": year,
                "published_date": pub_ts.isoformat() if pub_ts is not None else "",
                "title": title,
                "abstract": abstract,
            }
        )

    # Deduplicate (prefer DOI, otherwise title)
    seen = set()
    deduped = []
    for a in articles:
        key = a["doi"] or a["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    return deduped

def get_openai_client() -> OpenAI:
    # Prefer Streamlit secrets, then env var
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets or environment variables.")
        st.stop()

    return OpenAI(api_key=api_key)

def call_openai_prioritize(
    client: OpenAI,
    model: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ask the model to rank articles by immediate clinical impact and return
    an ordered list with fields we need for display.

    We request strict JSON to make parsing reliable. If parsing fails,
    we fall back to original order.
    """

    # Keep the prompt compact to avoid token bloat.
    # We include enough info to prioritize: title + journal + date + abstract + link.
    payload = [
        {
            "journal": a["journal"],
            "month": a["month"],
            "year": a["year"],
            "published_date": a["published_date"],
            "title": a["title"],
            "abstract": a["abstract"],
            "link": a["link"],
            "doi": a["doi"],
        }
        for a in articles
    ]

    system = (
        "You are a clinically oriented neurology editor. "
        "Prioritize papers by likelihood of immediate impact on clinical practice. "
        "Prefer RCTs, guideline-informing evidence, large high-quality cohorts, and actionable diagnostic/treatment findings. "
        "Return only valid JSON."
    )

    user = {
        "task": "Sort the articles by immediate clinical implication (most actionable first). "
                "Return an array named 'articles' in the sorted order. "
                "Each array item must contain exactly these keys: "
                "journal, month, year, title, link, doi, abstract. "
                "Do not invent data; if missing, use empty string. "
                "Do not add extra keys.",
        "input_articles": payload,
        "output_json_example": {
            "articles": [
                {
                    "journal": "Neurology",
                    "month": "November",
                    "year": "2025",
                    "title": "Example title",
                    "link": "https://doi.org/...",
                    "doi": "10.xxxx/xxxxx",
                    "abstract": "Full abstract text..."
                }
            ]
        }
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )

    text = resp.output_text.strip()

    # Basic JSON extraction (in case model wraps in whitespace)
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or "articles" not in data:
            raise ValueError("JSON missing 'articles'")
        if not isinstance(data["articles"], list):
            raise ValueError("'articles' is not a list")
        # Light validation
        needed = {"journal", "month", "year", "title", "link", "doi", "abstract"}
        out = []
        for item in data["articles"]:
            if not isinstance(item, dict):
                continue
            cleaned = {k: safe_str(item.get(k, "")) for k in needed}
            out.append(cleaned)
        if not out:
            raise ValueError("No valid articles returned")
        return out
    except Exception:
        st.warning("Could not parse model output as JSON. Showing original order instead.")
        return [
            {
                "journal": a["journal"],
                "month": a["month"],
                "year": a["year"],
                "title": a["title"],
                "link": a["link"],
                "doi": a["doi"],
                "abstract": a["abstract"],
            }
            for a in articles
        ]

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=["gpt-5.2", "gpt-5", "gpt-4.1"],
        index=0,
        help="Choose the model used to prioritize the papers."
    )
    st.markdown("---")
    st.subheader("API Key")
    st.write("Provide `OPENAI_API_KEY` via Streamlit secrets or environment variable.")

# -----------------------------
# Upload and process
# -----------------------------
uploaded = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read the Excel file: {e}")
    st.stop()

missing = validate_columns(df)
if missing:
    st.error(f"Your file is missing required columns: {', '.join(missing)}")
    st.stop()

articles = build_articles(df)
if not articles:
    st.warning("No articles found in the uploaded file (after cleaning).")
    st.stop()

st.success(f"Loaded {len(articles)} articles (deduplicated).")

with st.spinner("Calling OpenAI API to prioritize articles by clinical impact..."):
    client = get_openai_client()
    prioritized = call_openai_prioritize(client=client, model=model, articles=articles)

st.markdown("## Prioritized articles")

for idx, a in enumerate(prioritized, start=1):
    journal = a["journal"]
    month = a["month"]
    year = a["year"]
    title = a["title"]
    link = a["link"]
    doi = a["doi"]
    abstract = a["abstract"]

    header_left = f"**{journal}, {month} {year}**" if (journal or month or year) else "**(Journal/date not provided)**"
    doi_line = f"DOI: {doi}" if doi else "DOI: —"

    st.markdown(f"### {idx}. {header_left}")
    # Clickable title
    if link:
        st.markdown(f"**[{title}]({link})**")
    else:
        st.markdown(f"**{title}**")

    st.caption(doi_line)

    with st.expander("Abstract"):
        st.write(abstract if abstract else "No abstract provided.")
