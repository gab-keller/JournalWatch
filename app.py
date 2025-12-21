import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Clinical Article Digest",
    layout="wide"
)

st.title("Clinical Article Digest")
st.caption("Upload an Excel file → prioritize clinically actionable papers → expand abstracts")


# -------------------------------------------------
# Constants
# -------------------------------------------------
REQUIRED_COLUMNS = [
    "DOI",
    "Journal",
    "Published Date",
    "Article Title",
    "Abstract",
]


# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def safe_str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def parse_date(x: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def month_year_from_date(ts: Optional[pd.Timestamp]) -> Tuple[str, str]:
    if ts is None:
        return "", ""
    return ts.strftime("%B"), ts.strftime("%Y")


def doi_to_url(doi: str) -> str:
    if not doi:
        return ""
    return f"https://doi.org/{doi.strip()}"


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def build_articles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    articles = []

    for _, row in df.iterrows():
        doi = safe_str(row.get("DOI"))
        journal = safe_str(row.get("Journal"))
        title = safe_str(row.get("Article Title"))
        abstract = safe_str(row.get("Abstract"))

        ts = parse_date(row.get("Published Date"))
        month, year = month_year_from_date(ts)

        link = doi_to_url(doi)

        if not title:
            continue

        articles.append(
            {
                "journal": journal,
                "month": month,
                "year": year,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "link": link,
            }
        )

    # De-duplicate by DOI or title
    seen = set()
    deduped = []
    for a in articles:
        key = a["doi"] or a["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    return deduped


# -------------------------------------------------
# OpenAI client (user-provided key)
# -------------------------------------------------
def get_openai_client_from_user() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    return OpenAI(api_key=api_key)


# -------------------------------------------------
# OpenAI call
# -------------------------------------------------
def prioritize_articles(
    client: OpenAI,
    model: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    system_prompt = (
        "You are a clinically oriented neurology editor. "
        "Prioritize articles by likelihood of immediate impact on clinical practice. "
        "Give priority to randomized trials, guideline-informing evidence, "
        "and studies with direct diagnostic or therapeutic implications. "
        "Return ONLY valid JSON."
    )

    user_payload = {
        "task": (
            "Sort the articles by immediate clinical implication (most actionable first). "
            "Return a JSON object with a single key 'articles', which is a list. "
            "Each item must contain exactly these keys: "
            "journal, month, year, title, link, doi, abstract. "
            "Do not invent data."
        ),
        "articles": articles,
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    try:
        parsed = json.loads(response.output_text)
        return parsed["articles"]
    except Exception:
        st.warning("Model output could not be parsed. Showing original order.")
        return articles


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        options=["gpt-5.2", "gpt-5", "gpt-4.1"],
        index=0,
    )

    st.markdown("---")
    st.subheader("OpenAI API Key")

    st.text_input(
        "Enter your OpenAI API key",
        type="password",
        key="openai_api_key",
        placeholder="sk-...",
        help="Used only for this session. Never stored."
    )

    if st.session_state.get("openai_api_key"):
        st.success("API key provided")


# -------------------------------------------------
# File upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

missing = validate_columns(df)
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

articles = build_articles(df)
if not articles:
    st.warning("No valid articles found.")
    st.stop()

st.success(f"{len(articles)} articles loaded.")


# -------------------------------------------------
# Run OpenAI prioritization
# -------------------------------------------------
with st.spinner("Prioritizing articles by clinical impact..."):
    client = get_openai_client_from_user()
    prioritized = prioritize_articles(client, model, articles)


# -------------------------------------------------
# Output
# -------------------------------------------------
st.markdown("## Prioritized articles")

for i, a in enumerate(prioritized, start=1):
    header = f"**{a['journal']}, {a['month']} {a['year']}**".strip(", ")

    st.markdown(f"### {i}. {header}")

    if a["link"]:
        st.markdown(f"**[{a['title']}]({a['link']})**")
    else:
        st.markdown(f"**{a['title']}**")

    st.caption(f"DOI: {a['doi'] or '—'}")

    with st.expander("Abstract"):
        st.write(a["abstract"] or "No abstract available.")
