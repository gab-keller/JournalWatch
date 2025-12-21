import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Clinical Article Digest", layout="wide")

st.title("Clinical Article Digest")
st.caption(
    "Upload an Excel file → select the most clinically relevant articles → "
    "read concise summaries and structured abstracts"
)


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

MAX_OUTPUT_ARTICLES = 20


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def safe_str(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def parse_date(x: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def month_year_from_date(ts: Optional[pd.Timestamp]) -> Tuple[str, str]:
    if ts is None:
        return "", ""
    return ts.strftime("%B"), ts.strftime("%Y")


def doi_to_url(doi: str) -> str:
    return f"https://doi.org/{doi}" if doi else ""


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def build_articles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    articles = []

    for _, row in df.iterrows():
        title = safe_str(row.get("Article Title"))
        if not title:
            continue

        doi = safe_str(row.get("DOI"))
        journal = safe_str(row.get("Journal"))
        abstract = safe_str(row.get("Abstract"))

        ts = parse_date(row.get("Published Date"))
        month, year = month_year_from_date(ts)

        articles.append(
            {
                "journal": journal,
                "month": month,
                "year": year,
                "title": title,
                "doi": doi,
                "link": doi_to_url(doi),
                "abstract": abstract,
            }
        )

    # De-duplicate
    seen = set()
    out = []
    for a in articles:
        key = a["doi"] or a["title"].lower()
        if key not in seen:
            seen.add(key)
            out.append(a)

    return out


# -------------------------------------------------
# OpenAI client (user-provided key)
# -------------------------------------------------
def get_openai_client_from_user() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


# -------------------------------------------------
# OpenAI processing
# -------------------------------------------------
def select_and_enrich_articles(
    client: OpenAI,
    model: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    1) If >20 articles → select top 20 by clinical relevance
    2) Sort by immediate clinical impact
    3) Generate:
       - 2-sentence clinical summary
       - Structured abstract with bold highlights
    """

    system_prompt = (
        "You are a senior clinical neurologist and journal editor. "
        "Your task is to identify the articles with the highest immediate "
        "clinical relevance and present them in a clinician-friendly format."
    )

    user_prompt = {
        "instructions": (
            "From the list of articles below:\n"
            "1. If there are more than 20 articles, select ONLY the 20 most clinically relevant.\n"
            "2. Rank them by immediate impact on clinical practice.\n"
            "3. For each selected article, produce:\n"
            "   - A brief clinical summary (max 2 sentences) focused on main conclusion and implications.\n"
            "   - A structured abstract, broken into paragraphs when possible "
            "     (Background, Methods, Results, Conclusion).\n"
            "   - Highlight the most clinically important findings in **bold**.\n\n"
            "Return ONLY valid JSON with the key 'articles'."
        ),
        "articles": articles,
        "output_format": {
            "articles": [
                {
                    "journal": "",
                    "month": "",
                    "year": "",
                    "title": "",
                    "doi": "",
                    "link": "",
                    "clinical_summary": "",
                    "structured_abstract": "",
                }
            ]
        },
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )

    try:
        parsed = json.loads(response.output_text)
        return parsed["articles"]
    except Exception:
        st.error("Failed to parse model output.")
        st.stop()


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
    )


# -------------------------------------------------
# File upload
# -------------------------------------------------
uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel file to continue.")
    st.stop()

df = pd.read_excel(uploaded)

missing = validate_columns(df)
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

articles = build_articles(df)
st.success(f"{len(articles)} articles loaded.")


# -------------------------------------------------
# OpenAI processing
# -------------------------------------------------
with st.spinner("Selecting and enriching the most clinically relevant articles..."):
    client = get_openai_client_from_user()
    final_articles = select_and_enrich_articles(client, model, articles)


# -------------------------------------------------
# Output
# -------------------------------------------------
st.markdown("## Prioritized articles")

for i, a in enumerate(final_articles, start=1):
    # Smaller header text (same level as title)
    st.markdown(
        f"**{i}. {a['journal']}, {a['month']} {a['year']}**",
        help="Journal and publication date",
    )

    # Clickable title
    if a["link"]:
        st.markdown(f"[{a['title']}]({a['link']})")
    else:
        st.markdown(a["title"])

    # Clinical summary
    st.markdown(f"**Clinical summary:** {a['clinical_summary']}")

    st.caption(f"DOI: {a['doi'] or '—'}")

    # Structured abstract
    with st.expander("Abstract"):
        st.markdown(a["structured_abstract"], unsafe_allow_html=False)
