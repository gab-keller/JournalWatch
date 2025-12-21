import json
import tempfile
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
    "Select the most clinically relevant articles → generate a podcast-style summary → download audio"
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

    # Deduplicate
    seen = set()
    out = []
    for a in articles:
        key = a["doi"] or a["title"].lower()
        if key not in seen:
            seen.add(key)
            out.append(a)

    return out


# -------------------------------------------------
# OpenAI client (user key)
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


# -------------------------------------------------
# OpenAI processing
# -------------------------------------------------
def select_and_enrich_articles(client: OpenAI, model: str, articles: List[Dict[str, Any]]):
    system_prompt = (
        "You are a senior clinical neurologist and medical editor. "
        "Select the most clinically relevant articles and present them clearly for practicing physicians."
    )

    user_prompt = {
        "instructions": (
            "If there are more than 20 articles, select ONLY the 20 most clinically relevant. "
            "Rank them by immediate impact on clinical practice. "
            "For each selected article, generate:\n"
            "- A brief clinical summary (max 2 sentences).\n"
            "- A structured abstract with sections when possible "
            "(Background, Methods, Results, Conclusion).\n"
            "- Highlight the most clinically important findings in **bold**.\n\n"
            "Return ONLY valid JSON."
        ),
        "articles": articles,
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )

    parsed = json.loads(response.output_text)
    return parsed["articles"]


def generate_podcast_script(client: OpenAI, model: str, selected_articles: List[Dict[str, Any]]) -> str:
    system_prompt = (
        "You are a medical podcast host speaking to neurologists. "
        "Your tone is clear, engaging, and clinically focused."
    )

    user_prompt = {
        "task": (
            "Create a podcast-style script lasting approximately 10 minutes "
            "(~1300–1500 words). The script should:\n"
            "- Be written for spoken audio\n"
            "- Smoothly transition between topics\n"
            "- Summarize and contextualize the selected studies\n"
            "- Emphasize clinical relevance and practice implications\n"
            "- Avoid reading abstracts verbatim\n"
        ),
        "articles": selected_articles,
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )

    return response.output_text


def generate_tts_audio(client: OpenAI, script: str) -> bytes:
    # OpenAI TTS (mp3)
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=script,
    )
    return speech.read()


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

with st.spinner("Selecting and enriching the most clinically relevant articles..."):
    client = get_openai_client()
    final_articles = select_and_enrich_articles(client, model, articles)


# -------------------------------------------------
# Output + selection
# -------------------------------------------------
st.markdown("## Prioritized articles")

selected_for_podcast = []

for i, a in enumerate(final_articles, start=1):
    checked = st.checkbox(
        f"Include in podcast: {a['title']}",
        key=f"select_{i}",
    )

    if checked:
        selected_for_podcast.append(a)

    st.markdown(f"**{i}. {a['journal']}, {a['month']} {a['year']}**")

    st.markdown(f"[{a['title']}]({a['link']})")
    st.markdown(f"**Clinical summary:** {a['clinical_summary']}")
    st.caption(f"DOI: {a['doi'] or '—'}")

    with st.expander("Abstract"):
        st.markdown(a["structured_abstract"])


# -------------------------------------------------
# Podcast generation
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🎙️ Podcast")

if st.button("Generate podcast script from selected articles"):
    if not selected_for_podcast:
        st.warning("Please select at least one article.")
    else:
        with st.spinner("Generating podcast script..."):
            script = generate_podcast_script(client, model, selected_for_podcast)
            st.session_state["podcast_script"] = script

if "podcast_script" in st.session_state:
    with st.expander("Podcast script preview"):
        st.write(st.session_state["podcast_script"])

    if st.button("Generate audio podcast (MP3)"):
        with st.spinner("Generating audio..."):
            audio_bytes = generate_tts_audio(client, st.session_state["podcast_script"])

            st.download_button(
                label="Download podcast audio (MP3)",
                data=audio_bytes,
                file_name="clinical_podcast.mp3",
                mime="audio/mpeg",
            )
