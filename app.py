import json
import re
import requests
import feedparser
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


# =================================================
# Page configuration
# =================================================
st.set_page_config(
    page_title="Clinical Article Digest – PubMed RSS",
    layout="wide",
)

st.title("Clinical Article Digest – PubMed RSS")
st.caption(
    "Paste a PubMed RSS link → review the most clinically relevant articles → "
    "generate a podcast-style summary and audio"
)


# =================================================
# Constants
# =================================================
MAX_OUTPUT_ARTICLES = 20
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# =================================================
# Utility helpers
# =================================================
def safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def doi_to_url(doi: str) -> str:
    return f"https://doi.org/{doi}" if doi else ""


def extract_pmid_from_link(link: str) -> Optional[str]:
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return match.group(1) if match else None


# =================================================
# PubMed RSS + E-utilities
# =================================================
def fetch_pmids_from_rss(rss_url: str) -> List[str]:
    feed = feedparser.parse(rss_url)
    pmids = []

    for entry in feed.entries:
        pmid = extract_pmid_from_link(entry.link)
        if pmid:
            pmids.append(pmid)

    return list(dict.fromkeys(pmids))


def fetch_pubmed_articles(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
    }

    response = requests.get(PUBMED_EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        art = medline.find("Article")

        title = safe_str(art.findtext("ArticleTitle"))
        journal = safe_str(art.findtext("Journal/Title"))

        abstract_parts = art.findall(".//AbstractText")
        abstract = "\n".join(safe_str(a.text) for a in abstract_parts)

        year = safe_str(art.findtext("Journal/JournalIssue/PubDate/Year"))
        month = safe_str(art.findtext("Journal/JournalIssue/PubDate/Month"))

        doi = ""
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.attrib.get("IdType") == "doi":
                doi = safe_str(id_elem.text)

        if not title or not abstract:
            continue

        articles.append(
            {
                "journal": journal,
                "month": month,
                "year": year,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "link": doi_to_url(doi),
            }
        )

    return articles


# =================================================
# OpenAI client (user-provided key)
# =================================================
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


# =================================================
# OpenAI – Structured Outputs (CRITICAL FIX)
# =================================================
def select_and_enrich_articles(
    client: OpenAI,
    model: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    schema = {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "journal": {"type": "string"},
                        "month": {"type": "string"},
                        "year": {"type": "string"},
                        "title": {"type": "string"},
                        "doi": {"type": "string"},
                        "link": {"type": "string"},
                        "clinical_summary": {"type": "string"},
                        "structured_abstract": {"type": "string"},
                    },
                    "required": [
                        "journal",
                        "month",
                        "year",
                        "title",
                        "doi",
                        "link",
                        "clinical_summary",
                        "structured_abstract",
                    ],
                },
            }
        },
        "required": ["articles"],
    }

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a senior clinical neurologist and medical journal editor. "
                    "Select the articles with the highest immediate clinical relevance."
                ),
            },
            {
                "role": "user",
                "content": (
                    "From the provided articles:\n"
                    "- If more than 20, select ONLY the top 20.\n"
                    "- Rank by immediate clinical impact.\n"
                    "- Write a clinical summary (max 2 sentences).\n"
                    "- Provide a structured abstract (Background, Methods, Results, Conclusion).\n"
                    "- Highlight key findings in **bold**."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(articles, ensure_ascii=False),
            },
        ],
        schema=schema,  # ✅ CORRECT for openai==2.14.0
    )

    return response.output_parsed["articles"]



def generate_podcast_script(
    client: OpenAI,
    model: str,
    selected_articles: List[Dict[str, Any]],
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a medical podcast host speaking to neurologists.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": (
                            "Create a podcast-style script of about 10 minutes "
                            "(~1300–1500 words). Use natural spoken language, "
                            "smooth transitions, and focus on clinical implications."
                        ),
                        "articles": selected_articles,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return response.output_text


def generate_tts_audio(client: OpenAI, script: str) -> bytes:
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=script,
    )
    return speech.read()


# =================================================
# Sidebar
# =================================================
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


# =================================================
# RSS input
# =================================================
rss_url = st.text_input(
    "Paste PubMed RSS link",
    placeholder="https://pubmed.ncbi.nlm.nih.gov/rss/search/...",
)

if not rss_url:
    st.info("Paste a PubMed RSS link to continue.")
    st.stop()


# =================================================
# Fetch articles
# =================================================
with st.spinner("Fetching articles from PubMed..."):
    pmids = fetch_pmids_from_rss(rss_url)
    articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles could be retrieved from this RSS feed.")
    st.stop()

st.success(f"{len(articles)} articles retrieved from PubMed.")


# =================================================
# Enrichment
# =================================================
with st.spinner("Selecting and enriching the most clinically relevant articles..."):
    client = get_openai_client()
    final_articles = select_and_enrich_articles(client, model, articles)


# =================================================
# Display + selection
# =================================================
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


# =================================================
# Podcast
# =================================================
st.markdown("---")
st.markdown("## 🎙️ Podcast")

if st.button("Generate podcast script from selected articles"):
    if not selected_for_podcast:
        st.warning("Select at least one article.")
    else:
        with st.spinner("Generating podcast script..."):
            st.session_state["podcast_script"] = generate_podcast_script(
                client, model, selected_for_podcast
            )

if "podcast_script" in st.session_state:
    with st.expander("Podcast script preview"):
        st.write(st.session_state["podcast_script"])

    if st.button("Generate audio podcast (MP3)"):
        with st.spinner("Generating audio..."):
            audio = generate_tts_audio(client, st.session_state["podcast_script"])
            st.download_button(
                "Download podcast audio (MP3)",
                data=audio,
                file_name="clinical_podcast.mp3",
                mime="audio/mpeg",
            )
