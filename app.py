import json
import re
import requests
import feedparser
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import xml.etree.ElementTree as ET


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Clinical Article Digest (PubMed RSS)", layout="wide")

st.title("Clinical Article Digest – PubMed RSS")
st.caption(
    "Paste a PubMed RSS link → select the most clinically relevant articles → "
    "generate a podcast-style summary and audio"
)


# -------------------------------------------------
# Constants
# -------------------------------------------------
MAX_OUTPUT_ARTICLES = 20
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def doi_to_url(doi: str) -> str:
    return f"https://doi.org/{doi}" if doi else ""


def extract_pmid_from_link(link: str) -> Optional[str]:
    """
    PubMed RSS links usually look like:
    https://pubmed.ncbi.nlm.nih.gov/12345678/
    """
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return match.group(1) if match else None


# -------------------------------------------------
# PubMed RSS + API
# -------------------------------------------------
def fetch_pmids_from_rss(rss_url: str) -> List[str]:
    feed = feedparser.parse(rss_url)
    pmids = []

    for entry in feed.entries:
        pmid = extract_pmid_from_link(entry.link)
        if pmid:
            pmids.append(pmid)

    return list(dict.fromkeys(pmids))  # de-duplicate


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
        article_data = medline.find("Article")

        title = safe_str(article_data.findtext("ArticleTitle"))
        abstract_elems = article_data.findall(".//AbstractText")
        abstract = "\n".join([safe_str(a.text) for a in abstract_elems])

        journal = safe_str(article_data.findtext("Journal/Title"))

        # Publication date
        year = safe_str(article_data.findtext("Journal/JournalIssue/PubDate/Year"))
        month = safe_str(article_data.findtext("Journal/JournalIssue/PubDate/Month"))

        # DOI
        doi = ""
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.attrib.get("IdType") == "doi":
                doi = safe_str(id_elem.text)

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


# -------------------------------------------------
# OpenAI
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


def select_and_enrich_articles(
    client: OpenAI,
    model: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    system_prompt = (
        "You are a senior clinical neurologist and journal editor. "
        "Identify the articles with the highest immediate clinical relevance."
    )

    user_prompt = {
        "instructions": (
            "From the list below:\n"
            "1. If more than 20 articles, select ONLY the top 20 by immediate clinical impact.\n"
            "2. Rank them by clinical relevance.\n"
            "3. For each article, generate:\n"
            "   - A brief clinical summary (max 2 sentences).\n"
            "   - A structured abstract (Background, Methods, Results, Conclusion when possible).\n"
            "   - Highlight key clinically relevant findings in **bold**.\n\n"
            "Return ONLY valid JSON with key 'articles'."
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
                            "(~1300–1500 words). Use spoken language, smooth transitions, "
                            "and emphasize clinical implications."
                        ),
                        "articles": selected_articles,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return response.output_text


def generate_tts_audio(client: OpenAI, text: str) -> bytes:
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
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
# RSS input
# -------------------------------------------------
rss_url = st.text_input(
    "Paste PubMed RSS link",
    placeholder="https://pubmed.ncbi.nlm.nih.gov/rss/search/...",
)

if not rss_url:
    st.info("Paste a PubMed RSS link to continue.")
    st.stop()


# -------------------------------------------------
# Fetch articles
# -------------------------------------------------
with st.spinner("Fetching articles from PubMed RSS..."):
    pmids = fetch_pmids_from_rss(rss_url)
    articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles could be retrieved from this RSS feed.")
    st.stop()

st.success(f"{len(articles)} articles retrieved from PubMed.")


# -------------------------------------------------
# OpenAI enrichment
# -------------------------------------------------
with st.spinner("Selecting and enriching the most clinically relevant articles..."):
    client = get_openai_client()
    final_articles = select_and_enrich_articles(client, model, articles)


# -------------------------------------------------
# Display + selection
# -------------------------------------------------
st.markdown("## Prioritized articles")

selected_for_podcast = []

for i, a in enumerate(final_articles, start=1):
    checked = st.checkbox(f"Include in podcast: {a['title']}", key=f"select_{i}")

    if checked:
        selected_for_podcast.append(a)

    st.markdown(f"**{i}. {a['journal']}, {a['month']} {a['year']}**")
    st.markdown(f"[{a['title']}]({a['link']})")
    st.markdown(f"**Clinical summary:** {a['clinical_summary']}")
    st.caption(f"DOI: {a['doi'] or '—'}")

    with st.expander("Abstract"):
        st.markdown(a["structured_abstract"])


# -------------------------------------------------
# Podcast
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🎙️ Podcast")

if st.button("Generate podcast script from selected articles"):
    if not selected_for_podcast:
        st.warning("Select at least one article.")
    else:
        with st.spinner("Generating podcast script..."):
            script = generate_podcast_script(client, model, selected_for_podcast)
            st.session_state["podcast_script"] = script

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
