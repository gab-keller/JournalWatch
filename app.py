import re
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from openai import OpenAI


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="PubMed RSS – Clinical Digest", layout="wide")

st.title("PubMed RSS – Clinical Digest")
st.caption("Paste a PubMed RSS link and view clinically summarized articles")


# -------------------------------------------------
# Constants
# -------------------------------------------------
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_pmid_from_link(link: str) -> Optional[str]:
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return match.group(1) if match else None


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

    response = requests.get(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)},
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    articles = []

    for pubmed_article in root.findall(".//PubmedArticle"):
        medline = pubmed_article.find("MedlineCitation")
        article = medline.find("Article")

        title = article.findtext("ArticleTitle", default="")
        journal = article.findtext("Journal/Title", default="")
        year = article.findtext("Journal/JournalIssue/PubDate/Year", default="")
        month = article.findtext("Journal/JournalIssue/PubDate/Month", default="")

        abstract_parts = article.findall(".//AbstractText")
        abstract = "\n".join(
            part.text.strip() for part in abstract_parts if part.text
        )

        if not title or not abstract:
            continue

        articles.append(
            {
                "title": title,
                "journal": journal,
                "year": year,
                "month": month,
                "abstract": abstract,
            }
        )

    return articles


# -------------------------------------------------
# OpenAI helpers (simple, safe)
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


def generate_clinical_summary(client: OpenAI, abstract: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a clinician summarizing medical articles. "
                    "Be concise, practical, and clinically oriented."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the most important conclusion of this article "
                    "and its clinical implications in no more than 2 sentences:\n\n"
                    f"{abstract}"
                ),
            },
        ],
    )
    return response.output_text.strip()


def generate_structured_abstract(client: OpenAI, abstract: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a medical editor improving readability of abstracts. "
                    "Highlight clinically relevant findings."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite the abstract below into structured paragraphs "
                    "(Background, Methods, Results, Conclusion) when appropriate. "
                    "Bold the most clinically important findings.\n\n"
                    f"{abstract}"
                ),
            },
        ],
    )
    return response.output_text.strip()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
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
with st.spinner("Fetching articles from PubMed..."):
    pmids = fetch_pmids_from_rss(rss_url)
    articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles retrieved.")
    st.stop()

st.success(f"Retrieved {len(articles)} articles")

client = get_openai_client()


# -------------------------------------------------
# Output
# -------------------------------------------------
for idx, article in enumerate(articles, start=1):
    header = f"{article['journal']}, {article['month']} {article['year']} — {article['title']}"
    st.markdown(f"**{header}**")

    with st.spinner("Generating clinical summary..."):
        summary = generate_clinical_summary(client, article["abstract"])
    st.markdown(summary)

    with st.expander("Abstract"):
        with st.spinner("Improving abstract readability..."):
            structured = generate_structured_abstract(client, article["abstract"])
        st.markdown(structured)
