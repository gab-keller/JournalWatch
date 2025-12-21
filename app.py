import re
import time
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI, RateLimitError


# =================================================
# Page setup
# =================================================
st.set_page_config(page_title="PubMed RSS – Clinical Digest", layout="wide")
st.title("PubMed RSS – Clinical Digest")
st.caption(
    "Robust clinical literature ranking from PubMed RSS "
    "(designed to scale to 100+ articles safely)"
)

# =================================================
# Constants
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MAX_LLM_RANK_POOL = 60          # hard cap sent to LLM
RANK_CHUNK_SIZE = 10            # small to avoid TPM bursts
RANK_ABSTRACT_MAX_CHARS = 1200  # aggressive truncation
RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"


# =================================================
# Helpers
# =================================================
def extract_pmid_from_link(link: str) -> Optional[str]:
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return m.group(1) if m else None


def fetch_pmids_from_rss(rss_url: str) -> List[str]:
    feed = feedparser.parse(rss_url)
    pmids = []
    for e in feed.entries:
        pmid = extract_pmid_from_link(getattr(e, "link", ""))
        if pmid:
            pmids.append(pmid)
    return list(dict.fromkeys(pmids))


def fetch_pubmed_articles(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    r = requests.get(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)},
        timeout=30,
    )
    r.raise_for_status()

    root = ET.fromstring(r.text)
    articles = []

    for pa in root.findall(".//PubmedArticle"):
        art = pa.find(".//Article")
        if art is None:
            continue

        title = (art.findtext("ArticleTitle") or "").strip()
        journal = (art.findtext("Journal/Title") or "").strip()
        year = (art.findtext("Journal/JournalIssue/PubDate/Year") or "").strip()
        month = (art.findtext("Journal/JournalIssue/PubDate/Month") or "").strip()

        abstract_parts = art.findall(".//AbstractText")
        abstract = "\n".join(
            (p.text or "").strip() for p in abstract_parts if p.text
        ).strip()

        if not title:
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


# =================================================
# Stage 1 — deterministic pre-ranking (NO LLM)
# =================================================
KEYWORDS = [
    "randomized",
    "trial",
    "phase",
    "guideline",
    "meta-analysis",
    "systematic review",
]

def heuristic_score(article: Dict[str, Any]) -> int:
    score = 0
    text = f"{article['title']} {article['abstract']}".lower()

    for kw in KEYWORDS:
        if kw in text:
            score += 2

    if article["abstract"]:
        score += 1

    if article["year"].isdigit():
        score += max(0, int(article["year"]) - 2015)

    return score


def prefilter_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [(heuristic_score(a), a) for a in articles]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:MAX_LLM_RANK_POOL]]


# =================================================
# OpenAI helpers
# =================================================
def get_openai_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "").strip()
    if not key:
        st.warning("Enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=key)


def call_with_retry(func, max_retries=5):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait = 2 ** i
            time.sleep(wait)
    raise RuntimeError("Repeated rate-limit failures.")


# =================================================
# Stage 2 — LLM ranking (reduced set only)
# =================================================
def rank_articles_llm(
    client: OpenAI,
    articles: List[Dict[str, Any]],
) -> List[Tuple[int, float]]:

    ranked = []
    payload = []

    for i, a in enumerate(articles):
        payload.append(
            f"{i} | {a['title']} | {a['journal']} | "
            f"{(a['abstract'][:RANK_ABSTRACT_MAX_CHARS])}"
        )

    chunks = [payload[i:i + RANK_CHUNK_SIZE] for i in range(0, len(payload), RANK_CHUNK_SIZE)]

    for chunk in chunks:
        def _call():
            return client.responses.create(
                model=RANK_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Score articles by immediate clinical relevance.\n"
                            "Negative RCTs = intermediate scores, "
                            "always above basic science or case series."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Score each line from 0–100.\n"
                            "Format strictly: id | score\n\n"
                            + "\n".join(chunk)
                        ),
                    },
                ],
            )

        resp = call_with_retry(_call)
        for line in resp.output_text.splitlines():
            if "|" in line:
                try:
                    i, s = line.split("|")
                    ranked.append((int(i.strip()), float(s.strip())))
                except:
                    pass

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =================================================
# Enrichment (Top-N only)
# =================================================
def summarize(client: OpenAI, abstract: str) -> str:
    if not abstract:
        return "No abstract available."
    return client.responses.create(
        model=ENRICH_MODEL,
        input=f"Summarize the clinical implication in ≤2 sentences:\n\n{abstract}",
    ).output_text.strip()


# =================================================
# UI
# =================================================
with st.sidebar:
    st.text_input("OpenAI API key", type="password", key="openai_api_key")
    top_k = st.number_input("Number of articles to display", 1, 50, 20)

rss = st.text_input("Paste PubMed RSS link")
if not rss:
    st.stop()

pmids = fetch_pmids_from_rss(rss)
articles = fetch_pubmed_articles(pmids)

st.success(f"{len(articles)} articles retrieved.")

filtered = prefilter_articles(articles)
st.info(f"{len(filtered)} articles passed heuristic pre-filter.")

client = get_openai_client()
ranked = rank_articles_llm(client, filtered)
top = ranked[:top_k]

st.markdown("## Results")

for idx, score in top:
    art = filtered[idx]
    st.markdown(f"**{art['journal']}, {art['month']} {art['year']} — {art['title']}**")
    st.caption(f"Relevance score: {score:.1f}")
    st.markdown(summarize(client, art["abstract"]))
