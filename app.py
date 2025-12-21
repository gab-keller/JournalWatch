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
    "Scalable clinical literature ranking from PubMed RSS "
    "(robust for 100+ articles)"
)

# =================================================
# Constants
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MAX_LLM_RANK_POOL = 40
RANK_CHUNK_SIZE = 10
RANK_ABSTRACT_MAX_CHARS = 1200

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

        doi = ""
        for aid in pa.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()

        if not title:
            continue

        articles.append(
            {
                "title": title,
                "journal": journal,
                "year": year,
                "month": month,
                "abstract": abstract,
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else "",
            }
        )

    return articles


# =================================================
# Heuristic pre-filter (NO LLM)
# =================================================
KEYWORDS = [
    "randomized", "trial", "phase",
    "guideline", "meta-analysis",
    "systematic review"
]

def heuristic_score(a: Dict[str, Any]) -> int:
    text = f"{a['title']} {a['abstract']}".lower()
    score = sum(2 for k in KEYWORDS if k in text)
    if a["abstract"]:
        score += 1
    if a["year"].isdigit():
        score += max(0, int(a["year"]) - 2016)
    return score


def prefilter_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        articles,
        key=lambda a: heuristic_score(a),
        reverse=True,
    )
    return ranked[:MAX_LLM_RANK_POOL]


# =================================================
# OpenAI helpers
# =================================================
def get_openai_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "").strip()
    if not key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=key)


def call_with_retry(fn, retries=5):
    for i in range(retries):
        try:
            return fn()
        except RateLimitError:
            time.sleep(2 ** i)
    raise RuntimeError("Rate limit exceeded repeatedly.")


# =================================================
# LLM ranking (progress-aware)
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

    chunks = [
        payload[i:i + RANK_CHUNK_SIZE]
        for i in range(0, len(payload), RANK_CHUNK_SIZE)
    ]

    progress = st.progress(0.0, text="Ranking articles by clinical relevance…")

    for idx, chunk in enumerate(chunks):
        def _call():
            return client.responses.create(
                model=RANK_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Score articles by immediate clinical relevance. "
                            "Negative RCTs = intermediate score, above basic science."
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

        progress.progress((idx + 1) / len(chunks))

    progress.empty()
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =================================================
# Enrichment
# =================================================
def summarize(client: OpenAI, abstract: str) -> str:
    if not abstract:
        return "No abstract available."
    return client.responses.create(
        model=ENRICH_MODEL,
        input=f"Summarize clinical implications in ≤2 sentences:\n\n{abstract}",
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
st.info(f"{len(filtered)} articles selected for LLM ranking.")

client = get_openai_client()
ranked = rank_articles_llm(client, filtered)
top = ranked[:top_k]

st.markdown("## Results")

for idx, score in top:
    a = filtered[idx]

    # Journal + date (less emphasis)
    st.markdown(
        f"<span style='color:#666'>{a['journal']} · {a['month']} {a['year']}</span>",
        unsafe_allow_html=True,
    )

    # Clickable bold title
    if a["doi_url"]:
        st.markdown(f"[**{a['title']}**]({a['doi_url']})")
    else:
        st.markdown(f"**{a['title']}**")

    st.caption(f"Relevance score: {score:.1f}")

    st.markdown(summarize(client, a["abstract"]))

    # Abstract dropdown
    with st.expander("Abstract"):
        st.write(a["abstract"] or "No abstract available.")
