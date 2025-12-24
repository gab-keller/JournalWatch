# =================================================
# Imports
# =================================================
import re
import time
import hashlib
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from openai import RateLimitError


# =================================================
# Page setup
# =================================================
st.set_page_config(page_title="PubMed RSS – Clinical Digest", layout="wide")
st.title("PubMed RSS – Clinical Digest")
st.caption(
    "PubMed RSS → scalable ranking by immediate clinical relevance → "
    "structured abstracts and podcast mode"
)


# =================================================
# Constants (DEFINED EARLY)
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"
PODCAST_MODEL = "gpt-4.1"

RANK_CHUNK_SIZE = 10
RANK_ABSTRACT_MAX_CHARS = 1200

RSS_PRESETS = {
    "Neurovascular (last three months)": (
        "https://pubmed.ncbi.nlm.nih.gov/rss/search/"
        "1pMPQVclnMM8hoClXf46VJRk2UJoMOEwMqY_dcxDxxSnsm1MyU/"
        "?limit=100&utm_campaign=pubmed-2&fc=20251224163345"
    ),
    "Neurovascular (last year)": (
        "https://pubmed.ncbi.nlm.nih.gov/rss/search/"
        "1B3BG_B5Yy7ac5xlgtFJnBammAP6bZfKOpXYiqFqF7-TrXvoBy/"
        "?limit=100&utm_campaign=pubmed-2&fc=20251224163713"
    ),
    "Custom PubMed RSS link": None,
}


# =================================================
# Generic utilities
# =================================================
def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def call_with_retry(fn, retries: int = 6):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except RateLimitError as e:
            last_err = e
            time.sleep(max(1.0, 1.5 * (2 ** i)))
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err


def get_openai_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "").strip()
    if not key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=key)


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# =================================================
# PubMed helpers
# =================================================
def extract_pmid_from_link(link: str) -> Optional[str]:
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return m.group(1) if m else None


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_pmids_from_rss(rss_url: str) -> List[str]:
    feed = feedparser.parse(rss_url)
    pmids = []
    for e in feed.entries:
        pmid = extract_pmid_from_link(getattr(e, "link", "") or "")
        if pmid:
            pmids.append(pmid)
    return list(dict.fromkeys(pmids))


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_pubmed_articles(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    r = requests.get(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)},
        timeout=45,
    )
    r.raise_for_status()

    root = ET.fromstring(r.text)
    articles = []

    for pa in root.findall(".//PubmedArticle"):
        art = pa.find(".//Article")
        if art is None:
            continue

        title = (art.findtext("ArticleTitle") or "").strip()
        if not title:
            continue

        journal = (art.findtext("Journal/Title") or "").strip()
        year = (art.findtext("Journal/JournalIssue/PubDate/Year") or "").strip()
        month = (art.findtext("Journal/JournalIssue/PubDate/Month") or "").strip()

        abstract = "\n".join(
            (p.text or "").strip()
            for p in art.findall(".//AbstractText")
            if p.text
        ).strip()

        doi = ""
        for aid in pa.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()

        articles.append(
            {
                "title": title,
                "journal": journal,
                "year": year,
                "month": month,
                "abstract": abstract,
                "doi_url": f"https://doi.org/{doi}" if doi else "",
            }
        )

    return articles


# =================================================
# Heuristic score
# =================================================
def heuristic_score(a: Dict[str, Any]) -> int:
    journal = (a.get("journal") or "").lower()
    text = f"{a.get('title','')} {a.get('abstract','')}".lower()
    score = 0

    if "new england journal of medicine" in journal:
        score += 50
    elif journal.startswith("lancet"):
        score += 45
    elif journal == "jama":
        score += 42
    elif "jama neurology" in journal:
        score += 38
    elif "jama network open" in journal:
        score += 34
    elif journal == "stroke":
        score += 30
    elif journal == "neurology":
        score += 26
    else:
        score += 15

    if any(k in text for k in ["no benefit", "did not improve", "noninferior", "superior"]):
        score += 5

    return score


def prefilter_articles(articles: List[Dict[str, Any]], max_pool: int):
    return sorted(articles, key=heuristic_score, reverse=True)[:max_pool]


# =================================================
# LLM ranking (THIS WAS MISSING)
# =================================================
def rank_articles_llm(client: OpenAI, articles: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    payload = [
        f"{i} | {a['title']} | {a['journal']} | "
        f"{truncate_text(a['abstract'], RANK_ABSTRACT_MAX_CHARS)}"
        for i, a in enumerate(articles)
    ]

    chunks = chunk_list(payload, RANK_CHUNK_SIZE)
    ranked: List[Tuple[int, float]] = []

    progress = st.progress(0.0, text="Ranking articles by clinical relevance…")

    for ci, chunk in enumerate(chunks):
        resp = call_with_retry(
            lambda: client.responses.create(
                model=RANK_MODEL,
                input=[
                    {"role": "system", "content": "Score clinical relevance 0–100."},
                    {
                        "role": "user",
                        "content": "Return strictly: id | score\n\n" + "\n".join(chunk),
                    },
                ],
            )
        )

        for line in resp.output_text.splitlines():
            if "|" not in line:
                continue
            try:
                i, s = line.split("|", 1)
                ranked.append((int(i.strip()), float(s.strip())))
            except Exception:
                continue

        progress.progress((ci + 1) / len(chunks))

    progress.empty()
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =================================================
# Sidebar
# =================================================
with st.sidebar:
    st.subheader("OpenAI")
    st.text_input("OpenAI API key", type="password", key="openai_api_key")

    st.markdown("---")
    st.subheader("RSS source")

    rss_choice = st.selectbox("Choose PubMed RSS source", RSS_PRESETS.keys())
    if rss_choice == "Custom PubMed RSS link":
        rss_url = st.text_input("Paste custom RSS link")
    else:
        rss_url = RSS_PRESETS[rss_choice]

    st.markdown("---")
    top_k = st.number_input("Number of articles to display", 1, 50, 20)
    max_llm_rank_pool = st.number_input("MAX_LLM_RANK_POOL", 10, 200, 40, step=5)


# =================================================
# Main flow
# =================================================
if not rss_url:
    st.stop()

pmids = fetch_pmids_from_rss(rss_url)
articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles retrieved.")
    st.stop()

filtered = prefilter_articles(articles, int(max_llm_rank_pool))

ranking_signature = (rss_url, int(max_llm_rank_pool))
if "ranking_signature" not in st.session_state or st.session_state["ranking_signature"] != ranking_signature:
    client = get_openai_client()
    ranked = rank_articles_llm(client, filtered)
    st.session_state["ranked"] = ranked
    st.session_state["ranking_signature"] = ranking_signature
else:
    ranked = st.session_state["ranked"]

top_ranked = ranked[: min(int(top_k), len(ranked))]
top_articles = [(i, score, filtered[i]) for i, score in top_ranked]


# =================================================
# Render
# =================================================
st.markdown("## Ranked results")

for display_idx, (idx, score, a) in enumerate(top_articles, 1):
    st.markdown(f"<span style='color:#666'>{a['journal']} · {a['month']} {a['year']}</span>", unsafe_allow_html=True)
    title = f"[**{a['title']}**]({a['doi_url']})" if a["doi_url"] else f"**{a['title']}**"
    st.markdown(f"{display_idx}. {title}")
    st.caption(f"Relevance score: {score:.1f}/100")
    st.markdown("---")
