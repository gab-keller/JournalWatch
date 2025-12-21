import re
import math
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="PubMed RSS – Clinical Digest", layout="wide")
st.title("PubMed RSS – Clinical Digest")
st.caption(
    "Paste a PubMed RSS link → rank by immediate clinical relevance → show top 20 with summaries + improved abstracts"
)

# -------------------------------------------------
# Constants / knobs
# -------------------------------------------------
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOP_K = 20

# Ranking safety knobs
RANK_CHUNK_SIZE = 15          # number of articles per LLM ranking call
RANK_ABSTRACT_MAX_CHARS = 1800  # truncate abstract sent for ranking

# Enrichment models (keep consistent for debugging)
RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"


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
        pmid = extract_pmid_from_link(getattr(entry, "link", "") or "")
        if pmid:
            pmids.append(pmid)
    # de-duplicate preserving order
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
    articles: List[Dict[str, Any]] = []

    for pubmed_article in root.findall(".//PubmedArticle"):
        medline = pubmed_article.find("MedlineCitation")
        if medline is None:
            continue
        article = medline.find("Article")
        if article is None:
            continue

        title = article.findtext("ArticleTitle", default="").strip()
        journal = article.findtext("Journal/Title", default="").strip()
        year = article.findtext("Journal/JournalIssue/PubDate/Year", default="").strip()
        month = article.findtext("Journal/JournalIssue/PubDate/Month", default="").strip()

        abstract_parts = article.findall(".//AbstractText")
        abstract = "\n".join(
            (part.text or "").strip() for part in abstract_parts if (part.text or "").strip()
        ).strip()

        # Some PubMed entries have no abstract; we can still rank by title/journal/date
        # but enrichment will be limited. Keep them; the ranker can decide.
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


def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# -------------------------------------------------
# OpenAI helpers
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key", "").strip()
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)


def rank_articles_by_relevance(
    client: OpenAI,
    articles: List[Dict[str, Any]],
) -> List[Tuple[int, float]]:
    """
    Returns list of (index_in_articles, score) where higher is more clinically actionable.
    Uses chunked calls and a schema-enforced parse to avoid JSON formatting issues.
    """

    # Schema for one ranking batch
    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "score": {"type": "number"},
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["scores"],
    }

    # Build “ranking payload” with truncated abstracts to keep tokens manageable
    rank_payload = []
    for idx, a in enumerate(articles):
        rank_payload.append(
            {
                "id": idx,
                "title": a.get("title", ""),
                "journal": a.get("journal", ""),
                "date": f"{a.get('month','')} {a.get('year','')}".strip(),
                "abstract": truncate_text(a.get("abstract", ""), RANK_ABSTRACT_MAX_CHARS),
            }
        )

    ranked: List[Tuple[int, float]] = []

    batches = chunk_list(rank_payload, RANK_CHUNK_SIZE)
    progress = st.progress(0.0, text="Ranking articles by clinical relevance…")

    for bi, batch in enumerate(batches):
        # We ask for a 0–100 score. The model is forced to output schema.
        response = client.responses.parse(
            model=RANK_MODEL,
            schema=schema,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior clinical neurologist and journal editor. "
                        "Score each article by likelihood of IMMEDIATE clinical impact (0–100). "
                        "Prioritize: randomized clinical trials, guideline-informing evidence, large high-quality cohorts, "
                        "practice-changing diagnostics/therapeutics, and results that would change management now. "
                        "Lower score: basic science, small case series, hypothesis-generating work without actionable steps."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return a score for each item. "
                        "Use only the provided information. "
                        "Score range: 0 to 100 (higher = more immediately actionable)."
                    ),
                },
                {"role": "user", "content": str(batch)},
            ],
        )

        parsed = response.output_parsed
        for item in parsed.get("scores", []):
            ranked.append((int(item["id"]), float(item["score"])))

        progress.progress((bi + 1) / max(1, len(batches)), text="Ranking articles by clinical relevance…")

    progress.empty()

    # If any IDs missing (rare), assign neutral low score
    got = {i for i, _ in ranked}
    for i in range(len(articles)):
        if i not in got:
            ra
