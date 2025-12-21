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
            ranked.append((i, 0.0))

    # Sort by score desc
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_clinical_summary(client: OpenAI, abstract: str) -> str:
    if not abstract.strip():
        return "No abstract available to summarize."
    response = client.responses.create(
        model=ENRICH_MODEL,
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
                    "Summarize the most important conclusion of this article and its clinical implications "
                    "in no more than 2 sentences:\n\n"
                    f"{abstract}"
                ),
            },
        ],
    )
    return response.output_text.strip()


def generate_structured_abstract(client: OpenAI, abstract: str) -> str:
    if not abstract.strip():
        return "No abstract available."
    response = client.responses.create(
        model=ENRICH_MODEL,
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
                    "Rewrite the abstract below into structured paragraphs (Background, Methods, Results, Conclusion) "
                    "WHEN appropriate. If the abstract does not support a clean split, keep logical paragraphs anyway. "
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
    st.markdown("---")
    st.caption("This step ranks articles (chunked) and then enriches only the top 20.")


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
# Fetch from PubMed
# -------------------------------------------------
with st.spinner("Fetching articles from PubMed (RSS → PMIDs → E-utilities)…"):
    try:
        pmids = fetch_pmids_from_rss(rss_url)
        articles = fetch_pubmed_articles(pmids)
    except Exception as e:
        st.error(f"Error fetching PubMed articles: {e}")
        st.stop()

if not articles:
    st.error("No articles retrieved.")
    st.stop()

st.success(f"Retrieved {len(articles)} articles from PubMed.")


# -------------------------------------------------
# Rank and keep top 20
# -------------------------------------------------
client = get_openai_client()

with st.spinner("Ranking articles by immediate clinical relevance…"):
    ranked = rank_articles_by_relevance(client, articles)

top_ranked = ranked[: min(TOP_K, len(ranked))]
top_articles = [(i, score, articles[i]) for i, score in top_ranked]

st.markdown(f"## Top {len(top_articles)} most clinically relevant articles")


# -------------------------------------------------
# Display top 20 with enrichment
# -------------------------------------------------
for display_idx, (orig_idx, score, article) in enumerate(top_articles, start=1):
    header = f"{article.get('journal','')}, {article.get('month','')} {article.get('year','')} — {article.get('title','')}"
    header = re.sub(r"\s+", " ", header).strip(" ,—")

    # Smaller text (same size as title): use bold inline text, not markdown headings
    st.markdown(f"**{display_idx}. {header}**")
    st.caption(f"Relevance score: {score:.1f}/100")

    # Summary immediately under title
    with st.spinner("Generating clinical summary…"):
        summary = generate_clinical_summary(client, article.get("abstract", ""))
    st.markdown(summary)

    # Structured abstract with bold highlights
    with st.expander("Abstract"):
        with st.spinner("Formatting abstract…"):
            structured = generate_structured_abstract(client, article.get("abstract", ""))
        st.markdown(structured)
