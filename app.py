import re
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI


# =================================================
# Page setup
# =================================================
st.set_page_config(page_title="PubMed RSS – Clinical Digest", layout="wide")
st.title("PubMed RSS – Clinical Digest")
st.caption(
    "Paste a PubMed RSS link → rank by immediate clinical relevance → "
    "display the most relevant articles with concise summaries"
)


# =================================================
# Constants / knobs
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Ranking safety
RANK_CHUNK_SIZE = 15            # articles per ranking call
RANK_ABSTRACT_MAX_CHARS = 1800  # truncate abstracts for ranking

RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"


# =================================================
# Helpers
# =================================================
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
    return list(dict.fromkeys(pmids))  # de-duplicate, preserve order


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
            (part.text or "").strip()
            for part in abstract_parts
            if (part.text or "").strip()
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


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# =================================================
# OpenAI helpers
# =================================================
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
    Returns list of (index_in_articles, score) sorted by score desc.
    Uses plain-text scoring for maximum SDK compatibility.
    """

    ranked: List[Tuple[int, float]] = []

    payload = []
    for idx, a in enumerate(articles):
        payload.append(
            {
                "id": idx,
                "title": a.get("title", ""),
                "journal": a.get("journal", ""),
                "date": f"{a.get('month','')} {a.get('year','')}".strip(),
                "abstract": truncate_text(a.get("abstract", ""), RANK_ABSTRACT_MAX_CHARS),
            }
        )

    batches = chunk_list(payload, RANK_CHUNK_SIZE)
    progress = st.progress(0.0, text="Ranking articles by clinical relevance…")

    for bi, batch in enumerate(batches):
        response = client.responses.create(
            model=RANK_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior clinical neurologist and journal editor. "
                        "Score each article by likelihood of IMMEDIATE impact on clinical practice."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Score each article below from 0 to 100 based on immediate clinical relevance.\n\n"
                        "Scoring guidance:\n"
                        "- Highest scores: randomized clinical trials with positive or practice-changing results, "
                        "guideline-informing evidence, or large high-quality cohorts.\n"
                        "- Intermediate scores: randomized clinical trials with NEGATIVE or neutral results that "
                        "still meaningfully inform clinical decision-making.\n"
                        "- Lower scores: observational studies with limited applicability.\n"
                        "- Lowest scores: basic science, small case series, hypothesis-generating work without "
                        "clear actionable clinical implications.\n\n"
                        "IMPORTANT:\n"
                        "- Negative RCTs MUST score higher than basic science, small case series, or purely "
                        "hypothesis-generating studies.\n\n"
                        "Output format (STRICT):\n"
                        "id | score\n"
                        "One article per line.\n"
                        "No explanations.\n\n"
                        f"{batch}"
                    ),
                },
            ],
        )

        lines = response.output_text.strip().splitlines()
        for line in lines:
            if "|" not in line:
                continue
            try:
                left, right = line.split("|", 1)
                idx = int(left.strip())
                score = float(right.strip())
                ranked.append((idx, score))
            except Exception:
                continue

        progress.progress((bi + 1) / max(1, len(batches)))

    progress.empty()

    seen = {i for i, _ in ranked}
    for i in range(len(articles)):
        if i not in seen:
            ranked.append((i, 0.0))

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
                    "Summarize the most important conclusion of this article and its "
                    "clinical implications in no more than 2 sentences:\n\n"
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
                    "Rewrite the abstract below into structured paragraphs "
                    "(Background, Methods, Results, Conclusion) when appropriate. "
                    "Bold the most clinically important findings.\n\n"
                    f"{abstract}"
                ),
            },
        ],
    )
    return response.output_text.strip()


# =================================================
# Sidebar
# =================================================
with st.sidebar:
    st.subheader("OpenAI API Key")
    st.text_input(
        "Enter your OpenAI API key",
        type="password",
        key="openai_api_key",
        placeholder="sk-...",
    )

    st.markdown("---")
    st.subheader("Ranking settings")

    top_k = st.number_input(
        "Number of top articles to display",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
        help="How many of the most clinically relevant articles should be ranked and displayed.",
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
with st.spinner("Fetching articles from PubMed…"):
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


# =================================================
# Rank + select Top N
# =================================================
client = get_openai_client()

with st.spinner("Ranking articles by immediate clinical relevance…"):
    ranked = rank_articles_by_relevance(client, articles)

top_ranked = ranked[: min(top_k, len(ranked))]
top_articles = [(i, score, articles[i]) for i, score in top_ranked]

st.markdown(f"## Top {len(top_articles)} most clinically relevant articles")
st.caption(
    "Ranking prioritizes immediately actionable clinical evidence. "
    "Negative RCTs are ranked above basic science and non-actionable studies."
)


# =================================================
# Display results
# =================================================
for display_idx, (orig_idx, score, article) in enumerate(top_articles, start=1):
    header = f"{article.get('journal','')}, {article.get('month','')} {article.get('year','')} — {article.get('title','')}"
    header = re.sub(r"\s+", " ", header).strip(" ,—")

    st.markdown(f"**{display_idx}. {header}**")
    st.caption(f"Relevance score: {score:.1f}/100")

    with st.spinner("Generating clinical summary…"):
        summary = generate_clinical_summary(client, article.get("abstract", ""))
    st.markdown(summary)

    with st.expander("Abstract"):
        with st.spinner("Formatting abstract…"):
            structured = generate_structured_abstract(client, article.get("abstract", ""))
        st.markdown(structured)
