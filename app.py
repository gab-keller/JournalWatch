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
# Constants
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RANK_CHUNK_SIZE = 10
RANK_ABSTRACT_MAX_CHARS = 1200

RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"
PODCAST_MODEL = "gpt-4.1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

KEYWORDS = [
    "randomized", "randomised", "trial", "phase",
    "guideline", "meta-analysis", "systematic review",
    "multicenter", "double-blind", "placebo"
]


# =================================================
# Helpers (RSS / PubMed)
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


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# =================================================
# Heuristic score (journal-centric, RCT-aware)
# =================================================
def heuristic_score(a: Dict[str, Any]) -> int:
    score = 0
    journal = (a.get("journal") or "").lower()
    text = f"{a.get('title','')} {a.get('abstract','')}".lower()

    # Journal prestige
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

    # Phase / scale signals
    if "phase 3" in text or "phase iii" in text:
        score += 12
    elif "phase 2" in text or "phase ii" in text:
        score += 8

    if "multicenter" in text or "multi-center" in text:
        score += 6
    if "international" in text:
        score += 4
    if "pragmatic" in text:
        score += 4

    # Hard outcomes vs surrogates
    HARD_OUTCOMES = [
        "mortality", "death", "functional outcome", "disability",
        "modified rankin", "stroke recurrence", "hospitalization",
        "major bleeding",
    ]
    SURROGATES = [
        "imaging", "biomarker", "surrogate",
        "feasibility", "pharmacokinetic",
    ]

    for kw in HARD_OUTCOMES:
        if kw in text:
            score += 3
    for kw in SURROGATES:
        if kw in text:
            score -= 3

    # Decision-changing language (positive or negative RCTs)
    DECISION_TERMS = [
        "did not improve", "no benefit", "no significant difference",
        "superior", "noninferior", "non-inferior",
        "reduced risk", "head-to-head", "first",
        "questions current practice", "supports",
    ]
    for kw in DECISION_TERMS:
        if kw in text:
            score += 5

    return score


def prefilter_articles(articles: List[Dict[str, Any]], max_pool: int) -> List[Dict[str, Any]]:
    ranked = sorted(articles, key=lambda a: heuristic_score(a), reverse=True)
    return ranked[:max_pool]


# =================================================
# LLM ranking (session-state guarded)
# =================================================
def rank_articles_llm(client: OpenAI, articles: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    payload_lines = []
    for i, a in enumerate(articles):
        payload_lines.append(
            f"{i} | {a.get('title','')} | {a.get('journal','')} | "
            f"{truncate_text(a.get('abstract',''), RANK_ABSTRACT_MAX_CHARS)}"
        )

    chunks = chunk_list(payload_lines, RANK_CHUNK_SIZE)
    ranked: List[Tuple[int, float]] = []

    progress = st.progress(0.0, text="Ranking articles by clinical relevance…")

    for ci, chunk in enumerate(chunks):
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
                            "Score each line from 0 to 100.\n"
                            "STRICT FORMAT: id | score\n\n"
                            + "\n".join(chunk)
                        ),
                    },
                ],
            )

        resp = call_with_retry(_call)
        for line in resp.output_text.splitlines():
            if "|" not in line:
                continue
            try:
                left, right = line.split("|", 1)
                ranked.append((int(left.strip()), float(right.strip())))
            except Exception:
                continue

        progress.progress((ci + 1) / max(1, len(chunks)))

    progress.empty()
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =================================================
# Enrichment
# =================================================
def generate_summary_2sent(client: OpenAI, abstract: str) -> str:
    abstract = (abstract or "").strip()
    if not abstract:
        return "No abstract available to summarize."

    key = f"summary::{_hash_text(abstract)}"
    return cached_generate(
        key,
        lambda: call_with_retry(
            lambda: client.responses.create(
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
                            "clinical implications in NO MORE than 2 sentences:\n\n"
                            f"{abstract}"
                        ),
                    },
                ],
            ).output_text.strip()
        ),
    )


def generate_structured_abstract(client: OpenAI, abstract: str) -> str:
    """
    IMPORTANT UI REQUIREMENT:
    - Do NOT use markdown headers (#, ##, ###)
    - Use bold labels only, each on its own line:
      **Background**, **Methods**, **Results**, **Conclusion**
    """
    abstract = (abstract or "").strip()
    if not abstract:
        return "No abstract available."

    key = f"structabs::{_hash_text(abstract)}"
    return cached_generate(
        key,
        lambda: call_with_retry(
            lambda: client.responses.create(
                model=ENRICH_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical editor improving readability of abstracts for clinicians.\n"
                            "CRITICAL FORMATTING RULES:\n"
                            "- DO NOT use markdown headers (#, ##, ###).\n"
                            "- Use ONLY bold subsection labels on separate lines.\n"
                            "- Example:\n"
                            "**Background**\n"
                            "Text...\n\n"
                            "**Methods**\n"
                            "Text...\n\n"
                            "Bold the most clinically important findings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the abstract into structured paragraphs when appropriate.\n"
                            "Use these labels ONLY if appropriate:\n"
                            "Background, Methods, Results, Conclusion.\n\n"
                            f"Abstract:\n{abstract}"
                        ),
                    },
                ],
            ).output_text.strip()
        ),
    )


# =================================================
# Sidebar (RSS selector updated)
# =================================================
with st.sidebar:
    st.subheader("OpenAI")
    st.text_input("OpenAI API key", type="password", key="openai_api_key")

    st.markdown("---")
    st.subheader("RSS source")

    rss_choice = st.selectbox(
        "Choose PubMed RSS source",
        options=list(RSS_PRESETS.keys()),
        index=0,
    )

    if rss_choice == "Custom PubMed RSS link":
        rss_url = st.text_input(
            "Paste custom PubMed RSS link",
            placeholder="https://pubmed.ncbi.nlm.nih.gov/rss/search/...",
        )
    else:
        rss_url = RSS_PRESETS[rss_choice]

    st.markdown("---")
    st.subheader("Ranking settings")
    top_k = st.number_input("Number of articles to display", 1, 50, 20)
    max_llm_rank_pool = st.number_input("MAX_LLM_RANK_POOL", 10, 200, 40, step=5)


# =================================================
# Main flow
# =================================================
rss_url = st.text_input("Paste PubMed RSS link")
if not rss_url:
    st.stop()

with st.spinner("Fetching articles from PubMed…"):
    pmids = fetch_pmids_from_rss(rss_url)
    articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles retrieved.")
    st.stop()

st.success(f"{len(articles)} articles retrieved.")

filtered = prefilter_articles(articles, int(max_llm_rank_pool))
st.info(f"{len(filtered)} articles selected for LLM ranking.")

ranking_signature = (rss_url, int(max_llm_rank_pool))

if (
    "ranking_signature" not in st.session_state
    or st.session_state["ranking_signature"] != ranking_signature
):
    client = get_openai_client()
    with st.spinner("Ranking articles by immediate clinical relevance…"):
        ranked = rank_articles_llm(client, filtered)
    st.session_state["ranked"] = ranked
    st.session_state["ranking_signature"] = ranking_signature
else:
    ranked = st.session_state["ranked"]

top_ranked = ranked[: min(int(top_k), len(ranked))]
top_articles = [(i, score, filtered[i]) for i, score in top_ranked]


# =================================================
# Render results + podcast selection
# =================================================
if "selected_ids" not in st.session_state:
    st.session_state["selected_ids"] = set()

st.markdown("## Ranked results")

for display_idx, (idx_f, score, a) in enumerate(top_articles, start=1):
    journal_date = f"{a.get('journal','')} · {a.get('month','')} {a.get('year','')}".strip(" ·")
    st.markdown(f"<span style='color:#666'>{journal_date}</span>", unsafe_allow_html=True)

    if a.get("doi_url"):
        st.markdown(f"{display_idx}. [**{a.get('title','')}**]({a.get('doi_url')})")
    else:
        st.markdown(f"{display_idx}. **{a.get('title','')}**")

    st.caption(f"Relevance score: {score:.1f}/100")

    checked = st.checkbox(
        "Select for podcast",
        key=f"select_{idx_f}",
        value=(idx_f in st.session_state["selected_ids"]),
    )

    if checked:
        st.session_state["selected_ids"].add(idx_f)
    else:
        st.session_state["selected_ids"].discard(idx_f)

    with st.spinner("Generating summary…"):
        summary = generate_summary_2sent(get_openai_client(), a.get("abstract", ""))
    st.markdown(summary)

    with st.expander("Abstract"):
        with st.spinner("Formatting abstract…"):
            formatted = generate_structured_abstract(get_openai_client(), a.get("abstract", ""))
        st.markdown(formatted)

    st.markdown("---")


# =================================================
# Podcast generation
# =================================================
st.markdown("## Podcast mode")

selected_articles = [filtered[i] for i in sorted(st.session_state["selected_ids"]) if i < len(filtered)]
st.write(f"Selected articles: **{len(selected_articles)}**")

if st.button("Generate podcast script (≈10 minutes)", disabled=(len(selected_articles) == 0)):
    with st.spinner("Generating podcast script…"):
        podcast_text = cached_generate(
            f"podcast::{_hash_text(str([a['title'] for a in selected_articles]))}",
            lambda: call_with_retry(
                lambda: get_openai_client().responses.create(
                    model=PODCAST_MODEL,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a clinician-host producing a practical medical podcast. "
                                "Target length ~10 minutes. Clear, didactic, clinically focused."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Create a podcast-style script summarizing these articles, "
                                "highlighting key results and clinical implications:\n\n"
                                + "\n\n".join(
                                    f"{a['title']}\n{truncate_text(a.get('abstract',''), 1500)}"
                                    for a in selected_articles
                                )
                            ),
                        },
                    ],
                ).output_text.strip()
            ),
        )
    st.session_state["podcast_text"] = podcast_text

if "podcast_text" in st.session_state:
    with st.expander("Podcast script"):
        st.markdown(st.session_state["podcast_text"])
