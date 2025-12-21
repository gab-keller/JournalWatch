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
    "PubMed RSS → scalable ranking by immediate clinical relevance → summaries, structured abstracts, and podcast mode"
)

# =================================================
# Constants (defaults; user can override MAX_LLM_RANK_POOL in sidebar)
# =================================================
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RANK_CHUNK_SIZE = 10            # smaller chunks = lower TPM spikes
RANK_ABSTRACT_MAX_CHARS = 1200  # aggressive truncation for ranking calls

RANK_MODEL = "gpt-4.1"
ENRICH_MODEL = "gpt-4.1"
PODCAST_MODEL = "gpt-4.1"
TTS_MODEL = "gpt-4o-mini-tts"   # adjust if needed
TTS_VOICE = "alloy"             # adjust if needed

# Heuristic keywords for prefilter
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
# Stage 1 — deterministic prefilter (NO LLM)
# =================================================
def heuristic_score(a: Dict[str, Any]) -> int:
    text = f"{a.get('title','')} {a.get('abstract','')}".lower()
    score = 0

    # keywords
    for k in KEYWORDS:
        if k in text:
            score += 2

    # presence of abstract helps applicability
    if (a.get("abstract") or "").strip():
        score += 1

    # crude recency boost
    y = a.get("year", "")
    if y.isdigit():
        score += max(0, int(y) - 2016)

    return score


def prefilter_articles(articles: List[Dict[str, Any]], max_pool: int) -> List[Dict[str, Any]]:
    ranked = sorted(articles, key=lambda a: heuristic_score(a), reverse=True)
    return ranked[:max_pool]


# =================================================
# OpenAI helpers (robust retry + lightweight caching)
# =================================================
def get_openai_client() -> OpenAI:
    key = st.session_state.get("openai_api_key", "").strip()
    if not key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=key)


def call_with_retry(fn, retries: int = 6):
    """
    Exponential backoff for rate limits.
    """
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except RateLimitError as e:
            last_err = e
            # backoff: 0.8, 1.6, 3.2, 6.4, 12.8...
            time.sleep(max(0.8, 0.8 * (2 ** i)))
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err if last_err else RuntimeError("Unknown failure")


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def get_llm_cache() -> Dict[str, str]:
    if "llm_cache" not in st.session_state:
        st.session_state["llm_cache"] = {}
    return st.session_state["llm_cache"]


def cached_generate(cache_key: str, generator_fn) -> str:
    cache = get_llm_cache()
    if cache_key in cache:
        return cache[cache_key]
    out = generator_fn()
    cache[cache_key] = out
    return out


# =================================================
# Stage 2 — LLM ranking (reduced set only)
# =================================================
def rank_articles_llm(client: OpenAI, articles: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """
    Returns list of (index_in_articles, score), sorted desc.
    Plain-text output parsing to maximize SDK compatibility.
    Uses chunking + retries to avoid TPM crashes.
    """
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
                            "You are a senior clinical neurologist and journal editor. "
                            "Score articles by likelihood of IMMEDIATE impact on clinical practice.\n\n"
                            "Rules:\n"
                            "- Highest: RCTs with practice-changing results; guideline-informing evidence; large, high-quality cohorts.\n"
                            "- Intermediate: RCTs with NEGATIVE/neutral results that still inform decisions. "
                            "Negative RCTs MUST score ABOVE basic science, small case series, or purely hypothesis-generating work.\n"
                            "- Lowest: basic science, small case series, non-actionable hypothesis work.\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Score each line from 0 to 100.\n"
                            "Output STRICTLY one per line: id | score\n"
                            "No commentary.\n\n"
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
                idx = int(left.strip())
                score = float(right.strip())
                ranked.append((idx, score))
            except Exception:
                continue

        progress.progress((ci + 1) / max(1, len(chunks)))

    progress.empty()

    # Fill missing IDs with low score (defensive)
    seen = {i for i, _ in ranked}
    for i in range(len(articles)):
        if i not in seen:
            ranked.append((i, 0.0))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =================================================
# Enrichment — summary + structured abstract (Top-N only)
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
                            "Summarize the most important conclusion of this article and its clinical implications "
                            "in NO MORE than 2 sentences:\n\n"
                            f"{abstract}"
                        ),
                    },
                ],
            ).output_text.strip()
        ),
    )


def generate_structured_abstract(client: OpenAI, abstract: str) -> str:
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
                            "You are a medical editor improving readability of research abstracts for clinicians. "
                            "Return markdown. Use paragraphs and section headers when appropriate."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the abstract into structured paragraphs whenever appropriate, using these headers:\n"
                            "Background, Methods, Results, Conclusion.\n"
                            "If the abstract does not support a clean split, still use logical paragraphing.\n"
                            "Bold the most clinically important findings and key quantitative results.\n\n"
                            f"Abstract:\n{abstract}"
                        ),
                    },
                ],
            ).output_text.strip()
        ),
    )


# =================================================
# Podcast + TTS
# =================================================
def build_podcast_script(client: OpenAI, selected_articles: List[Dict[str, Any]]) -> str:
    # Keep the input manageable: title + journal/date + truncated abstracts
    blocks = []
    for a in selected_articles:
        header = f"{a.get('journal','')} · {a.get('month','')} {a.get('year','')}".strip(" ·")
        title = a.get("title", "")
        abs_short = truncate_text(a.get("abstract", ""), 1800)
        blocks.append(f"- {header}\n  {title}\n  Abstract: {abs_short}")

    payload = "\n\n".join(blocks)
    key = f"podcast::{_hash_text(payload)}"
    return cached_generate(
        key,
        lambda: call_with_retry(
            lambda: client.responses.create(
                model=PODCAST_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinician-host producing a practical medical podcast script. "
                            "Tone: clear, engaging, clinically oriented. Avoid hype. "
                            "Target length: ~10 minutes when read aloud."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Create a podcast-style script summarizing the selected articles below. "
                            "For each article: briefly introduce the topic, summarize the key result, and explain "
                            "the clinical implication. Use smooth transitions.\n\n"
                            "Selected articles:\n"
                            f"{payload}"
                        ),
                    },
                ],
            ).output_text.strip()
        ),
    )


def tts_generate_mp3(client: OpenAI, text: str) -> bytes:
    """
    Uses ChatGPT TTS if available in the installed SDK.
    Returns MP3 bytes.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("No podcast text to synthesize.")

    # Some SDK builds expose audio.speech.create; keep defensive.
    if not hasattr(client, "audio") or not hasattr(client.audio, "speech"):
        raise RuntimeError("This OpenAI SDK build does not expose audio.speech (TTS).")

    def _call():
        # Expected API shape in modern OpenAI SDKs:
        # client.audio.speech.create(model=..., voice=..., input=...)
        return client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            format="mp3",
        )

    resp = call_with_retry(_call)

    # Depending on SDK, resp may provide .read() or be bytes-like.
    if hasattr(resp, "read"):
        return resp.read()
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    if hasattr(resp, "content"):
        return resp.content
    raise RuntimeError("Unexpected TTS response type; cannot extract MP3 bytes.")


# =================================================
# Sidebar controls
# =================================================
with st.sidebar:
    st.subheader("OpenAI")
    st.text_input("OpenAI API key", type="password", key="openai_api_key", placeholder="sk-...")

    st.markdown("---")
    st.subheader("Ranking settings")
    top_k = st.number_input("Number of articles to display", min_value=1, max_value=50, value=20, step=1)

    max_llm_rank_pool = st.number_input(
        "MAX_LLM_RANK_POOL (articles sent to LLM for ranking)",
        min_value=10,
        max_value=200,
        value=40,
        step=5,
        help="Higher = potentially better ranking, but more cost/time and higher chance of rate limiting.",
    )

    st.caption(
        "Tip: if you use RSS limit=100, keeping MAX_LLM_RANK_POOL around 30–60 is usually stable."
    )


# =================================================
# Main: RSS input + fetch
# =================================================
rss_url = st.text_input("Paste PubMed RSS link", placeholder="https://pubmed.ncbi.nlm.nih.gov/rss/search/...")

if not rss_url:
    st.info("Paste a PubMed RSS link to continue.")
    st.stop()

with st.spinner("Fetching PMIDs from RSS…"):
    pmids = fetch_pmids_from_rss(rss_url)

with st.spinner("Fetching articles from PubMed (E-utilities)…"):
    articles = fetch_pubmed_articles(pmids)

if not articles:
    st.error("No articles retrieved.")
    st.stop()

st.success(f"{len(articles)} articles retrieved.")

# Prefilter
filtered = prefilter_articles(articles, int(max_llm_rank_pool))
st.info(f"{len(filtered)} articles selected for LLM ranking (heuristic pre-filter).")

client = get_openai_client()

# Rank
with st.spinner("Ranking articles by immediate clinical relevance…"):
    ranked = rank_articles_llm(client, filtered)

top_ranked = ranked[: min(int(top_k), len(ranked))]
top_articles = [(i, score, filtered[i]) for i, score in top_ranked]

st.markdown(f"## Ranked results (Top {len(top_articles)})")
st.caption(
    "Ranking prioritizes immediately actionable clinical evidence. Negative/neutral RCTs receive intermediate scores "
    "but remain above basic science and non-actionable studies."
)

# =================================================
# Render ranked articles + selection checkboxes
# =================================================
if "selected_ids" not in st.session_state:
    st.session_state["selected_ids"] = set()

for display_idx, (idx_in_filtered, score, a) in enumerate(top_articles, start=1):
    # Journal/date line above title, low emphasis
    journal_date = f"{a.get('journal','')} · {a.get('month','')} {a.get('year','')}".strip(" ·")
    st.markdown(f"<span style='color:#666'>{journal_date}</span>", unsafe_allow_html=True)

    # Title line: bold + clickable to DOI if available
    if a.get("doi_url"):
        st.markdown(f"{display_idx}. [**{a.get('title','')}**]({a.get('doi_url')})")
    else:
        st.markdown(f"{display_idx}. **{a.get('title','')}**")

    st.caption(f"Relevance score: {score:.1f}/100")

    # Checkbox for podcast selection
    cb_key = f"select_{idx_in_filtered}"
    default_checked = idx_in_filtered in st.session_state["selected_ids"]
    checked = st.checkbox("Select for podcast", value=default_checked, key=cb_key)
    if checked:
        st.session_state["selected_ids"].add(idx_in_filtered)
    else:
        st.session_state["selected_ids"].discard(idx_in_filtered)

    # 2-sentence clinical summary
    with st.spinner("Generating summary…"):
        summary = generate_summary_2sent(client, a.get("abstract", ""))
    st.markdown(summary)

    # Structured abstract in dropdown with bold highlights
    with st.expander("Abstract (formatted)"):
        with st.spinner("Formatting abstract…"):
            formatted_abs = generate_structured_abstract(client, a.get("abstract", ""))
        st.markdown(formatted_abs)

    st.markdown("---")


# =================================================
# Podcast generation section
# =================================================
st.markdown("## Podcast mode")

selected_ids_sorted = sorted(list(st.session_state["selected_ids"]))
selected_articles = [filtered[i] for i in selected_ids_sorted if 0 <= i < len(filtered)]

st.write(f"Selected articles: **{len(selected_articles)}**")

colA, colB = st.columns([1, 1])
with colA:
    gen_podcast = st.button("Generate podcast script (≈10 minutes)", type="primary", disabled=(len(selected_articles) == 0))

with colB:
    clear_sel = st.button("Clear selections", disabled=(len(selected_articles) == 0))
    if clear_sel:
        st.session_state["selected_ids"] = set()
        st.rerun()

if gen_podcast:
    with st.spinner("Generating podcast script…"):
        podcast_text = build_podcast_script(client, selected_articles)
    st.session_state["podcast_text"] = podcast_text

# Podcast text dropdown
if "podcast_text" in st.session_state and st.session_state["podcast_text"].strip():
    with st.expander("Podcast script (click to view)"):
        st.markdown(st.session_state["podcast_text"])

    # TTS generation
    st.markdown("### Audio version")

    voice = st.selectbox(
        "Voice",
        options=["alloy", "verse", "aria", "sage", "ember", "coral"],
        index=0,
        help="Available voices may vary by account/model. If one fails, try another.",
    )
    TTS_VOICE = voice  # override chosen voice

    gen_audio = st.button("Generate audio (MP3) via ChatGPT TTS")

    if gen_audio:
        try:
            with st.spinner("Synthesizing audio…"):
                mp3_bytes = tts_generate_mp3(client, st.session_state["podcast_text"])
            st.session_state["podcast_mp3"] = mp3_bytes
            st.success("Audio generated.")
        except Exception as e:
            st.error(f"TTS failed: {e}")

    if "podcast_mp3" in st.session_state and st.session_state["podcast_mp3"]:
        st.audio(st.session_state["podcast_mp3"], format="audio/mp3")
        st.download_button(
            "Download podcast MP3",
            data=st.session_state["podcast_mp3"],
            file_name="journalwatch_podcast.mp3",
            mime="audio/mpeg",
        )
else:
    st.info("Select at least one article above, then generate a podcast script to enable audio.")

