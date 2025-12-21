import re
import requests
import feedparser
import streamlit as st
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="PubMed RSS Tester", layout="wide")

st.title("PubMed RSS → Article Extractor")
st.caption("Paste a PubMed RSS link and view the extracted article metadata")


# -------------------------------------------------
# Constants
# -------------------------------------------------
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_pmid_from_link(link: str) -> Optional[str]:
    """
    Example link:
    https://pubmed.ncbi.nlm.nih.gov/12345678/
    """
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", link)
    return match.group(1) if match else None


def fetch_pmids_from_rss(rss_url: str) -> List[str]:
    feed = feedparser.parse(rss_url)
    pmids = []

    for entry in feed.entries:
        pmid = extract_pmid_from_link(entry.link)
        if pmid:
            pmids.append(pmid)

    # De-duplicate while preserving order
    return list(dict.fromkeys(pmids))


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

        doi = ""
        for article_id in pubmed_article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text.strip()

        articles.append(
            {
                "Title": title,
                "Journal": journal,
                "Year": year,
                "Month": month,
                "DOI": doi,
                "Abstract": abstract,
            }
        )

    return articles


# -------------------------------------------------
# UI
# -------------------------------------------------
rss_url = st.text_input(
    "Paste PubMed RSS link",
    placeholder="https://pubmed.ncbi.nlm.nih.gov/rss/search/...",
)

if not rss_url:
    st.info("Paste a PubMed RSS link to begin.")
    st.stop()


# -------------------------------------------------
# Processing
# -------------------------------------------------
with st.spinner("Parsing RSS feed and fetching articles from PubMed..."):
    try:
        pmids = fetch_pmids_from_rss(rss_url)
        articles = fetch_pubmed_articles(pmids)
    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        st.stop()


# -------------------------------------------------
# Output
# -------------------------------------------------
st.success(f"Retrieved {len(articles)} articles")

for idx, article in enumerate(articles, start=1):
    st.markdown(f"### {idx}. {article['Title']}")
    st.markdown(f"**Journal:** {article['Journal']}")
    st.markdown(f"**Publication date:** {article['Month']} {article['Year']}")
    st.markdown(f"**DOI:** {article['DOI'] or '—'}")

    with st.expander("Abstract"):
        st.write(article["Abstract"] or "No abstract available")
