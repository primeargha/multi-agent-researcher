# Tools the agents can call: search the web, scrape a page.
from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re

# Load API keys from .env
load_dotenv()

# Tavily client for web search
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs, and Snippets of the search results."""
    # Ask Tavily for top 5 results
    results = tavily.search(query=query, max_results=5)

    out = []

    # Keep title, URL, and a short snippet for each hit
    for r in results["results"]:
        print(r)
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    # Join results with a separator so the agent can read them easily
    return "\n----\n".join(out)


@tool
def scrape_website(url: str) -> str:
    """Scrape a website for its main text content. Pass a URL from web_search results."""
    try:
        # 1st try: trafilatura (good at pulling main article text)
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
            )
            if text and text.strip():
                return _clean_and_trim(text)

        # 2nd try: download page ourselves, then clean with readability + BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # readability = keep the main article HTML; drop menus/ads when possible
        doc = Document(response.text)
        html = doc.summary()
        title = doc.title() or ""
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n", strip=True)

        # Last fallback: strip junk tags from the full page
        if not text.strip():
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        body = _clean_and_trim(text)
        return f"Title: {title}\n\n{body}" if title else body
    except Exception as e:
        # Don't crash the agent — return the error as text
        return f"Failed to scrape {url}: {e}"


def _clean_and_trim(text: str, max_chars: int = 5000) -> str:
    # Tidy whitespace and cap length so the LLM context stays small
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]"
    return text
