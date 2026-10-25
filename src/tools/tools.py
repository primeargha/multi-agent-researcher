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

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs, and Snippets of the search results."""
    results = tavily.search(query=query, max_results=5)

    # print(results)

    out = []

    for r in results['results']:
        print(r)
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    return "\n----\n".join(out)


@tool
def scrape_website(url: str) -> str:
    """Scrape a website for its main text content. Pass a URL from web_search results."""
    try:
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

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        doc = Document(response.text)
        html = doc.summary()
        title = doc.title() or ""
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n", strip=True)

        if not text.strip():
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        body = _clean_and_trim(text)
        return f"Title: {title}\n\n{body}" if title else body
    except Exception as e:
        return f"Failed to scrape {url}: {e}"


def _clean_and_trim(text: str, max_chars: int = 5000) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]"
    return text