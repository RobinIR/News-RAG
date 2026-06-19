from pathlib import Path
from urllib.parse import urlparse
import html
from trafilatura import fetch_url, extract


OUTPUT_DIR = Path("data/raw_html")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_article(
    url,
    title,
    news_source,
    political_leaning,
    news_type,
    published_date="Unknown",
    topic="Unknown",
    filename="Unknown",
):

    downloaded = fetch_url(url)

    if not downloaded:
        print(f"Failed to download: {url}")
        return

    # Extract text
    text = extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        include_links=False
    )

    if not text:
        print(f"No content extracted: {url}")
        return


    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
</head>
<body>


<br/>

Title: {title}<br/>
News Source: {news_source}<br/>
Political Leaning: {political_leaning}<br/>
News Type: {news_type}<br/>
Published Date: {published_date}<br/>
Source Link: {url}<br/>
Topic: {topic}<br/>

<br/>

{text}

</div>

</body>
</html>
"""

    filename = f"{filename}.html"

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Saved: {filepath}")


# Fill in the details for the article you want to save.
save_article(
    url="www.forbes.com/sites/mikestunson/2026/01/09/majority-of-americans-disapprove-of-how-ice-operates-new-survey-shows/",
    title="Majority Of Americans Disapprove Of How ICE Operates, New Survey Shows",
    news_source="Forbes",
    political_leaning="Center",
    news_type="News",
    published_date="09 January 2026",
    topic="Deportation Campaign",
    filename="DC4__Center__News",
)