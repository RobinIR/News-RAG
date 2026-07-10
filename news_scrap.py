from pathlib import Path
from urllib.parse import urlparse
import html
import json
from bs4 import BeautifulSoup
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

Title: {title}
News Source: {news_source}
Political Leaning: {political_leaning}
News Type: {news_type}
Published Date: {published_date}
Source Link: {url}
Topic: {topic}

<br/>
<div>

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
# save_article(
#     url="www.forbes.com/sites/mikestunson/2026/01/09/majority-of-americans-disapprove-of-how-ice-operates-new-survey-shows/",
#     title="Majority Of Americans Disapprove Of How ICE Operates, New Survey Shows",
#     news_source="Forbes",
#     political_leaning="Center",
#     news_type="News",
#     published_date="09 January 2026",
#     topic="Deportation Campaign",
#     filename="DC4__Center__News",
# )

def process_json(json_file):
    """
    Reads all articles from the JSON file and downloads them.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    saved = 0
    skipped = 0
    failed = 0

    for topic, articles in data.items():

        print(f"\nProcessing topic: {topic}")

        for article in articles:

            total += 1

            output_file = OUTPUT_DIR / f"{article['filename']}.html"

            if output_file.exists():
                skipped += 1
                print(f"[SKIP] {article['filename']}")
                continue

            try:
                save_article(
                    url=article["url"],
                    title=article["title"],
                    news_source=article["news_source"],
                    political_leaning=article["political_leaning"],
                    news_type=article["news_type"],
                    published_date=article.get("published_date", "Unknown"),
                    topic=article.get("topic", topic),
                    filename=article["filename"],
                )
                saved += 1

            except Exception as e:
                failed += 1
                print(f"[ERROR] {article['filename']}: {e}")

    print("\n========== SUMMARY ==========")
    print(f"Total   : {total}")
    print(f"Saved   : {saved}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    process_json("dataset.json")