import json
import html
import re
from pathlib import Path
from html import unescape

OUTPUT_DIR = Path("data/raw_html")
TEXT_DIR = Path("data/broadcast_transcript")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_transcript(text: str) -> str:
    """
    Remove timestamps from a YouTube transcript while preserving
    the original line breaks.
    """

    lines = []

    for line in text.splitlines():

        # Remove HTML encoding first (&#x27; -> ')
        line = unescape(line)

        # Remove timestamps like 00:01:23 or 00:01:23.123
        line = re.sub(r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*', '', line)

        if line.startswith("#"):
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def save_transcript(
    transcript_file,
    title,
    news_source,
    political_leaning,
    news_type,
    published_date,
    topic,
    url,
    filename,
):
    transcript = Path(transcript_file).read_text(encoding="utf-8")

    transcript = clean_transcript(transcript)

    html_doc = f"""<!DOCTYPE html>
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
{transcript}
</div>

</body>
</html>
"""

    output_file = OUTPUT_DIR / f"{filename}.html"
    output_file.write_text(html_doc, encoding="utf-8")

    print(f"Saved: {output_file}")


def process_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for topic, articles in data.items():

        for article in articles:

            transcript_file = TEXT_DIR / f"{article['filename']}.txt"

            if not transcript_file.exists():
                print(f"Missing transcript: {transcript_file}")
                continue

            output_file = OUTPUT_DIR / f"{article['filename']}.html"

            if output_file.exists():
                print(f"Skipping existing: {output_file.name}")
                continue

            save_transcript(
                transcript_file=transcript_file,
                title=article["title"],
                news_source=article["news_source"],
                political_leaning=article["political_leaning"],
                news_type=article["news_type"],
                published_date=article["published_date"],
                topic=article.get("topic", topic),
                url=article["url"],
                filename=article["filename"],
            )


if __name__ == "__main__":
    process_json("transcripts.json")