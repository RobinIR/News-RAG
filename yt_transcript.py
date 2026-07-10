from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    parsed = urlparse(url)

    if parsed.netloc == "youtu.be":
        return parsed.path.lstrip("/")

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]

    raise ValueError("Invalid YouTube URL")


# Your YouTube URL
url = "https://www.youtube.com/watch?v=7FQzbQNj10M"

video_id = extract_video_id(url)

ytt_api = YouTubeTranscriptApi()

try:
    transcript = ytt_api.fetch(video_id)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        for snippet in transcript:
            f.write(snippet.text + "\n")

    print("Transcript saved to transcript.txt")

except Exception as e:
    print(f"Error: {e}")