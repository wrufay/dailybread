"""
Download pre-built KJV Bible and convert to verses.json
Uses public domain dataset from GitHub - takes seconds instead of hours
"""

import json
import requests
from pathlib import Path

# Public domain KJV JSON hosted on GitHub
BIBLE_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

def download_bible() -> list[dict]:
    """Download pre-built KJV Bible JSON."""
    print("Downloading KJV Bible...")
    response = requests.get(BIBLE_URL, timeout=30)
    response.raise_for_status()
    # Handle UTF-8 BOM
    content = response.content.decode('utf-8-sig')
    return json.loads(content)

def convert_to_verses(bible_data: list[dict]) -> list[dict]:
    """Convert Bible JSON to flat verse list."""
    verses = []

    for book in bible_data:
        book_name = book["name"]
        chapters = book["chapters"]

        for chapter_idx, chapter_verses in enumerate(chapters, start=1):
            for verse_idx, verse_text in enumerate(chapter_verses, start=1):
                verses.append({
                    "book": book_name,
                    "chapter": chapter_idx,
                    "verse": verse_idx,
                    "text": verse_text.strip(),
                    "reference": f"{book_name} {chapter_idx}:{verse_idx}"
                })

    return verses

def main():
    print("=" * 50)

    bible_data = download_bible()
    print(f"Downloaded {len(bible_data)} books")

    verses = convert_to_verses(bible_data)
    print(f"Extracted {len(verses)} verses")

    output_path = Path(__file__).parent / "verses.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    print("=" * 50)
    print("Done! Now run: python embed.py")

if __name__ == "__main__":
    main()
