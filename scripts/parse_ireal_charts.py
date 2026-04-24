#!/usr/bin/env python3
"""Gold-tier ingestion: iReal Pro chord charts → harmonic sequences.

iReal Pro exports playlists as HTML files containing ``irealb://`` URLs.
Each URL encodes a chord chart with form structure, key, time signature,
and chord symbols.  This script parses those URLs and extracts harmonic
vocabulary suitable for training the candidate scorer's chord and
harmonic-role fields.

Usage
-----
    python scripts/parse_ireal_charts.py \\
        --input  data/raw/ireal/jazz_1410.html \\
        --output-dir data/gold/ireal/

Input formats
-------------
- HTML file exported from iReal Pro (contains ``irealb://`` URLs)
- Plain text file with one ``irealb://`` URL per line

Dependencies
------------
    pip install beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# iReal Pro URL decoder
# ---------------------------------------------------------------------------
# The irealb:// URL format encodes songs with the following structure:
#   irealb://<title>=<composer>=<style>=<key>=<n/a>=<chart_data>
# The chart data uses a custom encoding for bars, chords, and form markers.

# Chord quality patterns (simplified)
CHORD_QUALITIES = {
    "^7": "maj7", "^": "maj7", "6": "6", "69": "69",
    "-7": "min7", "-": "min", "-6": "min6", "-69": "min69",
    "-^7": "minmaj7", "-^": "minmaj7",
    "7": "dom7", "9": "dom9", "13": "dom13",
    "7b9": "dom7b9", "7#9": "dom7s9", "7b5": "dom7b5",
    "7#11": "dom7s11", "7alt": "alt",
    "o7": "dim7", "o": "dim", "h7": "hdim7", "h": "hdim",
    "+": "aug", "+7": "aug7",
    "sus": "sus4", "sus4": "sus4", "sus2": "sus2",
    "7sus": "dom7sus4", "7sus4": "dom7sus4",
    "": "maj",  # no quality = major triad
}

# Note names for mapping
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NOTE_TO_PITCH = {}
for i, n in enumerate(NOTE_NAMES):
    NOTE_TO_PITCH[n] = i
# Enharmonic equivalents
NOTE_TO_PITCH.update({
    "C#": 1, "D#": 3, "F#": 6, "G#": 8, "A#": 10,
    "Cb": 11, "Fb": 4, "E#": 5, "B#": 0,
})


@dataclass
class ChordEvent:
    """A single chord in the chart."""
    root: str
    quality: str
    root_pitch_class: int
    bar_index: int
    beat_index: int


@dataclass
class IrealSong:
    """Parsed iReal Pro song."""
    title: str
    composer: str
    style: str
    key: str
    time_signature: tuple[int, int]
    chords: list[ChordEvent] = field(default_factory=list)
    form_markers: list[dict] = field(default_factory=list)
    total_bars: int = 0


def _decode_ireal_url(url: str) -> Optional[dict]:
    """Decode an irealb:// URL into its component parts."""
    if not url.startswith("irealb://"):
        return None

    # Remove the protocol prefix
    payload = url[len("irealb://"):]
    # URL-decode
    payload = urllib.parse.unquote(payload)

    # Split by '=' separator
    parts = payload.split("=")
    if len(parts) < 6:
        return None

    return {
        "title": parts[0].strip(),
        "composer": parts[1].strip(),
        "style": parts[2].strip() if len(parts) > 2 else "",
        "key": parts[3].strip() if len(parts) > 3 else "C",
        "unused": parts[4] if len(parts) > 4 else "",
        "chart_data": parts[5] if len(parts) > 5 else "",
    }


def _deobfuscate_chart(chart_data: str) -> str:
    """Remove iReal Pro's obfuscation from chart data.

    iReal Pro applies a simple character-shuffling obfuscation to the
    chart string.  This reverses it.
    """
    # Remove leading/trailing whitespace
    s = chart_data.strip()

    # The obfuscation reverses groups of characters
    # First, handle the 50-character block reversal
    result = list(s)
    i = 0
    while i < len(result):
        block_size = min(50, len(result) - i)
        if i % 2 == 0 and block_size == 50:
            # Reverse this block
            result[i:i + block_size] = result[i:i + block_size][::-1]
        i += 50

    return "".join(result)


def _parse_chart_data(chart_data: str) -> tuple[list[ChordEvent], list[dict], int]:
    """Parse the chart data string into chord events and form markers.

    The chart format uses:
      - ``|`` for bar lines
      - ``[`` and ``]`` for section markers
      - ``{`` and ``}`` for repeat markers
      - ``T44``, ``T34`` etc. for time signatures
      - ``*A``, ``*B`` etc. for rehearsal marks
      - ``N1``, ``N2`` for endings
      - ``x`` for repeat bar
      - ``p`` for slash (beat without chord change)
      - Chord symbols like ``C^7``, ``Db-7``, ``G7``
    """
    chords: list[ChordEvent] = []
    form_markers: list[dict] = []
    bar_index = 0
    beat_index = 0
    beats_per_bar = 4  # default

    # Clean up the data
    data = chart_data.replace("LZ", "").replace("XyQ", "")

    # Simple state machine parser
    i = 0
    while i < len(data):
        ch = data[i]

        # Time signature
        if ch == "T" and i + 2 < len(data) and data[i + 1:i + 3].isdigit():
            beats_per_bar = int(data[i + 1])
            i += 3
            continue

        # Bar line
        if ch == "|" or ch == "[" or ch == "]":
            if ch == "|" and beat_index > 0:
                bar_index += 1
                beat_index = 0
            i += 1
            continue

        # Rehearsal mark
        if ch == "*" and i + 1 < len(data):
            marker = data[i + 1]
            form_markers.append({
                "type": "rehearsal",
                "label": marker,
                "bar": bar_index,
            })
            i += 2
            continue

        # Repeat / ending markers
        if ch in ("{", "}", "N"):
            i += 1
            if ch == "N" and i < len(data) and data[i].isdigit():
                i += 1
            continue

        # Slash (beat without chord change)
        if ch == "p" or ch == "/":
            beat_index += 1
            if beat_index >= beats_per_bar:
                bar_index += 1
                beat_index = 0
            i += 1
            continue

        # Repeat bar
        if ch == "x":
            bar_index += 1
            beat_index = 0
            i += 1
            continue

        # Spaces and other non-chord characters
        if ch in (" ", ",", "l", "f", "s", "n", "Q", "Y", "U"):
            i += 1
            continue

        # Try to parse a chord symbol
        chord_match = re.match(
            r"([A-G][b#]?)([\^o\+h\-]?[0-9a-z#]*(?:sus[24]?)?(?:alt)?)",
            data[i:],
        )
        if chord_match:
            root = chord_match.group(1)
            quality_str = chord_match.group(2)

            # Map quality string
            quality = CHORD_QUALITIES.get(quality_str, quality_str or "maj")
            root_pc = NOTE_TO_PITCH.get(root, 0)

            chords.append(ChordEvent(
                root=root,
                quality=quality,
                root_pitch_class=root_pc,
                bar_index=bar_index,
                beat_index=beat_index,
            ))

            beat_index += 1
            if beat_index >= beats_per_bar:
                bar_index += 1
                beat_index = 0

            i += chord_match.end()
            continue

        # Skip unrecognised characters
        i += 1

    total_bars = bar_index + (1 if beat_index > 0 else 0)
    return chords, form_markers, total_bars


def parse_ireal_song(url: str) -> Optional[IrealSong]:
    """Parse a single irealb:// URL into an IrealSong."""
    decoded = _decode_ireal_url(url)
    if decoded is None:
        return None

    chart_data = decoded["chart_data"]
    # Try deobfuscation
    chords, markers, total_bars = _parse_chart_data(chart_data)
    if not chords:
        # Try with deobfuscated data
        deobf = _deobfuscate_chart(chart_data)
        chords, markers, total_bars = _parse_chart_data(deobf)

    if not chords:
        log.warning("No chords found in: %s", decoded["title"])
        return None

    return IrealSong(
        title=decoded["title"],
        composer=decoded["composer"],
        style=decoded["style"],
        key=decoded["key"],
        time_signature=(4, 4),  # default; refined by parser
        chords=chords,
        form_markers=markers,
        total_bars=total_bars,
    )


# ---------------------------------------------------------------------------
# Extract URLs from HTML
# ---------------------------------------------------------------------------

def extract_urls_from_html(html_path: str | Path) -> list[str]:
    """Extract irealb:// URLs from an iReal Pro HTML export."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        sys.exit("beautifulsoup4 is required.  pip install beautifulsoup4")

    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    urls: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("irealb://"):
            urls.append(href)

    # Also search in plain text for URLs
    text = soup.get_text()
    for match in re.finditer(r"irealb://[^\s<>\"]+", text):
        url = match.group(0)
        if url not in urls:
            urls.append(url)

    return urls


def extract_urls_from_text(text_path: str | Path) -> list[str]:
    """Extract irealb:// URLs from a plain text file."""
    urls: list[str] = []
    with open(text_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("irealb://"):
                urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def song_to_dict(song: IrealSong) -> dict:
    """Convert an IrealSong to a JSON-serialisable dict."""
    return {
        "metadata": {
            "title": song.title,
            "composer": song.composer,
            "style": song.style,
            "key": song.key,
            "time_signature": list(song.time_signature),
            "total_bars": song.total_bars,
            "num_chords": len(song.chords),
            "tier": "gold",
            "source_format": "ireal_pro",
        },
        "chords": [asdict(c) for c in song.chords],
        "form_markers": song.form_markers,
        "harmonic_summary": {
            "unique_roots": sorted(set(c.root for c in song.chords)),
            "unique_qualities": sorted(set(c.quality for c in song.chords)),
            "chord_density_per_bar": round(
                len(song.chords) / max(song.total_bars, 1), 2
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse iReal Pro chord charts into harmonic sequences."
    )
    parser.add_argument(
        "--input", required=True, nargs="+",
        help="Input HTML or text file(s) containing irealb:// URLs",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write JSON output files",
    )
    parser.add_argument(
        "--filter-style", default=None,
        help="Only include songs matching this style (case-insensitive substring)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_urls: list[str] = []
    for input_path in args.input:
        p = Path(input_path)
        if p.suffix.lower() in (".html", ".htm"):
            all_urls.extend(extract_urls_from_html(p))
        else:
            all_urls.extend(extract_urls_from_text(p))

    log.info("Extracted %d irealb:// URLs from input files", len(all_urls))

    songs_written = 0
    for url in all_urls:
        song = parse_ireal_song(url)
        if song is None:
            continue

        # Optional style filter
        if args.filter_style:
            if args.filter_style.lower() not in song.style.lower():
                continue

        # Sanitise filename
        safe_title = re.sub(r"[^\w\s-]", "", song.title).strip()[:60]
        safe_title = re.sub(r"\s+", "_", safe_title)
        out_path = output_dir / f"{safe_title}.json"

        with open(out_path, "w") as f:
            json.dump(song_to_dict(song), f, indent=2)
        songs_written += 1

    log.info("Wrote %d song files to %s", songs_written, output_dir)

    # Write a summary index
    summary = {
        "total_urls": len(all_urls),
        "songs_written": songs_written,
        "style_filter": args.filter_style,
    }
    with open(output_dir / "index.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
