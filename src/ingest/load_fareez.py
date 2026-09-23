import json
import re
from pathlib import Path

TRANSCRIPT_DIR = Path("/Users/rashi/Downloads/Data/Clean Transcripts")
AUDIO_DIR = Path("/Users/rashi/Downloads/Data/Audio Recordings")
TURNS_PATH = Path("data/processed/turns.jsonl")

TURN_PATTERN = re.compile(r"^([DP])\s*[:;]\s*(.*)$", re.IGNORECASE)
SPEAKER_MAP = {"D": "doctor", "P": "patient"}


def read_transcript(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16")


def parse_transcript(text):
    turns = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = TURN_PATTERN.match(line)
        if m:
            speaker, turn_text = m.groups()
            turn_text = turn_text.strip()
            if turn_text:
                turns.append({"speaker": SPEAKER_MAP[speaker.upper()], "text": turn_text})
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


def main():
    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    audio_stems = {p.stem for p in AUDIO_DIR.glob("*.mp3")}

    new_records = []
    for path in transcript_files:
        stem = path.stem
        if stem not in audio_stems:
            print(f"Skipping {stem}: no matching audio file")
            continue
        turns = parse_transcript(read_transcript(path))
        for i, turn in enumerate(turns):
            new_records.append({
                "split": "test",
                "dataset": "fareez",
                "encounter_id": stem,
                "turn_idx": i,
                "speaker": turn["speaker"],
                "text": turn["text"],
            })

    existing_records = []
    if TURNS_PATH.exists():
        with open(TURNS_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec["dataset"] != "fareez":
                    existing_records.append(rec)

    all_records = existing_records + new_records
    with open(TURNS_PATH, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Parsed {len(new_records)} turns across {len(transcript_files)} fareez encounters")
    print(f"turns.jsonl now has {len(all_records)} total rows "
          f"({len(existing_records)} pre-existing + {len(new_records)} fareez)")


if __name__ == "__main__":
    main()
