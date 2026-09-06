# Runs on Google Colab with a GPU runtime (Runtime > Change runtime type > T4 GPU).
# Requires encounter_texts.jsonl uploaded to the Colab session and Google Drive mounted.
#
# Cell 1:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Cell 2:
#   from google.colab import files
#   uploaded = files.upload()   # select encounter_texts.jsonl
#
# Cell 3:
#   !pip install -q transformers accelerate soundfile openai-whisper
#
# Cell 4 (this script):

import json, re
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
import whisper

OUT_DIR = Path("/content/drive/MyDrive/riskaware-asr")
AUDIO_DIR = OUT_DIR / "audio"
WHISPER_DIR = OUT_DIR / "whisper_out"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
WHISPER_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading TTS model (facebook/mms-tts-eng)...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
SAMPLE_RATE = tts_model.config.sampling_rate

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

encounters = []
with open("encounter_texts.jsonl") as f:
    for line in f:
        encounters.append(json.loads(line))
print(f"Loaded {len(encounters)} encounters")

def split_sentences(text, min_len=15, max_len=200):
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    raw = [p.strip() for p in raw if p.strip()]

    broken = []
    for s in raw:
        if len(s) <= max_len:
            broken.append(s)
        else:
            words = s.split()
            buf, cur_len = [], 0
            for w in words:
                buf.append(w)
                cur_len += len(w) + 1
                if cur_len >= max_len:
                    broken.append(" ".join(buf))
                    buf, cur_len = [], 0
            if buf:
                broken.append(" ".join(buf))

    merged, buffer = [], ""
    for s in broken:
        buffer = (buffer + " " + s).strip() if buffer else s
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)
    return merged

def synthesize_sentence(text):
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    return output.squeeze().cpu().numpy().astype(np.float32)

def synthesize_encounter(text, out_path):
    chunks = []
    for s in split_sentences(text):
        wav = None
        for attempt in range(2):
            try:
                wav = synthesize_sentence(s)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 0:
                    continue
            except Exception as e:
                print(f"    (skipped a sentence: {e})")
                break
        if wav is not None:
            chunks.append(wav)
            chunks.append(np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32))
    if not chunks:
        raise RuntimeError("no sentences synthesized successfully")
    full_audio = np.concatenate(chunks)
    sf.write(str(out_path), full_audio, SAMPLE_RATE)

for i, enc in enumerate(encounters):
    eid = enc["encounter_id"]
    audio_path = AUDIO_DIR / f"{eid}.wav"
    whisper_path = WHISPER_DIR / f"{eid}.json"

    if whisper_path.exists():
        continue

    if not audio_path.exists():
        try:
            synthesize_encounter(enc["full_text"], audio_path)
        except Exception as e:
            print(f"[{eid}] TTS FAILED: {e}")
            torch.cuda.empty_cache()
            continue

    torch.cuda.empty_cache()

    try:
        result = whisper_model.transcribe(str(audio_path), word_timestamps=True)
    except Exception as e:
        print(f"[{eid}] Whisper FAILED: {e} — deleting audio to retry next run")
        audio_path.unlink(missing_ok=True)
        continue

    words = [{"word": w["word"], "confidence": w["probability"]}
             for seg in result["segments"] for w in seg.get("words", [])]

    with open(whisper_path, "w") as f:
        json.dump({
            "encounter_id": eid,
            "original_text": enc["full_text"],
            "whisper_text": result["text"],
            "words": words,
        }, f)

    print(f"[{i+1}/{len(encounters)}] {eid} done — {len(words)} words")

print("Batch complete.")