import json
from pathlib import Path


def main():
    output_path = Path("data/manifests/en/clean.jsonl")

    # Exemple minimal fictif
    items = [
        {
            "utt_id": "en_001",
            "lang": "en",
            "wav_path": "data/raw/en/001.wav",
            "ref_text": "hello world",
            "ref_phon": "həˈloʊ wɜːrld",
            "sr": 16000,
            "duration_s": 1.23,
            "snr_db": None,
        }
    ]

    # Écriture atomique
    tmp_path = output_path.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    tmp_path.rename(output_path)


if __name__ == "__main__":
    main()