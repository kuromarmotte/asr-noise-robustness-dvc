import json
import hashlib
from pathlib import Path

import yaml


def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())

    languages = params["languages"]
    sample_rate = params["sample_rate"]

    for lang in languages:
        output_path = Path(f"data/manifests/{lang}/clean.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        items = []

        # Exemple minimal simulé
        wav_path = f"data/raw/{lang}/001.wav"

        item = {
            "utt_id": f"{lang}_001",
            "lang": lang,
            "wav_path": wav_path,
            "ref_text": "hello world",
            "ref_phon": "həˈloʊ wɜːrld",
            "sr": sample_rate,
            "duration_s": 1.23,
            "snr_db": None,
            "audio_md5": compute_md5(wav_path),
        }

        items.append(item)

        tmp_path = output_path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        tmp_path.rename(output_path)


if __name__ == "__main__":
    main()