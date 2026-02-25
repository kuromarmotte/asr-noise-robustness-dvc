import json
import hashlib
from pathlib import Path

import yaml
import soundfile as sf


# ----------------------------
# Utilities
# ----------------------------

def compute_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_params():
    return yaml.safe_load(Path("params.yaml").read_text())


def process_language(lang: str, sample_rate: int) -> int:
    raw_dir = Path("data/raw") / lang

    if not raw_dir.exists():
        print(f"[WARNING] No raw directory for {lang}, skipping.")
        return 0

    wav_files = sorted(raw_dir.glob("*.wav"))

    if not wav_files:
        print(f"[WARNING] No wav files for {lang}, skipping.")
        return 0

    output_path = Path("data/manifests") / lang / "clean.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")

    count = 0

    with tmp_path.open("w", encoding="utf-8") as f_out:

        for wav_path in wav_files:

            signal, sr = sf.read(wav_path)

            if signal.ndim != 1:
                print(f"[WARNING] {wav_path} is not mono. Skipping.")
                continue

            if sr != sample_rate:
                raise ValueError(
                    f"{wav_path} has sample rate {sr}, expected {sample_rate}"
                )

            duration = len(signal) / sr
            stem = wav_path.stem

            txt_path = wav_path.with_suffix(".txt")
            ref_text = (
                txt_path.read_text(encoding="utf-8").strip()
                if txt_path.exists()
                else ""
            )

            if not ref_text:
                print(f"[WARNING] No transcript for {wav_path}")

            item = {
                "utt_id": f"{lang}_{stem}",
                "lang": lang,
                "wav_path": str(wav_path),
                "ref_text": ref_text,
                "ref_phon": "",
                "sr": sr,
                "duration_s": round(duration, 3),
                "snr_db": None,
                "audio_md5": compute_md5(wav_path),
            }

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    tmp_path.rename(output_path)
    return count


# ----------------------------
# Main
# ----------------------------

def main():
    params = load_params()
    languages = params.get("languages", [])
    sample_rate = params.get("sample_rate")

    if not languages or sample_rate is None:
        raise ValueError("Missing languages or sample_rate in params.yaml")

    for lang in languages:
        count = process_language(lang, sample_rate)
        if count:
            print(f"{lang} clean manifest generated with {count} items.")

    print("Manifest generation complete.")


if __name__ == "__main__":
    main()