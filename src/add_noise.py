import json
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml


# ----------------------------
# Core noise logic (pure)
# ----------------------------

def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def add_noise_to_file(input_wav: Path, output_wav: Path, snr_db: float, seed: int):
    signal, sr = sf.read(input_wav)

    if signal.ndim != 1:
        raise ValueError(f"{input_wav} is not mono audio.")

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, noisy_signal, sr)


# ----------------------------
# Utilities
# ----------------------------

def load_params():
    return yaml.safe_load(Path("params.yaml").read_text())


def process_manifest(clean_manifest: Path, lang: str, snr: int, seed: int):
    output_manifest = clean_manifest.parent / f"noisy_{snr}.jsonl"
    tmp_manifest = output_manifest.with_suffix(".tmp")

    count = 0

    with clean_manifest.open() as f_in, tmp_manifest.open("w") as f_out:

        for line in f_in:
            item = json.loads(line)
            input_wav = Path(item["wav_path"])

            if not input_wav.exists():
                continue

            output_wav = Path("data/interim") / lang / f"{snr}dB_{input_wav.name}"

            add_noise_to_file(input_wav, output_wav, snr, seed)

            item["wav_path"] = str(output_wav)
            item["snr_db"] = snr

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    tmp_manifest.rename(output_manifest)
    return count


# ----------------------------
# Main pipeline
# ----------------------------

def main():

    params = load_params()
    languages = params.get("languages", [])
    snr_levels = params.get("snr_levels", [])
    seed = params.get("seed")

    if not languages or not snr_levels:
        raise ValueError("Languages or SNR levels missing in params.yaml")

    for lang in languages:

        clean_manifest = Path("data/manifests") / lang / "clean.jsonl"

        if not clean_manifest.exists():
            print(f"[WARNING] Clean manifest for {lang} not found, skipping.")
            continue

        for snr in snr_levels:
            count = process_manifest(clean_manifest, lang, snr, seed)
            print(f"{lang} → SNR {snr} dB generated ({count} items).")

    print("Noise generation complete.")


if __name__ == "__main__":
    main()