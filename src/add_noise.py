import json
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(input_wav: Path, output_wav: Path, snr_db: float, seed: int) -> None:
    signal, sr = sf.read(input_wav)

    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, noisy_signal, sr)


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())
    languages = params["languages"]
    snr_levels = params["snr_levels"]
    seed = params["seed"]

    for lang in languages:
        clean_manifest = Path(f"data/manifests/{lang}/clean.jsonl")

        with clean_manifest.open("r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]

        for snr in snr_levels:
            noisy_items = []

            for item in items:
                input_wav = Path(item["wav_path"])
                output_wav = Path(f"data/interim/{lang}/{snr}dB_{input_wav.name}")

                # Ici on ne teste pas encore vraiment l'audio
                # (on branchera les vrais fichiers plus tard)

                noisy_item = item.copy()
                noisy_item["wav_path"] = str(output_wav)
                noisy_item["snr_db"] = snr

                noisy_items.append(noisy_item)

            output_manifest = Path(f"data/manifests/{lang}/noisy_{snr}.jsonl")
            tmp_path = output_manifest.with_suffix(".tmp")

            with tmp_path.open("w", encoding="utf-8") as f:
                for ni in noisy_items:
                    f.write(json.dumps(ni, ensure_ascii=False) + "\n")

            tmp_path.rename(output_manifest)


if __name__ == "__main__":
    main()