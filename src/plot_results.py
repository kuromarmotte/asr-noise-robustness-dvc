import json
from pathlib import Path
import matplotlib.pyplot as plt
import yaml


def load_params():
    return yaml.safe_load(Path("params.yaml").read_text())


def extract_curve(values: dict, snr_levels):
    snrs = []
    wers = []

    for snr in ["clean"] + [str(s) for s in snr_levels]:

        if snr not in values:
            continue

        x = -1 if snr == "clean" else int(snr)
        snrs.append(x)
        wers.append(values[snr])

    return snrs, wers


def main():

    params = load_params()
    languages = params.get("languages", [])
    snr_levels = params.get("snr_levels", [])

    summary_path = Path("data/metrics/summary.json")

    if not summary_path.exists():
        print("[WARNING] summary.json not found. Run compute_wer.py first.")
        return

    summary = json.loads(summary_path.read_text())

    if not summary:
        print("No data available in summary.")
        return

    plt.figure()

    # ---- Language curves ----
    for lang in languages:

        if lang not in summary:
            print(f"[WARNING] No results for {lang}, skipping.")
            continue

        snrs, wers = extract_curve(summary[lang], snr_levels)

        if snrs:
            plt.plot(snrs, wers, marker="o", label=lang)

    # ---- Mean curve ----
    if "mean" in summary:
        snrs, wers = extract_curve(summary["mean"], snr_levels)
        if snrs:
            plt.plot(snrs, wers, linestyle="--", linewidth=2, label="mean")

    plt.xlabel("SNR (dB)  (clean = -1)")
    plt.ylabel("WER")
    plt.title("WER vs SNR (Multi-language)")
    plt.legend()
    plt.grid(True)

    output_path = Path("data/figures/wer_vs_snr_multilang.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    print(f"Plot saved → {output_path}")


if __name__ == "__main__":
    main()