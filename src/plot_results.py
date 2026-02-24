import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    metrics_path = Path("data/metrics/en_clean_metrics.json")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    per_value = metrics["PER"]

    # Pour l’instant on simule un seul point (SNR = None)
    snr_values = [0]
    per_values = [per_value]

    plt.figure()
    plt.plot(snr_values, per_values)
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title("PER vs Noise (English)")
    plt.grid(True)

    output_path = Path("data/figures/per_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()