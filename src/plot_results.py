import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():

    metrics_dir = Path("data/metrics")

    snr_levels = []
    wer_values = []

    for file in metrics_dir.glob("noisy_*_metrics.json"):

        snr = file.name.split("_")[1]  # ex: noisy_20_metrics.json
        snr = int(snr)

        with file.open() as f:
            data = json.load(f)

        snr_levels.append(snr)
        wer_values.append(data["WER"])

    # Trier par SNR croissant
    snr_levels, wer_values = zip(*sorted(zip(snr_levels, wer_values)))

    plt.figure()
    plt.plot(snr_levels, wer_values)
    plt.xlabel("SNR (dB)")
    plt.ylabel("WER")
    plt.title("WER vs SNR")
    plt.savefig("data/figures/wer_vs_snr.png")

    print("Plot saved → data/figures/wer_vs_snr.png")


if __name__ == "__main__":
    main()
