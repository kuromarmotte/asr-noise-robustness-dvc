import json
from pathlib import Path
import yaml


def edit_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[i + j if i * j == 0 else 0 for j in range(n + 1)]
          for i in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


def load_params():
    return yaml.safe_load(Path("params.yaml").read_text())


def compute_wer_for_file(pred_path: Path):
    total_words = 0
    total_errors = 0

    with pred_path.open() as f:
        for line in f:
            item = json.loads(line)

            ref = item["ref_text"].lower().split()
            hyp = item["pred_text"].lower().split()

            if not ref:
                continue

            total_words += len(ref)
            total_errors += edit_distance(ref, hyp)

    return (total_errors / total_words) if total_words else 0.0


def main():

    params = load_params()
    languages = params.get("languages", [])
    snr_levels = params.get("snr_levels", [])

    pred_dir = Path("data/predictions")
    metrics_dir = Path("data/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    snr_keys = ["clean"] + [str(s) for s in snr_levels]

    for lang in languages:
        summary[lang] = {}

        for snr in snr_keys:

            filename = (
                f"{lang}_clean_pred.jsonl"
                if snr == "clean"
                else f"{lang}_noisy_{snr}_pred.jsonl"
            )

            pred_path = pred_dir / filename

            if not pred_path.exists():
                print(f"[WARNING] Missing {filename}, skipping.")
                continue

            wer = compute_wer_for_file(pred_path)
            summary[lang][snr] = wer

            print(f"{lang} | SNR {snr} → WER: {wer:.4f}")

    # ---- Mean across languages ----
    summary["mean"] = {}

    for snr in snr_keys:
        values = [
            summary[lang][snr]
            for lang in languages
            if snr in summary.get(lang, {})
        ]
        if values:
            summary["mean"][snr] = sum(values) / len(values)

    # ---- Atomic write ----
    summary_path = metrics_dir / "summary.json"
    tmp_path = summary_path.with_suffix(".tmp")

    with tmp_path.open("w") as f:
        json.dump(summary, f, indent=2)

    tmp_path.rename(summary_path)

    print("WER computation complete.")


if __name__ == "__main__":
    main()