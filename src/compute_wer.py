import json
from pathlib import Path


def edit_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]


def main():

    pred_dir = Path("data/predictions")
    metrics_dir = Path("data/metrics")
    metrics_dir.mkdir(exist_ok=True)

    prediction_files = list(pred_dir.glob("*_pred.jsonl"))

    for pred_path in prediction_files:

        total_words = 0
        total_errors = 0

        with pred_path.open() as f:
            for line in f:
                item = json.loads(line)

                ref_words = item["ref_text"].lower().split()
                hyp_words = item["pred_phon"].lower().split()

                errors = edit_distance(ref_words, hyp_words)

                total_words += len(ref_words)
                total_errors += errors

        wer = total_errors / total_words if total_words > 0 else 0.0

        output_name = pred_path.name.replace("_pred.jsonl", "_metrics.json")
        output_path = metrics_dir / output_name

        with output_path.open("w") as f:
            json.dump({"WER": wer}, f, indent=2)

        print(f"{pred_path.name} → WER: {wer:.4f}")

    print("WER computation complete.")


if __name__ == "__main__":
    main()
    