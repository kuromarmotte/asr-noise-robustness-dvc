import json
from pathlib import Path


def edit_distance(ref: str, hyp: str) -> int:
    m = len(ref)
    n = len(hyp)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


def main():
    pred_manifest = Path("data/predictions/en_clean_pred.jsonl")

    total_edits = 0
    total_ref_len = 0

    with pred_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ref = item["ref_phon"]
            hyp = item["pred_phon"]

            total_edits += edit_distance(ref, hyp)
            total_ref_len += len(ref)

    per = total_edits / total_ref_len if total_ref_len > 0 else 0.0

    output_path = Path("data/metrics/en_clean_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"PER": per}, f, indent=2)


if __name__ == "__main__":
    main()