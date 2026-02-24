import json
from pathlib import Path
import yaml


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())
    languages = params["languages"]

    for lang in languages:
        manifest_path = Path(f"data/manifests/{lang}/clean.jsonl")

        with manifest_path.open("r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]

        predictions = []

        for item in items:
            pred_item = item.copy()
            pred_item["pred_phon"] = "DUMMY_PREDICTION"
            predictions.append(pred_item)

        output_path = Path(f"data/predictions/{lang}_clean_pred.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = output_path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        tmp_path.rename(output_path)


if __name__ == "__main__":
    main()