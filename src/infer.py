import json
from pathlib import Path
import yaml
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_params():
    return yaml.safe_load(Path("params.yaml").read_text())


def main():

    params = load_params()
    languages = params.get("languages", [])
    sample_rate = params.get("sample_rate")

    if not languages:
        raise ValueError("No languages defined in params.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.to(device).eval()

    prediction_dir = Path("data/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:

        manifest_dir = Path("data/manifests") / lang
        if not manifest_dir.exists():
            print(f"[WARNING] No manifest directory for {lang}, skipping.")
            continue

        for manifest_path in manifest_dir.glob("*.jsonl"):

            print(f"Processing {lang}/{manifest_path.name}...")

            output_path = prediction_dir / f"{lang}_{manifest_path.stem}_pred.jsonl"
            tmp_output = output_path.with_suffix(".tmp")

            with manifest_path.open() as f_in, tmp_output.open("w") as f_out:

                for line in f_in:
                    item = json.loads(line)
                    wav_path = Path(item["wav_path"])

                    if not wav_path.exists():
                        continue

                    speech, sr = sf.read(wav_path)

                    if speech.ndim != 1 or sr != sample_rate:
                        continue

                    inputs = processor(
                        speech,
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        padding=True,
                    )

                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = model(**inputs).logits

                    pred_ids = torch.argmax(logits, dim=-1)
                    item["pred_text"] = processor.batch_decode(pred_ids)[0]

                    f_out.write(json.dumps(item) + "\n")

            tmp_output.rename(output_path)
            print(f"Saved → {output_path.name}")

    print("Inference complete.")


if __name__ == "__main__":
    main()