import json
from pathlib import Path

import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def main():

    print("Loading model...")

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    model.eval()

    manifest_dir = Path("data/manifests/en")
    prediction_dir = Path("data/predictions")
    prediction_dir.mkdir(exist_ok=True)

    manifests = [
        "clean.jsonl",
        "noisy_20.jsonl",
        "noisy_15.jsonl",
        "noisy_10.jsonl",
        "noisy_5.jsonl",
        "noisy_0.jsonl",
    ]

    for manifest_name in manifests:

        print(f"Processing {manifest_name}...")

        manifest_path = manifest_dir / manifest_name
        output_name = manifest_name.replace(".jsonl", "_pred.jsonl")
        output_path = prediction_dir / output_name

        with manifest_path.open() as f, output_path.open("w") as out_f:

            for line in f:
                item = json.loads(line)

                wav_path = "data/raw/en/001.wav"
                speech, sr = sf.read(wav_path)

                inputs = processor(
                    speech,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )

                with torch.no_grad():
                    logits = model(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                item["pred_phon"] = transcription
                out_f.write(json.dumps(item) + "\n")

        print(f"Saved → {output_name}")

    print("Inference complete.")


if __name__ == "__main__":
    main()