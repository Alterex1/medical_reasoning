"""
Download VQA-RAD dataset from HuggingFace and save images + metadata.

Source: flaviagiammarino/vqa-rad
Paper: Lau et al., "A dataset of clinically generated visual questions and
       answers about radiology images" (Scientific Data, 2018)

Dataset: 314 radiology images, 2,244 QA pairs (train: 1,793, test: 451)
Question types: binary (yes/no) and open-ended
Modalities: head, chest, abdomen

Usage:
    pip install datasets Pillow
    python download_medqa.py
"""
import json
import os
from datasets import load_dataset


def main():
    print("Downloading VQA-RAD from HuggingFace...")
    ds = load_dataset("flaviagiammarino/vqa-rad")

    os.makedirs("data/vqa_rad/images", exist_ok=True)

    for split in ds:
        data = []
        seen_images = set()

        for i, example in enumerate(ds[split]):
            # Save image to disk (deduplicate by content hash)
            img = example["image"]
            img_filename = f"{split}_{i:04d}.jpg"
            img_path = f"data/vqa_rad/images/{img_filename}"

            img.save(img_path)

            data.append({
                "idx": i,
                "image": img_path,
                "question": example["question"],
                "answer": example["answer"],
            })

        out_path = f"data/vqa_rad/VQA_RAD_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {split}: {len(data)} QA pairs -> {out_path}")

    # Print samples for verification
    for split in ds:
        sample = ds[split][0]
        print(f"\nSample ({split}[0]):")
        print(f"  Question: {sample['question']}")
        print(f"  Answer:   {sample['answer']}")
        print(f"  Image:    {sample['image'].size} {sample['image'].mode}")

    print("\nDone. Images saved to data/vqa_rad/images/")


if __name__ == "__main__":
    main()
