"""
Download PMC-VQA dataset from HuggingFace and save images + metadata.

Source: xmcmic/PMC-VQA
Paper:  Zhang et al., "PMC-VQA: Visual Instruction Tuning for Medical Visual
        Question Answering" (2023, arXiv:2305.10415)

PMC-VQA is a multiple-choice medical VQA benchmark built from PubMed Central
figures.  Each item has a question, four choices (A–D), and a letter answer.

This script downloads the metadata CSVs via ``hf_hub_download`` and the images
zip, then extracts ONLY the images referenced in the chosen split to keep the
on-disk footprint tractable.  (The full ``images.zip`` is ~19 GB; a test split
is a small subset of that.)

Usage
-----
    pip install datasets huggingface_hub pandas tqdm

    # Smallest footprint — `test_clean` (~2k curated MCQ) from v1 images:
    python data/download_pmc_vqa.py

    # Larger test split (~33k) from v1 images:
    python data/download_pmc_vqa.py --split test

    # v2 release (smaller ~2.2 GB zip, ~30k test questions):
    python data/download_pmc_vqa.py --version 2 --split test

    # Skip image extraction (CSV only, for inspection):
    python data/download_pmc_vqa.py --no-extract

Output
------
    data/pmc_vqa/PMC_VQA_<split>.json
    data/pmc_vqa/images/<Figure_path>
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "xmcmic/PMC-VQA"

# PMC-VQA's "Choice A" / "Choice B" / ... CSV columns sometimes already include
# the letter prefix (e.g. "A:X-ray", "B: Magnetic resonance imaging"). Strip
# any leading "<letter><sep>" so the formatted prompt doesn't end up with
# duplicated letters like "A) A:X-ray".
_CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d]\s*[:.)\-]\s*")

# Files available in the HF repo.  v1 (images.zip, 18.9 GB) pairs with
# train.csv / test.csv / test_clean.csv.  v2 (images_2.zip, 2.21 GB) pairs
# with train_2.csv / test_2.csv.
VERSIONS = {
    1: {
        "zip":    "images.zip",
        "splits": {"train": "train.csv", "test": "test.csv", "test_clean": "test_clean.csv"},
    },
    2: {
        "zip":    "images_2.zip",
        "splits": {"train": "train_2.csv", "test": "test_2.csv"},
    },
}


def _clean_choice(text: str) -> str:
    """Strip a duplicate leading letter prefix from a CSV choice value."""
    return _CHOICE_PREFIX_RE.sub("", text).strip()


def _format_question(q: str, choices: dict) -> str:
    """Embed A/B/C/D choices into the question text."""
    lines = [q.strip()]
    for letter in ("A", "B", "C", "D"):
        text = _clean_choice(choices.get(letter, ""))
        lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def _normalize_answer_letter(raw: str) -> str | None:
    """Pull a single A/B/C/D out of the CSV ``Answer`` column.

    Observed values in the dataset include ``"A"``, ``"A:foo bar"``, and
    occasionally the full choice text — so we look for a leading letter first
    and fall back to matching the choice text.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # Common case: "A", "A.", "A:...", "(A)"
    for i, ch in enumerate(s.upper()):
        if ch in "ABCD":
            # require this letter to be "standalone" — surrounded by
            # non-letter chars — to avoid matching the first letter of a
            # word like "Arterial".
            prev_ok = i == 0 or not s[i - 1].isalpha()
            next_ok = i + 1 == len(s) or not s[i + 1].isalpha()
            if prev_ok and next_ok:
                return ch
        # don't scan past the first few chars
        if i >= 3:
            break
    return None


def _resolve_letter(row) -> str | None:
    """Determine the answer letter for a CSV row.

    ``Answer_label`` is usually the canonical letter; fall back to parsing
    ``Answer`` or matching against the four choices.
    """
    label = str(row.get("Answer_label", "")).strip()
    letter = _normalize_answer_letter(label)
    if letter:
        return letter
    letter = _normalize_answer_letter(str(row.get("Answer", "")))
    if letter:
        return letter
    # Last resort: find which choice matches the free-text Answer
    ans = str(row.get("Answer", "")).strip().lower()
    for L in ("A", "B", "C", "D"):
        choice = str(row.get(f"Choice {L}", "")).strip().lower()
        if choice and (choice == ans or choice in ans or ans in choice):
            return L
    return None


def download(version: int, split: str, out_root: Path, extract: bool) -> None:
    cfg = VERSIONS[version]
    if split not in cfg["splits"]:
        raise ValueError(
            f"Split '{split}' not available for v{version}. "
            f"Options: {list(cfg['splits'])}"
        )
    csv_filename = cfg["splits"][split]
    zip_filename = cfg["zip"]

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. metadata CSV ───────────────────────────────────────────────────
    print(f"Fetching {csv_filename} …")
    csv_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=csv_filename
    )
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} rows")

    # ── 2. build JSON records ─────────────────────────────────────────────
    records = []
    skipped_no_letter = 0
    for i, row in df.iterrows():
        figure = str(row.get("Figure_path", "")).strip()
        question = str(row.get("Question", "")).strip()
        if not figure or not question:
            continue

        choices = {
            L: _clean_choice(str(row.get(f"Choice {L}", "")))
            for L in ("A", "B", "C", "D")
        }
        letter = _resolve_letter(row)
        if letter is None:
            skipped_no_letter += 1
            continue

        records.append({
            "idx":           int(i),
            "image":         str(images_dir / figure),
            "question":      _format_question(question, choices),
            "answer":        letter,
            "answer_text":   choices.get(letter, ""),
            "choices":       choices,
            "question_type": "mcq",
            "figure":        figure,
        })

    if skipped_no_letter:
        print(f"  skipped {skipped_no_letter} rows with unparseable answer")

    # ── 3. images.zip — download and selectively extract ──────────────────
    needed = {r["figure"] for r in records}
    missing = {n for n in needed if not (images_dir / n).exists()}
    if extract and missing:
        print(f"Fetching {zip_filename} ({len(missing)}/{len(needed)} images missing) …")
        zip_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=zip_filename
        )
        print(f"Extracting {len(missing)} images from {zip_path} …")
        with zipfile.ZipFile(zip_path) as zf:
            # Build a lookup of basename -> zip member name, since the zip
            # may store images in a nested folder.
            by_base = {}
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                by_base.setdefault(os.path.basename(member), member)

            hits = 0
            for fname in tqdm(sorted(missing), desc="  extracting"):
                member = by_base.get(fname)
                if member is None:
                    continue
                target = images_dir / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                hits += 1
            print(f"  extracted {hits}/{len(missing)} images")
    elif not extract:
        print("(--no-extract) skipping image extraction")

    # ── 4. drop records whose image is still missing (if extracted) ───────
    if extract:
        before = len(records)
        records = [r for r in records if Path(r["image"]).exists()]
        dropped = before - len(records)
        if dropped:
            print(f"  dropped {dropped} records with missing image files")

    # ── 5. write JSON ─────────────────────────────────────────────────────
    out_path = out_root / f"PMC_VQA_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(records)} questions -> {out_path}")

    # ── 6. sample preview ────────────────────────────────────────────────
    if records:
        s = records[0]
        print("\nSample[0]:")
        print(f"  image:    {s['image']}")
        print(f"  question: {s['question'][:120]}{'…' if len(s['question']) > 120 else ''}")
        print(f"  answer:   {s['answer']}  ({s['answer_text']})")


def main():
    p = argparse.ArgumentParser(description="Download PMC-VQA dataset")
    p.add_argument("--version", type=int, default=1, choices=[1, 2],
                   help="Image set version (1=images.zip 18.9GB, 2=images_2.zip 2.2GB)")
    p.add_argument("--split",   type=str, default="test_clean",
                   help="Which split to build (default: test_clean for v1)")
    p.add_argument("--out",     type=str, default="data/pmc_vqa",
                   help="Output root directory")
    p.add_argument("--no-extract", dest="extract", action="store_false",
                   default=True, help="Skip downloading/extracting images.zip")
    args = p.parse_args()

    download(args.version, args.split, Path(args.out), args.extract)


if __name__ == "__main__":
    main()
