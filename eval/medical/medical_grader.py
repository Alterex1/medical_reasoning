"""
medical_grader.py
=================
Grading logic for medical VQA tasks (VQA-RAD, SLAKE, PathVQA, etc.).

Handles two question types:
  - Closed-ended (yes/no): exact match after normalization
  - Open-ended (free-form): normalized string matching with common
    medical abbreviation handling

Unlike the math grader (SymPy + LaTeX), medical grading is primarily
string-based with domain-specific normalization.
"""
import re
from typing import Optional


# ─── Answer extraction ────────────────────────────────────────────────────────

def parse_medical_answer(text: str, ground_truth: str = None) -> Optional[str]:
    """
    Extract the final answer from a model completion.

    Strategy (in order of priority):
      1. Explicit answer patterns: "The answer is: ...", "Answer: ..."
      2. Markdown bold answer: "**Answer:** ..."
      3. For yes/no questions: scan the entire completion for yes/no signal
      4. Conclusion/summary patterns: "In conclusion, ...", "Therefore, ..."
      5. Fallback: last non-empty line
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. Pattern: "the answer is: X" or "the answer is X"
    match = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*[:\s]*(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().rstrip(".")

    # 2. Pattern: "**Answer:** X" (markdown bold)
    match = re.search(
        r"\*\*[Aa]nswer:?\*\*\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().rstrip(".")

    # 3. For yes/no questions: scan the full text for yes/no indicators
    #    This handles truncated CoT where the model reasons but never
    #    reaches a formal "Answer:" line.
    if ground_truth is not None and is_closed_ended(ground_truth):
        answer = _extract_yes_no_from_text(text)
        if answer is not None:
            return answer

    # 4. Conclusion/summary patterns
    match = re.search(
        r"(?:in\s+conclusion|therefore|thus|hence|so|overall)[,:]?\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().rstrip(".")

    # 5. Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].rstrip(".")

    return None


def _extract_yes_no_from_text(text: str) -> Optional[str]:
    """
    Scan a completion for yes/no signal, handling truncated CoT.

    Looks for strong indicators like:
      - "Yes, there is..."
      - "No, there is no..."
      - "there is evidence of..." → yes
      - "no evidence of..." → no
      - "is not seen" / "is absent" → no
      - "is present" / "is seen" → yes
    """
    text_lower = text.lower()

    # Explicit yes/no statements (check last occurrence — most likely the conclusion)
    yes_patterns = [
        r"\byes\b",
        r"\bthere\s+is\s+evidence\b",
        r"\bevidence\s+of\b(?!.*\bno\s+evidence\b)",
        r"\bis\s+(?:present|seen|visible|noted|observed|identified)\b",
        r"\bfindings?\s+(?:suggest|indicate|consistent)\b",
    ]
    no_patterns = [
        r"\bno\b(?:\s*,|\s+there|\s+evidence|\s+sign|\s+finding)",
        r"\bno\s+evidence\b",
        r"\bnot?\s+(?:present|seen|visible|noted|observed|identified)\b",
        r"\bis\s+absent\b",
        r"\bwithout\s+(?:evidence|sign|finding)\b",
        r"\bnormal\s+(?:appearing|size|shape)\b.*\bno\b",
    ]

    # Check from the end of the text backward — later statements are more conclusive
    # Split into sentences and check last few
    sentences = re.split(r'[.!?\n]', text_lower)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Check last 5 sentences for conclusive statements
    for sent in reversed(sentences[-5:]):
        for pat in no_patterns:
            if re.search(pat, sent):
                return "no"
        for pat in yes_patterns:
            if re.search(pat, sent):
                return "yes"

    # Broader scan of full text as last resort
    no_count = sum(1 for pat in no_patterns if re.search(pat, text_lower))
    yes_count = sum(1 for pat in yes_patterns if re.search(pat, text_lower))

    if no_count > yes_count:
        return "no"
    if yes_count > no_count:
        return "yes"

    return None


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_medical_answer(answer: str) -> str:
    """
    Normalize a medical answer for comparison.

    - Lowercase
    - Strip whitespace and punctuation
    - Normalize common yes/no variants
    - Remove articles
    """
    if answer is None:
        return ""

    s = answer.lower().strip()

    # Remove trailing punctuation
    s = re.sub(r"[.!?,;:]+$", "", s)

    # Remove leading articles
    s = re.sub(r"^(the|a|an)\s+", "", s)

    # Remove extra whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize yes/no
    yes_variants = {"yes", "yep", "yeah", "correct", "true", "positive", "affirmative"}
    no_variants = {"no", "nope", "nah", "incorrect", "false", "negative"}

    if s in yes_variants:
        return "yes"
    if s in no_variants:
        return "no"

    # Handle answers that start with yes/no followed by explanation
    # e.g., "no, there are no abnormalities" → "no"
    # e.g., "yes, there is evidence of..." → "yes"
    match = re.match(r"^(yes|no)\b", s)
    if match:
        return match.group(1)

    return s


# ─── Grading ──────────────────────────────────────────────────────────────────

def grade_medical_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Grade a medical VQA answer against ground truth.

    For closed-ended (yes/no): exact match after normalization.
    For open-ended: normalized string match, with substring fallback
    for short ground truth answers.
    """
    if predicted is None:
        return False

    pred_norm = normalize_medical_answer(predicted)
    gt_norm = normalize_medical_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return False

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Check if ground truth is contained in prediction (for short answers)
    # e.g., gt="liver", pred="the liver is enlarged" → match
    if len(gt_norm) >= 3 and gt_norm in pred_norm:
        return True

    # Check if prediction is contained in ground truth
    if len(pred_norm) >= 3 and pred_norm in gt_norm:
        return True

    return False


def is_closed_ended(answer: str) -> bool:
    """Check if a question has a closed-ended (yes/no) answer."""
    norm = normalize_medical_answer(answer)
    return norm in {"yes", "no"}
