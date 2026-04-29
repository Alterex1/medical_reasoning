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

def parse_mcq_answer(text: str, choices: Optional[dict] = None) -> Optional[str]:
    """
    Extract a single-letter MCQ answer (A/B/C/D) from a model completion.

    Strategy:
      1. Look after an explicit "Answer:" marker and take the first A-D there.
      2. Check the last few lines for a bare letter or "(A)"-style answer.
      3. Bold-formatted final answer: **A** or **(A)**
      4. Fallback: first standalone A-D anywhere in the text.
      5. (Optional) If still None and ``choices`` was provided, fuzzy-match
         the completion text against the four choice strings — needed for
         models like LLaVA-Med that state the answer textually
         ("magnetic resonance imaging") without picking a letter.

    Args:
        text:    Model completion (free-form).
        choices: Optional dict like ``{"A": "MRI", "B": "CT", ...}``. When
                 provided, enables the textual-match fallback.
    """
    if not text or not text.strip():
        return None

    # 1. After "Answer:" marker
    m = re.search(r"(?:^|[\s\*])[Aa]nswer\s*[:\-]?\s*\*{0,2}\s*\(?\s*([ABCD])\b",
                  text)
    if m:
        return m.group(1).upper()

    # 2. Last few lines — a line that's just a letter or "(A)"
    for line in reversed([l.strip() for l in text.strip().splitlines() if l.strip()][-4:]):
        m = re.match(r"^\**\s*\(?\s*([ABCD])\s*\)?\s*[.)]?\s*\**\s*$", line)
        if m:
            return m.group(1).upper()

    # 3. Bold-formatted final answer: **A** or **(A)**
    m = re.search(r"\*\*\s*\(?([ABCD])\)?\s*\*\*", text)
    if m:
        return m.group(1).upper()

    # 4. Fallback: first standalone A/B/C/D
    m = re.search(r"(?:^|[\s(\[\"'])([ABCD])(?=[\s.,;:)\]\"'?!]|$)", text)
    if m:
        return m.group(1).upper()

    # 5. Textual-match fallback (for free-text answers without a letter)
    if choices:
        return _fuzzy_match_choice(text, choices)

    return None


def _normalize_for_match(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fuzzy_match_choice(text: str, choices: dict) -> Optional[str]:
    """
    Match a free-text completion to one of the lettered choices by content
    overlap. Returns the letter (A/B/C/D) or None if no choice matches
    confidently.

    Strategy:
      - Normalize text and each choice (lowercase, strip punctuation).
      - For each choice, score by longest substring match (choice in text).
      - Pick the choice with the longest match.
      - Reject ambiguous cases where two choices match within 3 characters
        of each other — better to return None than guess.
    """
    if not text or not choices:
        return None

    text_norm = _normalize_for_match(text)
    if not text_norm:
        return None

    # Collect (letter, match_length) for each choice that's a substring
    # of the completion.
    matches: list[tuple[str, int]] = []
    for letter, choice_text in choices.items():
        if not choice_text:
            continue
        choice_norm = _normalize_for_match(choice_text)
        # Skip very short choices — too ambiguous (would match anywhere)
        if len(choice_norm) < 2:
            continue
        if choice_norm in text_norm:
            matches.append((letter, len(choice_norm)))

    if not matches:
        return None

    # Longest match wins (most specific); ties broken by letter order
    matches.sort(key=lambda x: (-x[1], x[0]))
    best_letter, best_len = matches[0]

    # Reject ambiguous: if a different letter matched almost as long
    if len(matches) > 1:
        second_letter, second_len = matches[1]
        if second_letter != best_letter and (best_len - second_len) < 3:
            return None

    return best_letter


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
        # Broader "no" pattern: "no" followed by any noun/adjective (catches
        # short-form responses like "no intraparenchymal abnormalities")
        r"\bno\s+\w+\s+(?:abnormalit|lesion|fracture|mass|effusion|finding|opacity|enlargement|patholog)",
        # "does not show" / "did not reveal" / "shows no" / "reveals no"
        r"\b(?:does|did)\s+not\s+(?:show|reveal|demonstrate|indicate)\b",
        r"\b(?:show|reveal|demonstrate|indicate)s?\s+no\b",
        # "there is/are no ..." (broader than first pattern)
        r"\bthere\s+(?:is|are)\s+no\b",
        # "not seen here" / "is not present"
        r"\bnot\s+seen\b",
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

    # Handle full sentences with implicit yes/no signal
    # e.g., "there are no intraparenchymal abnormalities in the lung fields" → "no"
    # e.g., "the image shows evidence of an aortic aneurysm" → "yes"
    # Only apply this when the answer looks like a full sentence (has spaces)
    if " " in s:
        extracted = _extract_yes_no_from_text(s)
        if extracted is not None:
            return extracted

    return s


# ─── Grading ──────────────────────────────────────────────────────────────────

def grade_medical_answer(predicted: Optional[str], ground_truth: str,
                         question_type: Optional[str] = None) -> bool:
    """
    Grade a medical VQA answer against ground truth.

    For MCQ (``question_type == "mcq"``): compare single letter A-D.
    For closed-ended (yes/no): exact match after normalization.
    For open-ended: normalized string match, with substring fallback
    for short ground truth answers.
    """
    if predicted is None:
        return False

    # Multiple-choice: compare by letter.  ``predicted`` may be either a
    # bare letter (already parsed by parse_mcq_answer) or raw completion
    # text — handle both.
    if question_type == "mcq":
        gt_letter = (ground_truth or "").strip().upper()
        if len(gt_letter) != 1 or gt_letter not in "ABCD":
            return False
        pred_letter = predicted.strip().upper()
        if len(pred_letter) != 1 or pred_letter not in "ABCD":
            pred_letter = parse_mcq_answer(predicted) or ""
        return pred_letter == gt_letter

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
