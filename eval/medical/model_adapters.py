"""
VLM Model Adapters — abstraction layer for different vision-language models.

Each adapter handles model loading, image processing, and input construction
so the eval scripts stay model-agnostic. The core Power-SMC algorithm
(core/power_smc.py) is already model-agnostic via prefill_kwargs.

Supported models:
  - Qwen2.5-VL (default)
  - LLaVA-Med (requires: pip install git+https://github.com/microsoft/LLaVA-Med.git)
  - Med-Gemma (google/medgemma-4b-it)
"""
import os
from abc import ABC, abstractmethod

import torch
from PIL import Image


# ─── Prompt templates (shared across all models) ────────────────────────────

PROMPT_COT_CLOSED = (
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Please reason step by step about what you observe in the image, "
    "then provide your final answer as a single word (Yes or No) after 'Answer:'."
)

PROMPT_COT_OPEN = (
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Please reason step by step about what you observe in the image, "
    "then provide your final answer as a short phrase after 'Answer:'."
)

PROMPT_DIRECT = (
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Answer:"
)


def get_prompt(question: str, cot: bool = True, question_type: str = "open") -> str:
    """Select and format the prompt text (model-agnostic)."""
    if not cot:
        return PROMPT_DIRECT.format(question=question)
    elif question_type == "closed":
        return PROMPT_COT_CLOSED.format(question=question)
    else:
        return PROMPT_COT_OPEN.format(question=question)


# ─── Base adapter ────────────────────────────────────────────────────────────

class VLMAdapter(ABC):
    """Abstract base for VLM model adapters."""

    model: torch.nn.Module
    tokenizer: object  # PreTrainedTokenizerBase
    device: str

    @classmethod
    @abstractmethod
    def load(cls, model_name: str, dtype: torch.dtype, device: str) -> "VLMAdapter":
        """Load model and tokenizer/processor."""
        ...

    @abstractmethod
    def prepare_inputs(self, question: str, image_path: str,
                       prompt_text: str) -> dict:
        """
        Build model inputs from a question, image, and prompt.

        Returns dict with:
            "input_ids":      torch.LongTensor [1, seq_len]
            "attention_mask": torch.LongTensor [1, seq_len] or None
            "prefill_kwargs": dict of extra kwargs for VLM prefill
            "prompt_len":     int (length of the prompt for trimming)
        """
        ...

    @abstractmethod
    def prepare_generate_inputs(self, question: str, image_path: str,
                                prompt_text: str) -> dict:
        """
        Build inputs for standard model.generate() (baseline eval).

        Returns dict that can be splatted into model.generate(**result).
        Must include "prompt_len" key (popped before calling generate).
        """
        ...


# ─── Qwen2.5-VL adapter ─────────────────────────────────────────────────────

class QwenVLAdapter(VLMAdapter):
    """Adapter for Qwen2.5-VL models."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device

    @classmethod
    def load(cls, model_name: str, dtype: torch.dtype, device: str) -> "QwenVLAdapter":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # Attention implementation: flash_attention_2 > sdpa
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"Using attention implementation: {attn_impl}")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        return cls(model, processor, device)

    def _build_messages(self, image_path: str, prompt_text: str) -> list:
        """Build Qwen2.5-VL chat message format."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _process(self, image_path: str, prompt_text: str):
        """Shared input processing for both SMC and baseline."""
        messages = self._build_messages(image_path, prompt_text)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def prepare_inputs(self, question: str, image_path: str,
                       prompt_text: str) -> dict:
        inputs = self._process(image_path, prompt_text)

        prefill_kwargs = {}
        if "pixel_values" in inputs:
            prefill_kwargs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            prefill_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "prefill_kwargs": prefill_kwargs,
            "prompt_len": inputs["input_ids"].shape[1],
        }

    def prepare_generate_inputs(self, question: str, image_path: str,
                                prompt_text: str) -> dict:
        inputs = self._process(image_path, prompt_text)
        result = {k: v for k, v in inputs.items()}
        result["prompt_len"] = inputs["input_ids"].shape[1]
        return result


# ─── LLaVA-Med adapter ──────────────────────────────────────────────────────

class LlavaMedAdapter(VLMAdapter):
    """Adapter for LLaVA-Med (microsoft/llava-med-v1.5-mistral-7b)."""

    def __init__(self, model, tokenizer, image_processor, device):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device

    @classmethod
    def load(cls, model_name: str, dtype: torch.dtype, device: str) -> "LlavaMedAdapter":
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except ImportError:
            raise ImportError(
                "LLaVA-Med requires the llava package. Install with:\n"
                "  pip install git+https://github.com/microsoft/LLaVA-Med.git"
            )

        llava_model_name = get_model_name_from_path(model_name)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_name,
            model_base=None,
            model_name=llava_model_name,
            device_map="auto",
            device=device,
        )
        # Cast to requested dtype after loading
        model = model.to(dtype=dtype).eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return cls(model, tokenizer, image_processor, device)

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for LLaVA."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"].to(self.device)
        return image_tensor

    def _build_prompt(self, prompt_text: str) -> str:
        """Build LLaVA Mistral Instruct prompt format."""
        return f"USER: <image>\n{prompt_text}\nASSISTANT:"

    def prepare_inputs(self, question: str, image_path: str,
                       prompt_text: str) -> dict:
        prompt = self._build_prompt(prompt_text)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        image_tensor = self._process_image(image_path)

        return {
            "input_ids": input_ids,
            "attention_mask": None,
            "prefill_kwargs": {"images": image_tensor},
            "prompt_len": input_ids.shape[1],
        }

    def prepare_generate_inputs(self, question: str, image_path: str,
                                prompt_text: str) -> dict:
        prompt = self._build_prompt(prompt_text)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        image_tensor = self._process_image(image_path)

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "prompt_len": input_ids.shape[1],
        }


# ─── Med-Gemma adapter ──────────────────────────────────────────────────────

class MedGemmaAdapter(VLMAdapter):
    """Adapter for Med-Gemma (google/medgemma-4b-it)."""

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device

    @classmethod
    def load(cls, model_name: str, dtype: torch.dtype, device: str) -> "MedGemmaAdapter":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Attention implementation
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        print(f"Using attention implementation: {attn_impl}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        return cls(model, processor, device)

    def _build_messages(self, image_path: str, prompt_text: str) -> list:
        """Build chat messages with image for Med-Gemma."""
        image = Image.open(image_path).convert("RGB")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def prepare_inputs(self, question: str, image_path: str,
                       prompt_text: str) -> dict:
        messages = self._build_messages(image_path, prompt_text)
        image = Image.open(image_path).convert("RGB")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        prefill_kwargs = {}
        if "pixel_values" in inputs:
            prefill_kwargs["pixel_values"] = inputs["pixel_values"]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "prefill_kwargs": prefill_kwargs,
            "prompt_len": inputs["input_ids"].shape[1],
        }

    def prepare_generate_inputs(self, question: str, image_path: str,
                                prompt_text: str) -> dict:
        messages = self._build_messages(image_path, prompt_text)
        image = Image.open(image_path).convert("RGB")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        result = {k: v for k, v in inputs.items()}
        result["prompt_len"] = inputs["input_ids"].shape[1]
        return result


# ─── Factory ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen":     QwenVLAdapter,
    "llava":    LlavaMedAdapter,
    "medgemma": MedGemmaAdapter,
}


def _detect_type(model_name: str) -> str:
    """Auto-detect model type from model name."""
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "qwen"
    if "llava" in name_lower:
        return "llava"
    if "medgemma" in name_lower or "gemma" in name_lower:
        return "medgemma"
    raise ValueError(
        f"Cannot auto-detect model type for '{model_name}'. "
        f"Use --model-type with one of: {list(MODEL_REGISTRY.keys())}"
    )


def create_adapter(model_name: str, dtype: torch.dtype, device: str,
                   model_type: str = None) -> VLMAdapter:
    """Create a VLM adapter for the given model.

    Args:
        model_name: HuggingFace model name or local path.
        dtype: Torch dtype for model weights.
        device: Target device ("cuda" or "cpu").
        model_type: Explicit model type override. If None, auto-detects
                     from model_name.
    """
    if model_type is None:
        model_type = _detect_type(model_name)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    print(f"Using adapter: {MODEL_REGISTRY[model_type].__name__}")
    return MODEL_REGISTRY[model_type].load(model_name, dtype, device)
