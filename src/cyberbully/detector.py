"""High-level cyberbullying detection engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from cyberbully.downloader import (
    download_model,
    get_default_cache_dir,
    is_model_directory_valid,
)
from cyberbully.model import MultiTaskClassifier
from cyberbully.preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.23  # Optimized validation threshold from training run
DEFAULT_MAX_LEN = 128


@dataclass
class PredictionResult:
    """Structured result of a cyberbullying classification query."""

    text: str
    cleaned_text: str
    is_cyberbullying: bool
    label: str
    score: float
    confidence: float
    threshold: float
    category: str
    category_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a standard dictionary."""
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def find_model_path(preferred_path: str | Path | None = None) -> Path | None:
    """Search candidate locations for a valid model directory."""
    candidates: list[Path] = []
    
    if preferred_path:
        candidates.append(Path(preferred_path))

    env_path = os.environ.get("CYBERBULLY_MODEL_DIR")
    if env_path:
        candidates.append(Path(env_path))

    cwd = Path.cwd()
    candidates.extend([
        cwd / "models" / "cyberbully_v0.1_run",
        cwd / "models",
        cwd / "cyberbully_v0.1_run",
    ])

    pkg_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        pkg_root / "models" / "cyberbully_v0.1_run",
        pkg_root / "models",
    ])

    cache_dir = get_default_cache_dir() / "cyberbully_v0.1_run"
    candidates.append(cache_dir)

    for cand in candidates:
        if is_model_directory_valid(cand):
            return cand.resolve()

    return None


class CyberbullyingDetector:
    """Primary inference engine for cyberbullying text classification."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float | None = None,
        device: str | torch.device | None = None,
        auto_download: bool = True,
    ):
        """Initialize the detector and load weights into memory.
        
        Args:
            model_dir: Directory containing model checkpoint, configs, and tokenizer.
            threshold: Decision threshold for cyberbullying probability (defaults to tuned value in config).
            device: Target device ('cuda', 'cpu', or torch.device). Auto-detected if None.
            auto_download: Whether to automatically download weights if missing locally.
        """
        resolved_dir = find_model_path(model_dir)

        if resolved_dir is None:
            if auto_download:
                logger.info("Local model not found. Starting automatic download...")
                resolved_dir = download_model()
            else:
                raise FileNotFoundError(
                    "Cyberbullying model directory not found. Please provide a path with model_dir, "
                    "set CYBERBULLY_MODEL_DIR, or initialize with auto_download=True."
                )

        self.model_dir = Path(resolved_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load run config
        config_path = self.model_dir / "run_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.run_config = json.load(f)

        self.categories: list[str] = self.run_config.get("categories", [])
        self.max_len: int = int(self.run_config.get("max_len", DEFAULT_MAX_LEN))
        
        saved_thr = float(self.run_config.get("threshold", DEFAULT_THRESHOLD))
        self.default_threshold: float = threshold if threshold is not None else saved_thr

        # Load Tokenizer
        tok_path = self.model_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

        # Load Model
        encoder_config_path = self.model_dir / "encoder_config"
        self.model = MultiTaskClassifier(
            model_name_or_config_dir=encoder_config_path,
            n_categories=len(self.categories),
            pretrained=False,
        )

        weights_path = self.model_dir / "best_model.pt"
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        texts: str | Sequence[str],
        threshold: float | None = None,
        batch_size: int = 32,
    ) -> PredictionResult | list[PredictionResult]:
        """Classify one or more input texts.
        
        Args:
            texts: A single text string or a list/sequence of texts.
            threshold: Optional threshold override for this prediction.
            batch_size: Batch size for tokenization and inference.
            
        Returns:
            A PredictionResult (if single text string passed) or a list of PredictionResult.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not text_list:
            return [] if not is_single else None  # type: ignore

        thr = threshold if threshold is not None else self.default_threshold
        cleaned_list = [clean_text(t) for t in text_list]
        results: list[PredictionResult] = []

        for i in range(0, len(cleaned_list), batch_size):
            batch_clean = cleaned_list[i : i + batch_size]
            batch_orig = text_list[i : i + batch_size]

            encoded = self.tokenizer(
                batch_clean,
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            bin_logits, cat_logits = self.model(input_ids, attention_mask)
            
            bin_probs = torch.softmax(bin_logits.float(), dim=-1).cpu().numpy()
            cat_probs = torch.softmax(cat_logits.float(), dim=-1).cpu().numpy()

            for orig_text, clean_t, b_prob, c_prob in zip(batch_orig, batch_clean, bin_probs, cat_probs):
                prob_bully = float(b_prob[1])
                is_bully = bool(prob_bully >= thr)
                label = "cyberbullying" if is_bully else "not_cyberbullying"
                confidence = prob_bully if is_bully else (1.0 - prob_bully)

                # Map category scores
                cat_dict: dict[str, float] = {}
                for idx, cname in enumerate(self.categories):
                    cat_dict[cname] = float(c_prob[idx])
                
                top_cat = self.categories[int(c_prob.argmax())] if self.categories else "unknown"

                # Sort top 5 category probabilities
                sorted_cats = dict(
                    sorted(cat_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
                )

                results.append(
                    PredictionResult(
                        text=orig_text,
                        cleaned_text=clean_t,
                        is_cyberbullying=is_bully,
                        label=label,
                        score=prob_bully,
                        confidence=confidence,
                        threshold=thr,
                        category=top_cat,
                        category_scores=sorted_cats,
                    )
                )

        return results[0] if is_single else results

    def is_cyberbullying(self, text: str, threshold: float | None = None) -> bool:
        """Convenience method returning True if text is classified as cyberbullying."""
        res = self.predict(text, threshold=threshold)
        return bool(res.is_cyberbullying)  # type: ignore

    def explain(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        """Generate human-readable breakdown and category diagnostics."""
        res: PredictionResult = self.predict(text, threshold=threshold)  # type: ignore
        return {
            "text": res.text,
            "cleaned_text": res.cleaned_text,
            "prediction": res.label,
            "cyberbullying_score": f"{res.score:.4f}",
            "threshold": f"{res.threshold:.4f}",
            "primary_intent_category": res.category,
            "top_categories": res.category_scores,
            "assessment": (
                f"Flagged as cyberbullying ({res.score:.1%} probability exceeds threshold {res.threshold:.1%})"
                if res.is_cyberbullying
                else f"Safe / non-cyberbullying ({res.score:.1%} probability below threshold {res.threshold:.1%})"
            ),
        }
