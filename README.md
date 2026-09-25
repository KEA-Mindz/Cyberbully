# kea-cyberbully

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/kea-cyberbully/)
[![Python Package](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://pypi.org/project/kea-cyberbully/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)

Production-grade **Cyberbullying Text Detection** powered by transformer neural networks and nuance-aware multi-task learning. Developed by **KEA MINDZ**.

Designed for social media moderation, chat moderation, comment screening, and trust & safety workflows.

---

## Key Features

- **Multi-Task Neural Architecture**: Joint binary classifier (cyberbullying vs. safe) with a 20-way auxiliary intent head (threats, targeted insults, implicit mockery, backhanded compliments, sarcasm among friends, venting, etc.).
- **Context & Nuance Aware**: Distinguishes harmful bullying from benign sarcasm among friends, self-venting, or constructive criticism.
- **Adversarial Robustness**: Handles leetspeak (e.g. `@$$h0le`, `sna7ch`), typo variations, and emoji sentiment without breaking casing or punctuation.
- **Lightweight Package (<15 KB)**: Compliant with PyPI upload limits by decoupling heavy 260MB model checkpoints and providing automatic background caching to `~/.cache/cyberbully/models/`.
- **Flexible Interface**: Simple Python API and a rich interactive CLI.

---

## Installation

### From PyPI

```bash
pip install kea-cyberbully
```

### From Source / Development

```bash
git clone https://github.com/KEA-Mindz/Cyberbully.git
cd Cyberbully
pip install -e .
```

### Optional Dependencies

For running model evaluations and plotting metrics:
```bash
pip install -e ".[eval]"
```

For testing:
```bash
pip install -e ".[dev]"
```

---

## Quickstart (Python API)

### 1. Simple Boolean Check

```python
import cyberbully

# True
print(cyberbully.is_cyberbullying("Go jump off a cliff you loser"))

# False
print(cyberbully.is_cyberbullying("You are a wonderful person!"))
```

### 2. Detailed Prediction

```python
import cyberbully

result = cyberbully.predict("Nobody likes you, you are ugly and pathetic")

print(result.label)         # 'cyberbullying'
print(result.score)         # 0.9996
print(result.confidence)    # 0.9996
print(result.category)      # 'targeted_insult_general'
print(result.category_scores)
# {
#   'targeted_insult_general': 0.8227,
#   'threat': 0.1511,
#   ...
# }
```

### 3. Explainability Diagnostics

```python
import cyberbully

explanation = cyberbully.explain("You are so annoying")
print(explanation)
# {
#   'text': 'You are so annoying',
#   'prediction': 'cyberbullying',
#   'cyberbullying_score': '0.9412',
#   'threshold': '0.2300',
#   'primary_intent_category': 'targeted_insult_general',
#   'assessment': 'Flagged as cyberbullying (94.1% probability exceeds threshold 23.0%)'
# }
```

### 4. Custom Model Instance & Batch Processing

```python
from cyberbully import CyberbullyingDetector

# Automatically detects local models/ or downloads to cache on first use
detector = CyberbullyingDetector()

texts = [
    "Have a fantastic day!",
    "You are an idiot.",
    "Can you please help review this PR?"
]

results = detector.predict(texts, batch_size=32)
for res in results:
    print(f"[{res.label.upper():17s}] ({res.score:.3f}) {res.text}")
```

---

## Command-Line Interface (CLI)

The package provides two alias commands: `kea-cyberbully` and `cyberbully`:

### Classify Single or Multiple Texts

```bash
kea-cyberbully "You are a wonderful person"
kea-cyberbully "Nobody likes you" --threshold 0.30
```

### Interactive Mode

Launch an interactive evaluation console:

```bash
kea-cyberbully -i
```

### Batch Processing from CSV or Text Files

```bash
kea-cyberbully --file comments.csv --output flagged_results.csv
```

### Check Model Info & Cache

```bash
kea-cyberbully --info
```

### Pre-download Model Checkpoint

```bash
kea-cyberbully --download
```

---

## Model Weights & Distribution

The trained checkpoint (`best_model.pt`, ~265MB) is decoupled from the PyPI wheel for fast installation:

1. **Local Search**: The detector first inspects:
   - `models/cyberbully_v0.1_run`
   - Explicit path specified via `--model-dir` or `CYBERBULLY_MODEL_DIR`
2. **Cache Fallback**: If not found in local directories, it checks `~/.cache/cyberbully/models/cyberbully_v0.1_run`.
3. **Auto-Download**: If missing, it downloads the checkpoint and tokenizer directly from Google Drive upon first execution.

---

## Repository Structure

```text
Cyberbully/
├── .github/
│   └── workflows/
│       └── release.yaml          # Automated release & PyPI publishing workflow
├── src/
│   └── cyberbully/               # Core Python package
│       ├── __init__.py           # Package exports & convenience API
│       ├── __main__.py           # python -m cyberbully entrypoint
│       ├── cli.py                # Command-line interface
│       ├── detector.py           # CyberbullyingDetector engine
│       ├── downloader.py         # Model downloader & cache management
│       ├── model.py              # PyTorch MultiTaskClassifier architecture
│       └── preprocessing.py      # Adversarial & social media text cleaner
├── models/
│   └── cyberbully_v0.1_run/      # Trained PyTorch model, tokenizer, and config
├── tests/
│   ├── test_detector.py          # Pytest suite for model inference
│   └── test_preprocessing.py     # Unit tests for text normalization
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Packaging exclusions
├── pyproject.toml                # Modern PEP 621 / setuptools packaging
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
