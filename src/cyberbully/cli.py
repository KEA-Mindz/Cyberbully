"""Command-line interface for the cyberbully package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cyberbully import __version__
from cyberbully.detector import CyberbullyingDetector, find_model_path
from cyberbully.downloader import download_model, get_default_cache_dir


def print_single_result(res, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(res.to_dict(), indent=2))
        return

    badge = "[CYBERBULLYING]" if res.is_cyberbullying else "[SAFE / NOT BULLYING]"
    print("\n" + "=" * 50)
    print(f"Prediction : {badge}")
    print(f"Score      : {res.score:.4f} (threshold: {res.threshold:.4f})")
    print(f"Confidence : {res.confidence * 100:.1f}%")
    print(f"Top Intent : {res.category}")
    print(f"Input Text : {res.text}")
    if res.text != res.cleaned_text:
        print(f"Cleaned    : {res.cleaned_text}")
    print("Top Categories:")
    for cat, prob in res.category_scores.items():
        print(f"  - {cat:32s}: {prob:.4f}")
    print("=" * 50)


def run_interactive(detector: CyberbullyingDetector, as_json: bool = False) -> None:
    print("\nCyberbullying Detector Interactive Console (Ctrl+C or 'exit' to quit)")
    print("-" * 65)
    while True:
        try:
            line = input("\nEnter text > ").strip()
            if not line:
                continue
            if line.lower() in {"exit", "quit", "q"}:
                break
            res = detector.predict(line)
            print_single_result(res, as_json=as_json)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cyberbully",
        description="Production-grade cyberbullying detection powered by transformer classifiers.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text string(s) to analyze. If omitted and --interactive/--file not specified, prompts interactively.",
    )
    parser.add_argument(
        "--model-dir",
        "-m",
        default=None,
        help="Custom path to model directory containing best_model.pt and configs.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Decision threshold override (default: tuned validation threshold).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        choices=["cuda", "cpu"],
        help="Target device for inference (default: auto-detected).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Input text file (one line per text) or CSV file with a 'text' column.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path for batch evaluation results.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive evaluation console.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model weights to local cache and exit.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display resolved model paths, cache directory, and exit.",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results in JSON format.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    if args.info:
        model_path = find_model_path(args.model_dir)
        cache_path = get_default_cache_dir()
        print(f"cyberbully package version: {__version__}")
        print(f"Default Cache Directory   : {cache_path}")
        print(f"Active Model Directory    : {model_path or 'Not found (auto-download required)'}")
        return 0

    if args.download:
        dest = download_model(target_dir=args.model_dir, force=True)
        print(f"Successfully downloaded model to: {dest}")
        return 0

    try:
        detector = CyberbullyingDetector(
            model_dir=args.model_dir,
            threshold=args.threshold,
            device=args.device,
            auto_download=True,
        )
    except Exception as e:
        sys.stderr.write(f"Error initializing detector: {e}\n")
        return 1

    if args.file:
        if not args.file.is_file():
            sys.stderr.write(f"Input file not found: {args.file}\n")
            return 1
        
        texts = []
        if args.file.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(args.file)
            col = "text" if "text" in df.columns else df.columns[0]
            texts = df[col].astype(str).tolist()
        else:
            with open(args.file, "r", encoding="utf-8") as handle:
                texts = [line.strip() for line in handle if line.strip()]

        print(f"Processing {len(texts):,} items from {args.file}...")
        results = detector.predict(texts, threshold=args.threshold)
        
        if args.output:
            import pandas as pd
            records = [r.to_dict() for r in results]  # type: ignore
            pd.DataFrame(records).to_csv(args.output, index=False)
            print(f"Saved results to: {args.output}")
        else:
            for r in results:  # type: ignore
                print_single_result(r, as_json=args.json)
        return 0

    if args.interactive or not args.text:
        run_interactive(detector, as_json=args.json)
        return 0

    results = detector.predict(args.text, threshold=args.threshold)
    if isinstance(results, list):
        for r in results:
            print_single_result(r, as_json=args.json)
    else:
        print_single_result(results, as_json=args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
