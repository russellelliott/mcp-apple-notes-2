#!/usr/bin/env python3
"""Load a BERTopic model saved with safetensors and print representative documents.

Usage examples:
  python load_topic_model.py                # prints representative docs for all topics
  python load_topic_model.py --topic 12     # prints for topic 12
  python load_topic_model.py --topic 12 --top-n 3
  python load_topic_model.py --output-file out.json
"""
from pathlib import Path
import argparse
import json
from bertopic import BERTopic


def load_model(model_dir: Path):
    model = BERTopic.load(model_dir)
    return model


def print_all_representative_docs(model, top_n=None, skip_negative=True):
    rep = getattr(model, "representative_docs_", None)
    if rep is None:
        print("No `representative_docs_` attribute found on the model.")
        return {}

    out = {}
    for topic, docs in rep.items():
        if skip_negative and topic == -1:
            continue
        out[topic] = docs if top_n is None else docs[:top_n]
        print(f"--- Topic {topic} ({len(out[topic])} docs) ---")
        for i, d in enumerate(out[topic], 1):
            print(f"{i}. {d}")
        print()

    return out


def print_topic_docs(model, topic: int, top_n: int = 5):
    try:
        docs = model.get_representative_docs(topic=topic)
    except Exception:
        docs = None
    if not docs:
        print(f"No representative docs found for topic {topic}")
        return {topic: []}

    docs = docs[:top_n]
    print(f"Representative docs for topic {topic}:")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d}")
    print()
    return {topic: docs}


def main():
    p = argparse.ArgumentParser(description="Load BERTopic model and print representative docs")
    p.add_argument("model_dir", nargs="?", default=str(Path.home() / ".mcp-apple-notes" / "bertopic_model"), help="Path to saved BERTopic model directory")
    p.add_argument("--topic", type=int, help="Specific topic id to fetch")
    p.add_argument("--top-n", type=int, default=5, help="Number of representative docs to show per topic")
    p.add_argument("--output-file", help="Optional JSON file to write representative docs mapping")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return

    print(f"🔁 Loading BERTopic model from {model_dir}...")
    model = load_model(model_dir)

    results = {}
    if args.topic is not None:
        results = print_topic_docs(model, args.topic, top_n=args.top_n)
    else:
        results = print_all_representative_docs(model, top_n=args.top_n)

    if args.output_file:
        out_path = Path(args.output_file)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote representative docs to {out_path}")


if __name__ == "__main__":
    main()
