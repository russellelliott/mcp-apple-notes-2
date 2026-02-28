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
    # Prefer to use topic_info if available because it may contain
    # labels, summaries, and representative docs generated during
    # the original training/representation step.
    try:
        topic_info = model.get_topic_info()
    except Exception:
        topic_info = None

    out = {}
    if topic_info is not None and not topic_info.empty:
        # Detect representative docs column if present
        rep_col = None
        cols = [c.lower() for c in topic_info.columns]
        for c in topic_info.columns:
            if 'represent' in c.lower():
                rep_col = c
                break

        for _, row in topic_info.iterrows():
            topic = int(row['Topic'])
            if skip_negative and topic == -1:
                continue

            # Get representative docs from the topic_info row if present,
            # otherwise fall back to model.representative_docs_
            docs = []
            if rep_col and row.get(rep_col) is not None and row.get(rep_col) != '':
                docs_val = row.get(rep_col)
                # It may already be a list, or a stringified list; try to normalize
                if isinstance(docs_val, (list, tuple)):
                    docs = list(docs_val)
                else:
                    try:
                        import ast
                        parsed = ast.literal_eval(docs_val)
                        if isinstance(parsed, (list, tuple)):
                            docs = list(parsed)
                        else:
                            docs = [str(docs_val)]
                    except Exception:
                        docs = [str(docs_val)]
            else:
                rep = getattr(model, "representative_docs_", None)
                if rep and topic in rep:
                    docs = rep[topic]

            docs = docs if top_n is None else docs[:top_n]
            out[topic] = docs

            # Print header with optional label/summary if available
            header = f"--- Topic {topic} ({len(docs)} docs) ---"
            if 'Label' in topic_info.columns:
                header += f"  Label: {row['Label']}"
            elif 'Name' in topic_info.columns:
                header += f"  Name: {row['Name']}"
            if 'Summary' in topic_info.columns:
                header += f"  Summary: {row['Summary']}"

            print(header)
            for i, d in enumerate(docs, 1):
                print(f"{i}. {d}")
            print()

        return out

    # Fallback: use model.representative_docs_
    rep = getattr(model, "representative_docs_", None)
    if rep is None:
        print("No `representative_docs_` attribute found on the model.")
        return {}

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
    # Try to use get_topic_info first to provide a richer output
    try:
        topic_info = model.get_topic_info()
    except Exception:
        topic_info = None

    docs = None
    label = None
    summary = None
    if topic_info is not None and not topic_info.empty:
        row = topic_info[topic_info['Topic'] == topic]
        if not row.empty:
            row = row.iloc[0]
            # Representative docs column detection
            rep_col = None
            for c in topic_info.columns:
                if 'represent' in c.lower():
                    rep_col = c
                    break
            if rep_col and row.get(rep_col) is not None and row.get(rep_col) != '':
                val = row.get(rep_col)
                if isinstance(val, (list, tuple)):
                    docs = list(val)
                else:
                    try:
                        import ast
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple)):
                            docs = list(parsed)
                        else:
                            docs = [str(val)]
                    except Exception:
                        docs = [str(val)]

            # Extract label/summary if present
            if 'Label' in topic_info.columns:
                label = row['Label']
            elif 'Name' in topic_info.columns:
                label = row['Name']
            if 'Summary' in topic_info.columns:
                summary = row['Summary']

    # Fallback to model.get_representative_docs
    if docs is None:
        try:
            docs = model.get_representative_docs(topic=topic)
        except Exception:
            docs = None

    if not docs:
        print(f"No representative docs found for topic {topic}")
        return {topic: []}

    docs = docs[:top_n]
    header = f"Representative docs for topic {topic}:"
    if label is not None:
        header += f"  Label: {label}"
    if summary is not None:
        header += f"  Summary: {summary}"
    print(header)
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
