#!/usr/bin/env python3
"""Load a BERTopic model saved with safetensors and export topic keywords
and representative document metadata to JSON or stdout.

Behavior:
- Prefer `model.get_topic_info()` for labels/summaries/rep-docs when available.
- Fallback to `model.representative_docs_` or `model.get_representative_docs()`.
- For each representative doc, look up `title`, `creation_date`, `modification_date`
  from the LanceDB `notes` table located at `~/.mcp-apple-notes/data`.
"""
from pathlib import Path
import argparse
import json
import ast
from typing import Optional, Dict, Any

import lancedb
import pandas as pd
from bertopic import BERTopic


def load_model(model_dir: Path) -> BERTopic:
    return BERTopic.load(model_dir)


def load_notes_df() -> Optional[pd.DataFrame]:
    try:
        DATA_DIR = Path.home() / ".mcp-apple-notes"
        DB_PATH = DATA_DIR / "data"
        db = lancedb.connect(DB_PATH)
        table = db.open_table("notes")
        return table.to_pandas()
    except Exception:
        return None


def _find_metadata_for_doc(doc_text: str, notes_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    default = {"title": "-", "creation_date": "-", "modification_date": "-"}
    if notes_df is None:
        return default

    if 'clean_chunk_content' in notes_df.columns:
        col = 'clean_chunk_content'
    elif 'chunk_content' in notes_df.columns:
        col = 'chunk_content'
    else:
        return default

    try:
        s = notes_df[col].fillna("")
        mask = s == doc_text
        matches = notes_df[mask]
        if matches.empty:
            mask2 = s.str.contains(str(doc_text), na=False)
            matches = notes_df[mask2]
        if not matches.empty:
            row = matches.iloc[0]
            return {
                "title": str(row.get('title', '-')) if row.get('title') is not None else '-',
                "creation_date": str(row.get('creation_date', '-')) if row.get('creation_date') is not None else '-',
                "modification_date": str(row.get('modification_date', '-')) if row.get('modification_date') is not None else '-',
            }
    except Exception:
        return default

    return default


def _normalize_docs_val(val) -> list:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass
        return [val]
    return [str(val)]


def _get_keywords_for_topic(model: BERTopic, topic: int) -> list:
    try:
        kws = model.get_topic(topic)
        return [str(w) for w, _ in kws] if kws else []
    except Exception:
        return []


def gather_topics(model: BERTopic, top_n: Optional[int] = None, skip_negative: bool = True) -> Dict[int, Dict[str, Any]]:
    notes_df = load_notes_df()
    out: Dict[int, Dict[str, Any]] = {}

    try:
        topic_info = model.get_topic_info()
    except Exception:
        topic_info = None

    if topic_info is not None and not topic_info.empty:
        # detect representative docs column name if present
        rep_col = None
        for c in topic_info.columns:
            if 'represent' in c.lower():
                rep_col = c
                break

        for _, row in topic_info.iterrows():
            topic = int(row['Topic'])
            if skip_negative and topic == -1:
                continue

            docs = []
            if rep_col and row.get(rep_col) not in (None, ''):
                docs = _normalize_docs_val(row.get(rep_col))
            else:
                rep = getattr(model, 'representative_docs_', None)
                if rep and topic in rep:
                    docs = list(rep[topic])

            if top_n is not None:
                docs = docs[:top_n]

            keywords = _get_keywords_for_topic(model, topic)

            topic_obj = {
                'label': str(row['Label']) if 'Label' in topic_info.columns and row.get('Label') is not None else (str(row['Name']) if 'Name' in topic_info.columns and row.get('Name') is not None else None),
                'summary': str(row['Summary']) if 'Summary' in topic_info.columns and row.get('Summary') is not None else None,
                'keywords': keywords,
                'representative_docs': [str(d) for d in docs],
                'representative_docs_meta': [],
            }

            for d in topic_obj['representative_docs']:
                topic_obj['representative_docs_meta'].append(_find_metadata_for_doc(d, notes_df))

            out[topic] = topic_obj

        return out

    # fallback to representative_docs_ mapping
    rep = getattr(model, 'representative_docs_', None)
    if rep is None:
        return out

    for topic, docs in rep.items():
        if skip_negative and topic == -1:
            continue
        docs = list(docs)
        if top_n is not None:
            docs = docs[:top_n]
        keywords = _get_keywords_for_topic(model, topic)
        topic_obj = {
            'label': None,
            'summary': None,
            'keywords': keywords,
            'representative_docs': [str(d) for d in docs],
            'representative_docs_meta': [],
        }
        for d in topic_obj['representative_docs']:
            topic_obj['representative_docs_meta'].append(_find_metadata_for_doc(d, notes_df))
        out[topic] = topic_obj

    return out


def print_topics_console(topics: Dict[int, Dict[str, Any]]):
    for topic, obj in topics.items():
        header = f"--- Topic {topic} ({len(obj['representative_docs'])} docs) ---"
        if obj.get('label'):
            header += f"  Label: {obj['label']}"
        if obj.get('summary'):
            header += f"  Summary: {obj['summary']}"
        if obj.get('keywords'):
            header += f"  Keywords: {', '.join(obj['keywords'][:5])}"
        print(header)
        for i, meta in enumerate(obj['representative_docs_meta'], 1):
            print(f"{i}. Title: {meta.get('title','-')}  Created: {meta.get('creation_date','-')}  Modified: {meta.get('modification_date','-')}")
        print()


def main():
    p = argparse.ArgumentParser(description='Load BERTopic model and export topics')
    p.add_argument('model_dir', nargs='?', default=str(Path.home() / '.mcp-apple-notes' / 'bertopic_model'))
    p.add_argument('--topic', type=int, help='Specific topic id to fetch')
    p.add_argument('--top-n', type=int, default=None, help='Number of representative docs to include per topic')
    p.add_argument('--output-file', help='Optional JSON file to write representative docs mapping')
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return

    model = load_model(model_dir)

    topics = gather_topics(model, top_n=args.top_n)
    if args.topic is not None:
        topics = {args.topic: topics.get(args.topic, {'representative_docs': [], 'representative_docs_meta': []})}

    print_topics_console(topics)

    if args.output_file:
        out_path = Path(args.output_file)
        # Convert integer keys to strings for JSON compatibility
        serializable = {str(k): v for k, v in topics.items()}
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Wrote representative docs to {out_path}")


if __name__ == '__main__':
    main()
