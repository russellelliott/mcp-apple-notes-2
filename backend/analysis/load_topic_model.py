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
    default = {"title": "-", "creation_date": "-", "modification_date": "-", "cluster_id": None, "cluster_confidence": None, "chunk_index": None}
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
                "cluster_id": str(row.get('cluster_id')) if row.get('cluster_id') is not None else None,
                "cluster_confidence": (float(row.get('cluster_confidence')) if row.get('cluster_confidence') is not None and _is_number(row.get('cluster_confidence')) else None),
                "chunk_index": (int(row.get('chunk_index')) if row.get('chunk_index') is not None and _is_number(row.get('chunk_index')) else None),
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


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _get_keywords_for_topic(model: BERTopic, topic: int) -> list:
    try:
        kws = model.get_topic(topic)
        return [str(w) for w, _ in kws] if kws else []
    except Exception:
        return []


def gather_topics(model: BERTopic, top_n: Optional[int] = None, skip_negative: bool = True, include_prob_top_n: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
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

            # Prefer DB-derived representative docs if available (use cluster_id and cluster_confidence)
            docs = []
            rep_docs_meta = []
            if notes_df is not None and 'cluster_id' in notes_df.columns:
                try:
                    s = notes_df['cluster_id'].fillna('').astype(str)
                    mask = s == str(topic)
                    rows = notes_df[mask]
                    if not rows.empty:
                        # Sort by cluster_confidence if present
                        if 'cluster_confidence' in rows.columns:
                            try:
                                rows = rows.copy()
                                rows['__conf__'] = pd.to_numeric(rows['cluster_confidence'], errors='coerce').fillna(0.0)
                                rows = rows.sort_values('__conf__', ascending=False)
                            except Exception:
                                pass
                        content_col = 'clean_chunk_content' if 'clean_chunk_content' in rows.columns else ('chunk_content' if 'chunk_content' in rows.columns else None)
                        if content_col is not None:
                            docs = rows[content_col].fillna("").astype(str).tolist()
                            for _, r in rows.iterrows():
                                rep_docs_meta.append({
                                    'title': str(r.get('title','-')) if r.get('title') is not None else '-',
                                    'creation_date': str(r.get('creation_date','-')) if r.get('creation_date') is not None else '-',
                                    'modification_date': str(r.get('modification_date','-')) if r.get('modification_date') is not None else '-',
                                    'cluster_id': str(r.get('cluster_id')) if r.get('cluster_id') is not None else None,
                                    'chunk_index': (int(r.get('chunk_index')) if r.get('chunk_index') is not None and _is_number(r.get('chunk_index')) else None),
                                    'cluster_confidence': (float(r.get('cluster_confidence')) if r.get('cluster_confidence') is not None and _is_number(r.get('cluster_confidence')) else None),
                                })
                        else:
                            docs = []
                except Exception:
                    docs = []

            # Fallback to topic_info representative column or model attribute
            if not docs:
                if rep_col and row.get(rep_col) not in (None, ''):
                    docs = _normalize_docs_val(row.get(rep_col))
                else:
                    rep = getattr(model, 'representative_docs_', None)
                    if rep and topic in rep:
                        docs = list(rep[topic])

            if top_n is not None:
                docs = docs[:top_n]

            keywords = _get_keywords_for_topic(model, topic)

            # Try to get label/summary from DB first (Ground Truth)
            db_label = None
            db_summary = None
            if notes_df is not None and 'cluster_id' in notes_df.columns:
                 # Find any row with this cluster_id (it's string in DB, usually)
                 mask = notes_df['cluster_id'].astype(str) == str(topic)
                 if mask.any():
                     first_match = notes_df.loc[mask].iloc[0]
                     if 'cluster_label' in first_match and pd.notna(first_match['cluster_label']):
                         db_label = str(first_match['cluster_label'])
                     if 'cluster_summary' in first_match and pd.notna(first_match['cluster_summary']):
                         db_summary = str(first_match['cluster_summary'])

            # Fallback to model info if DB lookup fails
            model_label = str(row['Label']) if 'Label' in topic_info.columns and row.get('Label') is not None else (str(row['Name']) if 'Name' in topic_info.columns and row.get('Name') is not None else None)
            model_summary = str(row['Summary']) if 'Summary' in topic_info.columns and row.get('Summary') is not None else None

            topic_obj = {
                'label': db_label if db_label is not None else model_label,
                'summary': db_summary if db_summary is not None else model_summary,
                'keywords': keywords,
                'representative_docs': [str(d) for d in docs],
                'representative_docs_meta': rep_docs_meta if rep_docs_meta else [],
            }

            # If we didn't build rep_docs_meta from DB rows, fill by lookup
            if not topic_obj['representative_docs_meta']:
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


def _add_probability_rankings(model: BERTopic, topics: Dict[int, Dict[str, Any]], notes_df: Optional[pd.DataFrame], prob_top_n: int = 5):
    """Compute document-topic probabilities by re-transforming all notes and
    attach top-N highest-probability docs per topic under key `top_probability_docs`
    (with metadata).
    """
    if notes_df is None:
        print("⚠️ Cannot compute probabilities: notes table not available.")
        return

    docs = notes_df['clean_chunk_content'].fillna("").tolist()
    try:
        topics_pred, probs = model.transform(docs)
    except Exception as e:
        print(f"⚠️ Failed to transform docs for probabilities: {e}")
        return

    # Map probs columns to topic ids using topic_info ordering
    try:
        topic_info = model.get_topic_info()
        topic_order = list(topic_info['Topic'])
    except Exception:
        # Fallback: try to infer from topics dict keys sorted
        topic_order = sorted(topics.keys())

    # For each topic, collect (doc_index, prob) pairs
    for t in list(topics.keys()):
        if t not in topic_order:
            # topic may be missing in order; skip
            continue
        idx = topic_order.index(t)
        doc_probs = []
        for i, p_row in enumerate(probs):
            try:
                prob_val = float(p_row[idx])
            except Exception:
                prob_val = 0.0
            doc_probs.append((i, prob_val))

        # sort by probability desc and take top N
        doc_probs.sort(key=lambda x: x[1], reverse=True)
        top_docs = []
        # Build a mapping of doc_text -> probability for later lookup
        doc_prob_map = {}
        for doc_idx, prob_val in doc_probs:
            doc_text = docs[doc_idx]
            doc_prob_map[str(doc_text)] = float(prob_val)

        for doc_idx, prob_val in doc_probs[:prob_top_n]:
            doc_text = docs[doc_idx]
            meta = _find_metadata_for_doc(doc_text, notes_df)
            meta_entry = {"probability": float(prob_val), "doc_text_snippet": (doc_text[:200] if doc_text else '')}
            meta_entry.update(meta)
            top_docs.append(meta_entry)

        topics[t]['top_probability_docs'] = top_docs

        # Annotate existing representative_docs_meta entries with their probability if available
        rep_docs = topics[t].get('representative_docs', [])
        rep_metas = topics[t].get('representative_docs_meta', [])
        for i, d in enumerate(rep_docs):
            key = str(d)
            prob = doc_prob_map.get(key)
            if prob is None:
                # try substring match
                for k, v in doc_prob_map.items():
                    if k and key and k in key or key in k:
                        prob = v
                        break
            if i < len(rep_metas):
                rep_metas[i]['probability'] = float(prob) if prob is not None else None
            else:
                # ensure we still store metadata mapping
                topics[t].setdefault('representative_docs_meta', []).append({"title": "-", "creation_date": "-", "modification_date": "-", "probability": (float(prob) if prob is not None else None), "chunk_index": None})



def print_topics_console(topics: Dict[int, Dict[str, Any]]):
    for topic, obj in topics.items():
        header = f"--- Topic {topic} ("
        if 'representative_docs' in obj:
             header += f"{len(obj['representative_docs'])} docs) ---"
        else:
             header += "0 docs) ---"
             
        if obj.get('label'):
            header += f"  Label: {obj['label']}"
        if obj.get('summary'):
            header += f"  Summary: {obj['summary']}"
        if obj.get('keywords'):
            header += f"  Keywords: {', '.join(obj['keywords'][:5])}"
        print(header)
        if 'representative_docs_meta' in obj:
            for i, meta in enumerate(obj['representative_docs_meta'], 1):
                chunk_idx = meta.get('chunk_index')
                chunk_str = f"  Chunk Index: {chunk_idx}" if chunk_idx is not None else ""
                print(f"{i}. Title: {meta.get('title','-')}  Created: {meta.get('creation_date','-')}  Modified: {meta.get('modification_date','-')}{chunk_str}")
        print()


def main():
    p = argparse.ArgumentParser(description='Load BERTopic model and export topics')
    p.add_argument('model_dir', nargs='?', default=str(Path.home() / '.mcp-apple-notes' / 'bertopic_model'))
    p.add_argument('--topic', type=int, help='Specific topic id to fetch')
    p.add_argument('--top-n', type=int, default=None, help='Number of representative docs to include per topic')
    p.add_argument('--output-file', help='Optional JSON file to write representative docs mapping')
    p.add_argument('--prob-top-n', type=int, default=None, help='Include top-N docs by model probability for each topic')
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return

    model = load_model(model_dir)

    topics = gather_topics(model, top_n=args.top_n)
    notes_df = load_notes_df()
    if args.prob_top_n:
        _add_probability_rankings(model, topics, notes_df, prob_top_n=args.prob_top_n)
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
