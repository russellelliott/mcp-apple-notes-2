import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import re
import sys
import json
from collections import Counter
import torch
import html
import time
from pathlib import Path
# Ensure repo root is on sys.path so backend.* imports work when running
# this script directly (not as an installed package).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, BaseRepresentation, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
import ollama
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from backend.scripts.common_words import clean_text
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

class Ollama(BaseRepresentation):
    def __init__(self, model="llama2", prompt=None, tokenizer=None):
        self.model = model
        self.prompt = prompt if prompt else """
        I have a topic that contains the following documents: 
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the above information, can you give a short label of the topic?
        """
        self.tokenizer = tokenizer

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        updated_topics = {}
        for topic, keywords in topics.items():
            # Prepare keywords
            keywords_text = ", ".join([word for word, _ in keywords])
            
            # Prepare documents
            if topic in topic_model.representative_docs_:
                representative_docs = topic_model.representative_docs_[topic]
            else:
                # Fallback if no representative docs found (e.g. for outliers or small topics)
                selection = documents.loc[documents['Topic'] == topic, 'Document']
                if len(selection) > 0:
                    representative_docs = selection.sample(min(5, len(selection))).tolist()
                else:
                    representative_docs = []

            # Limit to top 4 representative docs to focus on centroids
            representative_docs = representative_docs[:4]

            docs_text = "\n".join(representative_docs)
            
            # Generate prompt
            prompt = self.prompt.replace("[KEYWORDS]", keywords_text).replace("[DOCUMENTS]", docs_text)
            
            # Call Ollama
            try:
                response = ollama.generate(model=self.model, prompt=prompt)
                label = response['response'].strip()
                # Return as a list of (word, weight) tuples. 
                # We put the label as the single "word" with weight 1.
                updated_topics[topic] = [(label, 1)]
            except Exception as e:
                print(f"Error generating topic label for topic {topic}: {e}")
                updated_topics[topic] = keywords

        return updated_topics

# --- 1. CONFIGURATION & CLEANING ---
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
MODEL_DIR = Path.home() / ".mcp-apple-notes" / "bertopic_model"

def clean_note_content(text):
    """Sanitizes raw note text by removing non-textual data."""
    if not isinstance(text, str): return ""
    # Remove HTML noise first
    text = html.unescape(text)
    # Remove URLs entirely (this handles the https/www noise)
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[^\s"\'\)]+', '', text)
    text = re.sub(r'!\[.*?\]\([^\)]{100,}\)', '', text)
    text = re.sub(r'\S{100,}', '', text)
    return text

def backup_lancedb_table(db, table_name, verbose=True):
    """Creates a timestamped backup of a specific LanceDB table."""
    try:
        table = db.open_table(table_name)
        timestamp = int(datetime.now().timestamp())
        backup_name = f"{table_name}_backup_{timestamp}"
        
        df = table.to_pandas()
        db.create_table(backup_name, df)
        
        if verbose:
            print(f"   📦 Created backup table: {backup_name}")
        return backup_name
    except Exception as e:
        print(f"   ❌ Failed to create backup: {e}")
        return None

# --- Execution timing start ---
start_time = time.time()

db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)
df = table.to_pandas()
df['clean_chunk_content'] = df['chunk_content'].apply(clean_note_content)
docs = df['clean_chunk_content'].fillna("").tolist()

print(f"📥 Connecting to LanceDB at {DB_PATH}...")

# Create backup before any operations
backup_lancedb_table(db, TABLE_NAME)

print(f"🧹 Cleaning {len(df)} chunks of binary data...")
df['clean_chunk_content'] = df['chunk_content'].apply(clean_note_content)
docs = df['clean_chunk_content'].fillna("").tolist()

# Extract vectors already stored in LanceDB
vectors = np.vstack(df['vector'].values)

# --- 3. THE "NO-HARDCODE" CLUSTERING ENGINE ---

# 1. High-Frequency Filtering in the Vectorizer
# We'll compute stop words as any token that appears more than 0.1%
# of the total token count in the corpus using `clean_text` from
# `backend.scripts.common_words`. This list will be passed directly
# to CountVectorizer. `max_df` is set to 1.0 because we're handling
# high-frequency removal via the explicit stop-word list.
THRESHOLD_PROP = 0.001  # 0.1%

# Build word frequency counts from the cleaned chunk content
word_counter = Counter()
total_words = 0
for idx, txt in enumerate(df['clean_chunk_content'].fillna("")):
    if not txt or not isinstance(txt, str):
        continue
    words = clean_text(txt)
    word_counter.update(words)

total_words = sum(word_counter.values())

# Determine stop words: words whose share > THRESHOLD_PROP
stop_words = [w for w, c in word_counter.items() if total_words > 0 and (c / total_words) > THRESHOLD_PROP]

print(f"🔕 Computed {len(stop_words)} high-frequency stop words (>{THRESHOLD_PROP*100:.3f}% of tokens)")

# Remove stop words directly from the content
stop_words_set = set(stop_words)
def remove_stop_words(text):
    if not isinstance(text, str): return ""
    words = text.split()
    return " ".join([w for w in words if w.lower() not in stop_words_set])

df['clean_chunk_content'] = df['clean_chunk_content'].apply(remove_stop_words)
docs = df['clean_chunk_content'].fillna("").tolist()

vectorizer_model = CountVectorizer(
    max_df=1.0,
    min_df=1,
    stop_words=[] # Stop words already removed
)

# 2. Automated c-TF-IDF Reduction
# This tells BERTopic to mathematically down-weight words that appear 
# frequently across ALL topics (like "https", "note", "the").
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# Phi-3 XML Prompts
label_prompt = """<|user|>
I have a group of related notes.
<keywords>
[KEYWORDS]
</keywords>
<sample_docs>
[DOCUMENTS]
</sample_docs>

Task: Provide a concise, professional label (3-5 words) for this cluster. 
Output ONLY the label string. No preamble, no quotes.
<|assistant|>"""

summary_prompt = """<|user|>
I have a group of related notes.
<keywords>
[KEYWORDS]
</keywords>
<sample_docs>
[DOCUMENTS]
</sample_docs>

Task: Provide a one-sentence executive summary explaining the common theme of these notes.
<|assistant|>"""

representation_model = {
    "Main": MaximalMarginalRelevance(diversity=0.3)
}

# Initialize Embedding Model (Required for KeyBERTInspired)
# Device detection
if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("🚀 Using NVIDIA GPU (CUDA)")
else:
    device = "cpu"
    print("⚠️ Using CPU (no GPU acceleration)")

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

topic_model = BERTopic(
    embedding_model=embedding_model,
    # UMAP: n_neighbors=30 (A middle ground between local and global).
    # min_dist=0.0 (Default, packs points tightly)
    umap_model=UMAP(
        n_neighbors=30, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine'
    ),
    hdbscan_model=HDBSCAN(
        min_cluster_size=10,     # REDUCED from 15 - allows smaller clusters
        min_samples=3,           # VERY PERMISSIVE (was 10)
        cluster_selection_method='eom',
        prediction_data=True,
        cluster_selection_epsilon=0.2  # NEW - allows more aggressive merging
    ),
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    calculate_probabilities=True
)

# --- 4. EXECUTION & REFINEMENT ---
print("🚀 Running semantic clustering (BERTopic)...")
topics, probs = topic_model.fit_transform(docs, embeddings=vectors)

hierarchical_topics = topic_model.hierarchical_topics(docs)

# === AGGRESSIVE OUTLIER REASSIGNMENT ===
print("🎯 Aggressively reassigning outliers to nearby clusters...")

# Pass 1: High-confidence outliers (probability-based)
print("  📊 Pass 1: Probability-based reassignment...")
new_topics = topic_model.reduce_outliers(
    docs, 
    topics, 
    probabilities=probs, 
    strategy="probabilities", 
    threshold=0.05  # VERY LOW - captures weak matches
)
outliers_before = sum(1 for t in topics if t == -1)
outliers_after_p1 = sum(1 for t in new_topics if t == -1)
print(f"    ✓ Reassigned {outliers_before - outliers_after_p1} outliers ({outliers_after_p1} remaining)")

topic_model.update_topics(docs, topics=new_topics)

# === MEGA-CLUSTER SPLITTING (>=10% of corpus) ===
print("🧩 Evaluating mega-clusters for sub-splitting...")
total_chunks = len(docs)
mega_cluster_threshold = max(1, int(np.ceil(total_chunks * 0.10)))
topic_sizes = pd.Series(new_topics).value_counts().to_dict()
mega_clusters = [
    int(topic_id)
    for topic_id, size in topic_sizes.items()
    if int(topic_id) >= 0 and int(size) >= mega_cluster_threshold
]

subcluster_lookup = {}
submodel_manifest = {}
submodel_root = MODEL_DIR / "submodels"


def _extract_topic_name(raw_value, fallback: str) -> str:
    """Normalize BERTopic Label/Name values to a single display string."""
    if raw_value is None:
        return fallback
    if isinstance(raw_value, (list, tuple)) and len(raw_value) > 0:
        return str(raw_value[0])
    return str(raw_value)

if mega_clusters:
    print(
        f"  🎯 Found {len(mega_clusters)} mega-cluster(s) with threshold >= {mega_cluster_threshold}/{total_chunks} chunks"
    )
    submodel_root.mkdir(parents=True, exist_ok=True)

    for parent_topic in sorted(mega_clusters):
        parent_indices = np.where(np.array(new_topics) == parent_topic)[0]
        if len(parent_indices) < 2:
            continue

        parent_docs = [docs[i] for i in parent_indices]
        parent_embeddings = vectors[parent_indices]
        print(f"    🔬 Splitting Topic {parent_topic} ({len(parent_docs)} chunks)...")

        if len(parent_docs) < 3:
            print(f"      ⚠️ Skipping Topic {parent_topic}: not enough chunks for stable sub-clustering")
            continue

        local_neighbors = min(30, max(2, len(parent_docs) - 1))
        local_min_cluster_size = max(5, min(20, len(parent_docs) // 20))
        local_min_cluster_size = min(local_min_cluster_size, len(parent_docs))

        sub_topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=UMAP(
                n_neighbors=local_neighbors,
                n_components=5,
                min_dist=0.0,
                metric='cosine'
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=max(2, local_min_cluster_size),
                min_samples=2,
                cluster_selection_method='eom',
                prediction_data=True,
            ),
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            calculate_probabilities=True,
        )

        try:
            sub_topics, _ = sub_topic_model.fit_transform(parent_docs, embeddings=parent_embeddings)
        except Exception as e:
            print(f"      ⚠️ Failed to split Topic {parent_topic}: {e}")
            continue

        sub_topic_info = sub_topic_model.get_topic_info()
        sub_label_map = {}
        if sub_topic_info is not None and not sub_topic_info.empty:
            for _, row in sub_topic_info.iterrows():
                topic_id = int(row.get("Topic", -1))
                if topic_id < 0:
                    continue
                if "Label" in sub_topic_info.columns:
                    sub_label_map[topic_id] = _extract_topic_name(row.get("Label"), f"Subtopic {topic_id}")
                elif "Name" in sub_topic_info.columns:
                    sub_label_map[topic_id] = _extract_topic_name(row.get("Name"), f"Subtopic {topic_id}")
                else:
                    sub_label_map[topic_id] = f"Subtopic {topic_id}"

        # Normalize child ids for display. BERTopic can emit -1 for outliers;
        # we fold these into child 0 so every chunk has a parent.child id.
        child_distribution = Counter()
        for local_idx, child_topic in enumerate(sub_topics):
            normalized_child = int(child_topic) if int(child_topic) >= 0 else 0
            global_idx = int(parent_indices[local_idx])
            subcluster_lookup[global_idx] = {
                "base_topic_id": str(parent_topic),
                "display_topic_id": f"{parent_topic}.{normalized_child}",
                "subcluster_label": sub_label_map.get(normalized_child, f"Subtopic {normalized_child}"),
                "is_split_child": True,
            }
            child_distribution[str(normalized_child)] += 1

        topic_submodel_dir = submodel_root / f"topic_{parent_topic}"
        topic_submodel_dir.mkdir(parents=True, exist_ok=True)
        try:
            sub_topic_model.save(topic_submodel_dir, serialization="safetensors", save_ctfidf=True)
            submodel_manifest[str(parent_topic)] = {
                "parent_topic": parent_topic,
                "threshold": mega_cluster_threshold,
                "chunk_count": int(len(parent_docs)),
                "submodel_path": str(topic_submodel_dir),
                "child_distribution": dict(child_distribution),
            }
            print(f"      ✅ Saved submodel: {topic_submodel_dir}")
        except Exception as e:
            print(f"      ⚠️ Failed to save submodel for Topic {parent_topic}: {e}")

    if submodel_manifest:
        manifest_path = submodel_root / "manifest.json"
        try:
            with manifest_path.open("w", encoding="utf-8") as mf:
                json.dump(submodel_manifest, mf, indent=2)
            print(f"  🗂️ Wrote submodel manifest: {manifest_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to write submodel manifest: {e}")
else:
    print("  ℹ️ No mega-clusters met the 10% threshold.")

# Pass 2: Topic representation similarity (c-TF-IDF based)
print("  📊 Pass 2: c-TF-IDF similarity reassignment...")
new_topics = topic_model.reduce_outliers(
    docs, 
    new_topics, 
    strategy="c-tf-idf",
    threshold=0.05  # VERY LOW
)
outliers_after_p2 = sum(1 for t in new_topics if t == -1)
print(f"    ✓ Reassigned {outliers_after_p1 - outliers_after_p2} more outliers ({outliers_after_p2} remaining)")

topic_model.update_topics(docs, topics=new_topics)

# Pass 3: Embedding distance (cosine similarity in vector space)
print("  📊 Pass 3: Embedding-based reassignment...")
new_topics = topic_model.reduce_outliers(
    docs,
    new_topics,
    strategy="embeddings",
    embeddings=vectors,
    threshold=0.5  # Cosine similarity threshold (0-1, higher = stricter)
)
outliers_after_p3 = sum(1 for t in new_topics if t == -1)
print(f"    ✓ Reassigned {outliers_after_p2 - outliers_after_p3} more outliers ({outliers_after_p3} remaining)")

topic_model.update_topics(docs, topics=new_topics)

# Pass 4: Custom manual reassignment for stubborn outliers
print("  📊 Pass 4: Manual probability check for remaining outliers...")
reassigned_manual = 0
for idx, (topic, prob_dist) in enumerate(zip(new_topics, probs)):
    if topic == -1:  # Still an outlier
        # Get probabilities for all real topics (skip -1 at index 0)
        if len(prob_dist) > 1:
            topic_probs = prob_dist[1:]  # Skip outlier probability
            max_prob = max(topic_probs)
            
            # If ANY cluster has >3% probability, assign to it
            if max_prob > 0.03:  # VERY PERMISSIVE
                best_topic = topic_probs.argmax()
                new_topics[idx] = best_topic
                reassigned_manual += 1

print(f"    ✓ Manually reassigned {reassigned_manual} stubborn outliers")

outliers_final = sum(1 for t in new_topics if t == -1)
total_reassigned = outliers_before - outliers_final
print(f"\n✨ Total outlier reduction: {outliers_before} → {outliers_final} ({total_reassigned} reassigned, {100*(1-outliers_final/outliers_before):.1f}% reduction)")

topic_model.update_topics(docs, topics=new_topics)

# --- 5. SCHEMA MAPPING & PERSISTENCE ---
topic_info = topic_model.get_topic_info()
print(f"DEBUG: topic_info columns: {topic_info.columns.tolist()}")

# Map the Cluster ID (int) to the Label (string) and Summary (string)
# We convert the Phi-3 outputs to clean strings
if 'Label' in topic_info.columns:
    label_map = dict(zip(topic_info['Topic'], topic_info['Label'].apply(lambda x: str(x[0]))))
else:
    print("⚠️ 'Label' column missing. Using default topic names.")
    label_map = dict(zip(topic_info['Topic'], topic_info['Name']))

if 'Summary' in topic_info.columns:
    summary_map = dict(zip(topic_info['Topic'], topic_info['Summary'].apply(lambda x: str(x[0]))))
else:
    print("⚠️ 'Summary' column missing. Using default topic names.")
    summary_map = dict(zip(topic_info['Topic'], topic_info['Name']))

# Calculate Confidence Scores
confidences = [str(round(np.max(p), 3)) if np.max(p) > 0 else "0.0" for p in probs]

# Build base/display cluster mapping.
base_topic_ids = [str(int(t)) for t in new_topics]
display_topic_ids = []
is_split_children = []
for idx, base_topic in enumerate(base_topic_ids):
    mapped = subcluster_lookup.get(idx)
    if mapped:
        display_topic_ids.append(mapped['display_topic_id'])
        is_split_children.append(True)
    else:
        display_topic_ids.append(base_topic)
        is_split_children.append(False)

# Keep retrieval on base topic id, but expose display topic id for UI.
df['base_topic_id'] = base_topic_ids
df['display_topic_id'] = display_topic_ids
df['is_split_child'] = is_split_children

df['cluster_id'] = df['base_topic_id']
df['cluster_label'] = df['base_topic_id'].astype(int).map(label_map)
df['base_cluster_label'] = df['cluster_label']
df['cluster_summary'] = df['base_topic_id'].astype(int).map(summary_map)

split_mask = df['is_split_child']
if split_mask.any():
    split_indices = df.index[split_mask]
    split_labels = []
    for row_idx in split_indices:
        mapped = subcluster_lookup.get(int(row_idx), {})
        split_labels.append(mapped.get('subcluster_label'))
    df.loc[split_indices, 'cluster_label'] = split_labels

df['cluster_label'] = df['cluster_label'].fillna(df['base_cluster_label']).fillna('Unknown')
df['base_cluster_label'] = df['base_cluster_label'].fillna(df['cluster_label']).fillna('Unknown')

df['cluster_confidence'] = confidences
df['last_clustered'] = datetime.now().isoformat()

# Ensure we only write back the columns that exist in the schema
schema_columns = ['title', 'content', 'creation_date', 'modification_date', 'chunk_index',
       'total_chunks', 'chunk_content', 'clean_chunk_content', 'vector',
       'base_topic_id', 'display_topic_id', 'is_split_child', 'cluster_id',
    'cluster_label', 'base_cluster_label', 'cluster_confidence', 'cluster_summary',
       'last_clustered']
df = df[schema_columns]

print(f"💾 Overwriting table '{TABLE_NAME}' with clustered data...")
db.create_table(TABLE_NAME, data=df, mode="overwrite")

print(f"✨ Success! Identified {len(topic_info)-1} semantically distinct clusters.")

# --- 6. SAVE TOPIC MODEL (safetensors) ---
try:
    print(f"💾 Saving BERTopic model to {MODEL_DIR} (safetensors)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)
    print(f"✅ BERTopic model saved.")
except Exception as e:
    print(f"⚠️ Failed to save BERTopic model: {e}")

# --- Print total execution time ---
elapsed = time.time() - start_time
print(f"Total execution time: {elapsed:.2f} seconds")