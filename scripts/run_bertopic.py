import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import re
from pathlib import Path
from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, BaseRepresentation
import ollama
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

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

def clean_note_content(text):
    """Sanitizes raw note text by removing non-textual data."""
    if not isinstance(text, str): return ""
    text = re.sub(r'data:image\/[a-zA-Z]+;base64,[^\s"\'\)]+', '[IMAGE_REMOVED]', text)
    text = re.sub(r'!\[.*?\]\([^\)]{100,}\)', '[IMAGE_REMOVED]', text)
    text = re.sub(r'\S{100,}', '[LONG_DATA_REMOVED]', text)
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

# --- 2. DATA LOADING & BACKUP ---
db = lancedb.connect(DB_PATH)
print(f"📥 Connecting to LanceDB at {DB_PATH}...")

# Create backup before any operations
backup_lancedb_table(db, TABLE_NAME)

table = db.open_table(TABLE_NAME)
df = table.to_pandas()

print(f"🧹 Cleaning {len(df)} chunks of binary data...")
df['clean_chunk_content'] = df['chunk_content'].apply(clean_note_content)
docs = df['clean_chunk_content'].fillna("").tolist()

# Extract vectors already stored in LanceDB
vectors = np.vstack(df['vector'].values)

# --- 3. THE "NO-HARDCODE" CLUSTERING ENGINE ---

# max_df=0.1 calculates stopwords automatically based on your specific life/notes.
# If "UCSC" or "Notes" appears in >10% of documents, it is mathematically ignored.
vectorizer_model = CountVectorizer(max_df=0.1, min_df=2, stop_words="english")

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
    "Label": Ollama(model="phi3:3.8b-mini-128k-instruct-q4_K_M", prompt=label_prompt),
    "Summary": Ollama(model="phi3:3.8b-mini-128k-instruct-q4_K_M", prompt=summary_prompt),
    "KeyBERT": KeyBERTInspired()
}

topic_model = BERTopic(
    embedding_model=None, # Use pre-computed LanceDB vectors
    umap_model=UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True),
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    calculate_probabilities=True,
    verbose=True
)

# --- 4. EXECUTION & REFINEMENT ---
print("🚀 Running semantic clustering (BERTopic)...")
topics, probs = topic_model.fit_transform(docs, embeddings=vectors)

# Fix High-Confidence Outliers
# This solves the issue where high-confidence notes are left as outliers (-1)
print("🎯 Refining outliers using mathematical probability...")
new_topics = topic_model.reduce_outliers(docs, topics, strategy="probabilities")
topic_model.update_topics(docs, topics=new_topics)

# --- 5. SCHEMA MAPPING & PERSISTENCE ---
topic_info = topic_model.get_topic_info()

# Map the Cluster ID (int) to the Label (string) and Summary (string)
# We convert the Phi-3 outputs to clean strings
label_map = dict(zip(topic_info.Topic, topic_info.Label.apply(lambda x: str(x[0]))))
summary_map = dict(zip(topic_info.Topic, topic_info.Summary.apply(lambda x: str(x[0]))))

# Calculate Confidence Scores
confidences = [str(round(np.max(p), 3)) if np.max(p) > 0 else "0.0" for p in probs]

# Sync with your PyArrow Schema
df['cluster_id'] = [str(t) for t in new_topics]
df['cluster_label'] = df['cluster_id'].astype(int).map(label_map)
df['cluster_summary'] = df['cluster_id'].astype(int).map(summary_map)
df['cluster_confidence'] = confidences
df['last_clustered'] = datetime.now().isoformat()

# Ensure we only write back the columns that exist in the schema
schema_columns = ['title', 'content', 'creation_date', 'modification_date', 'chunk_index',
       'total_chunks', 'chunk_content', 'clean_chunk_content', 'vector', 'cluster_id',
       'cluster_label', 'cluster_confidence', 'cluster_summary',
       'last_clustered']
df = df[schema_columns]

print(f"💾 Overwriting table '{TABLE_NAME}' with clustered data...")
db.create_table(TABLE_NAME, data=df, mode="overwrite")

print(f"✨ Success! Identified {len(topic_info)-1} semantically distinct clusters.")