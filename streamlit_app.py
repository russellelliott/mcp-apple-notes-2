import streamlit as st
import lancedb
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from pathlib import Path
import random
import distinctipy
from colorsys import rgb_to_hsv

# Configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"

st.set_page_config(layout="wide", page_title="Apple Notes Clusters")

@st.cache_resource
def get_db_connection():
    try:
        db = lancedb.connect(str(DB_PATH))
        return db.open_table(TABLE_NAME)
    except Exception as e:
        st.error(f"Error connecting to database at {DB_PATH}: {e}")
        return None

@st.cache_data
def load_data():
    table = get_db_connection()
    if not table:
        return pd.DataFrame()
    
    # Load data into pandas DataFrame
    df = table.to_pandas()
    return df

@st.cache_data
def compute_umap(df):
    if df.empty or 'vector' not in df.columns:
        return df
    
    # Extract embeddings
    # Assuming 'vector' column contains the embeddings as lists or arrays
    embeddings = np.stack(df['vector'].values)
    
    # Initialize UMAP reducer
    # Matching sizing parameters from run_bertopic.py where possible
    # n_components is 2 here for visualization (vs 5 for clustering)
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42,
        n_neighbors=30,
        min_dist=0.0,
        metric='cosine'
    )
    
    # Fit and transform
    projections = reducer.fit_transform(embeddings)
    
    # Add coordinates to dataframe
    df['umap_x'] = projections[:, 0]
    df['umap_y'] = projections[:, 1]
    
    return df

def main():
    # Compact header
    with st.spinner("Loading data..."):
        df = load_data()
        
    if df.empty:
        st.warning("No data found.")
        return

    n_chunks = len(df)
    n_clusters = df['cluster_label'].nunique()
    
    # Use columns to make header compact
    c1, c2 = st.columns([3, 1])
    c1.subheader("Apple Notes Clusters")
    c2.caption(f"{n_chunks} chunks | {n_clusters} clusters")

    if 'vector' not in df.columns:
        st.error("Vector column not found in data.")
        return

    with st.spinner("Computing UMAP projections..."):
        df_viz = compute_umap(df)

    # Prepare tooltip columns
    hover_data = {
        'title': True,
        'creation_date': True,
        'modification_date': True,
        'cluster_label': True, # Show cluster label in tooltip
        'umap_x': False,
        'umap_y': False,
        'cluster_id': False
    }

    # Handle colors
    # We want a random color for each cluster, but consistent across refreshes implicitly by using cluster_id/label
    # Plotly handles color mapping automatically if we specify the color column.
    
    # Fill NaN cluster labels if any
    if 'cluster_label' in df_viz.columns:
        df_viz['cluster_label'] = df_viz['cluster_label'].fillna('Unclustered')
    else:
        df_viz['cluster_label'] = 'Unknown'

    # --- Distinctipy Color Generation Logic ---
    # 1. Get unique clusters
    unique_clusters = sorted(df_viz['cluster_label'].unique())
    n_clusters = len(unique_clusters)
    
    # 2. Calculate centroids for each cluster to determine "similarity" (spatial proximity)
    cluster_centroids = []
    for label in unique_clusters:
        mask = df_viz['cluster_label'] == label
        centroid_x = df_viz[mask]['umap_x'].mean()
        centroid_y = df_viz[mask]['umap_y'].mean()
        # Use angle for 1D sort (0 to 2pi), simple heuristic for 2D->1D mapping
        angle = np.arctan2(centroid_y, centroid_x)
        cluster_centroids.append({'label': label, 'x': centroid_x, 'y': centroid_y, 'angle': angle})
    
    # 3. Sort clusters by angle so adjacent clusters in the list are spatially adjacent(ish)
    cluster_centroids.sort(key=lambda c: c['angle'])
    sorted_labels = [c['label'] for c in cluster_centroids]
    
    # 4. Generate distinct colors
    colors = distinctipy.get_colors(n_clusters)
    
    # 5. Sort colors by Hue to create a gradient-like transition for similar clusters
    # distinctipy returns (r,g,b) tuples in 0-1 range
    colors.sort(key=lambda rgb: rgb_to_hsv(*rgb)[0])
    
    # 6. Map sorted labels to sorted colors
    # Convert RGB tuples to Hex strings for Plotly
    hex_colors = [distinctipy.get_hex(c) for c in colors]
    color_map = dict(zip(sorted_labels, hex_colors))

    fig = px.scatter(
        df_viz,
        x='umap_x',
        y='umap_y',
        color='cluster_label',
        color_discrete_map=color_map,
        hover_data=['title', 'creation_date', 'modification_date', 'cluster_label'],
        title="Notes Clusters (UMAP Projection)",
        template="plotly_white",
        height=800
    )

    # Customize the layout
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        xaxis_title="UMAP Dimension X",
        yaxis_title="UMAP Dimension Y",
        legend_title_text='Cluster'
    )

    # Remove axes ticks for cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)

    # Show raw data view (optional)
    with st.expander("View Raw Data"):
        st.dataframe(df_viz.drop(columns=['vector', 'umap_x', 'umap_y'], errors='ignore'))

if __name__ == "__main__":
    main()
