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
from sentence_transformers import SentenceTransformer
from scripts.search_notes import search_and_combine_results

# Configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

st.set_page_config(layout="wide", page_title="Apple Notes Clusters")

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def get_query_embedding(query):
    model = get_embedding_model()
    return model.encode(query).tolist()

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
    c1, c2, c3 = st.columns([2, 1, 2])
    c1.subheader("Apple Notes Clusters")
    c2.caption(f"{n_chunks} chunks | {n_clusters} clusters")
    
    search_query = c3.text_input("Search notes...", placeholder="Type to search...", label_visibility="collapsed")

    if 'vector' not in df.columns:
        st.error("Vector column not found in data.")
        return

    with st.spinner("Computing UMAP projections..."):
        df_viz = compute_umap(df)

    # Search Logic
    search_results = []
    if search_query:
        with st.spinner("Searching..."):
            table = get_db_connection()
            # We need to pass a function that returns the embedding for the query
            # However, search_and_combine_results expects a callable that takes a string and returns a list of floats
            # But get_query_embedding is cached and takes query.
            # We can just wrap the model's encode method if we wanted, or use our cached function.
            # Let's just use the cached function.
            
            # Note: search_and_combine_results calls compute_query_embedding(query)
            embedding_fn = lambda q: get_query_embedding(q)
            
            results = search_and_combine_results(
                table, 
                search_query, 
                display_limit=50, # Get enough results to be visible
                compute_query_embedding=embedding_fn
            )
            
            # Create a dict of keys for matching results to dataframe rows w/ score
            # Key: title + chunk_index -> score
            search_scores = {}
            for r in results:
                # search_and_combine_results returns logic that extracts fields.
                idx = r.get('_chunk_index', r.get('chunk_index', 0))
                title = r.get('title', '')
                score = r.get('_relevance_score', 0)
                # Ensure we handle potential float/None key issues
                key = f"{title}_{idx}"
                search_scores[key] = score
                
    # Add visual indicator column
    # Ensure chunk_index is int for consistent matching
    df_viz['chunk_index'] = df_viz['chunk_index'].fillna(0).astype(int)
    df_viz['unique_key'] = df_viz['title'].astype(str) + "_" + df_viz['chunk_index'].astype(str)
    
    if search_query:
        df_viz['is_search_result'] = df_viz['unique_key'].isin(search_scores.keys())
        # Map scores slightly inefficiently but safely
        df_viz['search_score'] = df_viz['unique_key'].map(search_scores).fillna(0)
        
        # Sort so search results are drawn on top
        df_viz = df_viz.sort_values('is_search_result', ascending=True)
        
        # Add a symbol column based on search result
        df_viz['marker_symbol'] = np.where(df_viz['is_search_result'], 'diamond', 'circle')
        # MUCH larger size for results
        df_viz['marker_size'] = np.where(df_viz['is_search_result'], 20, 6)
        # Keep opacity high for everyone, user disliked bluriness
        df_viz['opacity'] = 0.8 # Uniform opacity
        
        # We will add a line width column
        df_viz['line_width'] = np.where(df_viz['is_search_result'], 2, 0)
        
    else:
        df_viz['is_search_result'] = False
        df_viz['search_score'] = 0
        df_viz['marker_symbol'] = 'circle'
        df_viz['marker_size'] = 6
        df_viz['opacity'] = 0.8
        df_viz['line_width'] = 0

    # Prepare tooltip columns
    hover_data = {
        'title': True,
        'creation_date': True,
        'modification_date': True,
        'cluster_label': True, # Show cluster label in tooltip
        'search_score': True if search_query else False,
        'umap_x': False,
        'umap_y': False,
        'cluster_id': False,
        'marker_symbol': False,
        'marker_size': False,
        'opacity': False,
        'unique_key': False,
        'line_width': False
    }

    # Handle colors
    # We want a random color for each cluster, but consistent across refreshes implicitly by using cluster_id/label
    # Plotly handles color mapping automatically if we specify the color column.
    
    # Fill NaN cluster labels if any
    if 'cluster_label' in df_viz.columns:
        df_viz['cluster_label'] = df_viz['cluster_label'].fillna('Unclustered')
    else:
        df_viz['cluster_label'] = 'Unknown'


    # Filter Toggle
    view_mode = "All Notes"
    if search_query:
        # Just use a radio for quick toggling at top
        view_mode = st.radio("View Mode", ["All Notes", "Search Results Only"], horizontal=True, label_visibility="collapsed")
    
    if view_mode == "Search Results Only":
         df_viz = df_viz[df_viz['is_search_result']]

    # Layout Containers
    # We define containers to control visual order (Chart top, Table bottom)
    # But we execute Table logic first to capture selection state for the Chart.
    chart_container = st.container()
    table_container = st.container()

    # --- Table Logic & Selection (Executed first for state) ---
    
    # Initialize Session State Variables
    if "manual_selection" not in st.session_state:
        st.session_state.manual_selection = set()
    if "last_chart_selection" not in st.session_state:
        st.session_state.last_chart_selection = set()

    # 1. Capture Current Chart Selection
    current_chart_selection = set()
    if "cluster_chart" in st.session_state:
        selection_state = st.session_state.cluster_chart
        if selection_state and selection_state.get("selection", {}).get("points", []):
            for point in selection_state["selection"]["points"]:
                # unique_key is validly stored in customdata
                if "customdata" in point and len(point["customdata"]) > 0:
                    current_chart_selection.add(point["customdata"][0])

    # 2. Reconcile Selection State - ADDITIVE ONLY from Chart
    # Logic: Only ADD things that are newly selected in the chart. 
    # Do not remove things (allow Table checkboxes to handle removal).
    
    newly_selected_from_chart = current_chart_selection - st.session_state.last_chart_selection
    
    if newly_selected_from_chart:
        st.session_state.manual_selection.update(newly_selected_from_chart)
        # Force these new items to be visible if we wanted, but for now just update state.
        
    # Update baseline for next comparison
    st.session_state.last_chart_selection = current_chart_selection
    
    # Establish authoritative selection for this run
    selected_keys = st.session_state.manual_selection
    
    if search_query:
        with table_container:
            st.subheader("Search Results")
            # Filter 
            results_df = df_viz[df_viz['is_search_result']].copy().sort_values('search_score', ascending=False)
            
            # Display
            display_cols = ['search_score', 'title', 'cluster_label', 'creation_date', 'cluster_summary']
            
            # Prepare data for editor
            # We add a 'selected' column populated based on current effective selection
            editor_df = results_df.copy()
            editor_df['selected'] = editor_df['unique_key'].isin(selected_keys)
            
            # Order columns: selected first
            cols_order = ['selected'] + display_cols
            
            # Configure columns
            col_config = {
                "selected": st.column_config.CheckboxColumn("Select", width="small"),
                "search_score": st.column_config.ProgressColumn("Relevance", format="%.1f", min_value=0, max_value=100),
                "title": "Note Title",
                "cluster_label": "Cluster",
                "creation_date": "Created",
                "cluster_summary": "Cluster Context"
            }

            # Use data_editor to allow checkbox interaction
            # We disable editing for all columns except 'selected'
            edited_df = st.data_editor(
                editor_df[cols_order],
                column_config=col_config,
                disabled=display_cols, # Disable all content columns
                hide_index=True,
                use_container_width=True,
                key="results_editor" 
            )
            
            # 3. Update Selection from Table Interaction
            if not edited_df.empty:
                # Identify which visible rows are checked
                # We can now use unique_key column directly if exposed, but we rely on index alignment or id lookup
                # Let's trust index alignment for now as results_df and editor_df are synced.
                selected_indices = edited_df[edited_df['selected']].index
                current_visible_selection = set(results_df.loc[selected_indices, 'unique_key'])
                
                # Identify which keys are currently visible in the table
                all_visible_keys = set(results_df['unique_key'])
                
                # Update Manual Selection:
                # 1. Remove all currently visible keys from the master set (clear the slate for the visible portion)
                # 2. Add back the ones that are currently checked
                # This ensures unchecking a box actually removes it from state.
                
                new_state = (st.session_state.manual_selection - all_visible_keys).union(current_visible_selection)
                
                # Update State
                if new_state != st.session_state.manual_selection:
                    st.session_state.manual_selection = new_state
                    selected_keys = new_state # Update local var for immediate use if needed (though UI is already drawn)
                    
                if current_visible_selection:
                     st.caption(f"Selected: **{len(current_visible_selection)}** notes")


    # --- Chart Logic (Executed second, renders to top container) ---
    with chart_container:
        # Prepare tooltip columns
        hover_data = {
            'unique_key': False, # Index 0 in customdata for selection mapping
            'title': True,
            'creation_date': True,
            'modification_date': True,
            'cluster_label': True,
            'search_score': True if search_query else False,
            'umap_x': False,
            'umap_y': False,
            'cluster_id': False,
            'marker_symbol': False,
            'marker_size': False,
            'opacity': False,
            'line_width': False
        }
    
        # Handle colors logic
        if 'cluster_label' in df_viz.columns:
            df_viz['cluster_label'] = df_viz['cluster_label'].fillna('Unclustered')
        else:
            df_viz['cluster_label'] = 'Unknown'
    
        fig = px.scatter(
            df_viz,
            x='umap_x',
            y='umap_y',
            color='cluster_label',
            hover_data=hover_data,
            title="Notes Clusters (UMAP Projection)",
            template="plotly_white",
            height=700 
        )
    
        # --- Distinctipy Color Generation ---
        all_clusters = sorted(df['cluster_label'].fillna('Unclustered').unique())
        n_all_clusters = len(all_clusters)
        
        colors = distinctipy.get_colors(n_all_clusters)
        colors.sort(key=lambda rgb: rgb_to_hsv(*rgb)[0]) # Sort by Hue
        hex_colors = [distinctipy.get_hex(c) for c in colors]
        color_map = dict(zip(all_clusters, hex_colors))
    
        # Apply colors & Custom Sizing
        # We iterate traces to apply styles based on search status AND selection
        
        for trace in fig.data:
            cluster_name = trace.name
            cluster_df = df_viz[df_viz['cluster_label'] == cluster_name]
            
            if not cluster_df.empty:
                base_color = color_map.get(cluster_name, '#888888')
                
                if search_query and view_mode == "All Notes":
                    # Logic:
                    # 1. Non-results: Size 5, Gray, Low Opacity
                    # 2. Results: Size Proportional to Score (15-30), Cluster Color, High Opacity
                    # 3. Selected: Size 45, Cluster Color, High Opacity, Thick Line
                    
                    # Extract arrays
                    is_res = cluster_df['is_search_result'].values
                    scores = cluster_df['search_score'].values
                    keys = cluster_df['unique_key'].values
                    
                    c_array = []
                    s_array = []
                    o_array = []
                    l_width = []
                    l_color = []
                    
                    for i in range(len(keys)):
                        if keys[i] in selected_keys:
                            # SELECTED
                            c_array.append(base_color)
                            s_array.append(45) # Huge
                            o_array.append(1.0)
                            l_width.append(4) # Thick border
                            l_color.append('black')
                        elif is_res[i]:
                            # SEARCH RESULT
                            c_array.append(base_color)
                            # Size: Base 15 + up to 15 more based on score (assuming score 0-100)
                            # Clamped max 30
                            size_val = 15 + (scores[i] / 100.0 * 15)
                            s_array.append(size_val)
                            o_array.append(0.9) # Distinct
                            l_width.append(1)
                            l_color.append('black')
                        else:
                            # BACKGROUND NOISE
                            c_array.append('#e0e0e0')
                            s_array.append(5)
                            o_array.append(0.3)
                            l_width.append(0)
                            # Use fully transparent RGBA instead of 'transparent' string which scattergl might dislike in arrays
                            l_color.append('rgba(0,0,0,0)')

                    trace.marker.color = c_array
                    trace.marker.size = s_array
                    trace.marker.opacity = o_array
                    trace.marker.line.width = l_width
                    trace.marker.line.color = l_color
                
                elif search_query and view_mode == "Search Results Only":
                    # Logic:
                    # All visible are results.
                    # 1. Selected: Size 45
                    # 2. Results: Size Proportional (20-40) roughly? Or same as above.
                    
                    keys = cluster_df['unique_key'].values
                    scores = cluster_df['search_score'].values
                    
                    s_array = []
                    l_width = []
                    l_color = []
                    c_array = [] # Keep base color
    
                    for i in range(len(keys)):
                         if keys[i] in selected_keys:
                             s_array.append(45)
                             l_width.append(4)
                         else:
                             # Slightly larger in this mode since no noise
                             size_val = 20 + (scores[i] / 100.0 * 20)
                             s_array.append(size_val)
                             l_width.append(1)
                         
                         l_color.append('black')
                         c_array.append(base_color)
    
                    trace.marker.color = c_array
                    trace.marker.size = s_array
                    trace.marker.line.width = l_width
                    trace.marker.line.color = l_color
    
                else:
                     # Normal Mode (No Search)
                     trace.marker.color = base_color
                     trace.marker.size = 8
                     trace.marker.line.width = 0
    
        fig.update_layout(
            xaxis_title="UMAP Dimension X",
            yaxis_title="UMAP Dimension Y",
            legend_title_text='Cluster'
        )
        
        st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="cluster_chart")

    # (Previous table location was here)


    # Show raw data view (optional)
    with st.expander("View Raw Data"):
        st.dataframe(df_viz.drop(columns=['vector', 'umap_x', 'umap_y', 'unique_key'], errors='ignore'))

if __name__ == "__main__":
    main()
