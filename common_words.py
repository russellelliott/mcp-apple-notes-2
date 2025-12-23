#!/usr/bin/env python3
"""
Find the most common words across all chunk content in the vector database. 
"""

import sys
from pathlib import Path
from collections import Counter
import re
from typing import List, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys. path.insert(0, str(REPO_ROOT))

import lancedb

# Configuration
DATA_DIR = Path. home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"


def clean_text(text: str) -> List[str]:
    """
    Clean and tokenize text into words.
    
    Args:
        text: Raw text to clean
        
    Returns: 
        List of cleaned words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Extract words (letters only, 2+ characters)
    # This matches sequences of lowercase letters
    words = re.findall(r'\b[a-z]{2,}\b', text)
    
    return words


def get_most_common_words(
    limit: int = 50,
    min_word_length:  int = 2
) -> Tuple[List[Tuple[str, int]], int]:
    """
    Find the most common words across all chunk content in the database.
    
    Args:
        limit: Number of top words to return
        min_word_length:  Minimum word length to consider
        
    Returns:
        Tuple of (list of (word, count) tuples sorted by frequency, total word count)
    """
    print(f"📂 Connecting to LanceDB at: {DB_PATH}")
    db = lancedb.connect(str(DB_PATH))
    
    try:
        notes_table = db.open_table(TABLE_NAME)
    except Exception as e:
        print(f"❌ Error opening table '{TABLE_NAME}': {e}")
        return [], 0
    
    print(f"📥 Loading all chunks from database...")
    all_chunks = notes_table.to_pandas()
    
    total_chunks = len(all_chunks)
    print(f"   Found {total_chunks} chunks")
    
    if total_chunks == 0:
        print("⚠️  No chunks found in database")
        return [], 0
    
    print(f"🔍 Processing chunk content...")
    word_counter = Counter()
    
    for idx, row in all_chunks.iterrows():
        chunk_content = row. get('chunk_content', '')
        if not chunk_content or not isinstance(chunk_content, str):
            continue
        
        # Clean and tokenize
        words = clean_text(chunk_content)
        
        # Filter by minimum length
        filtered_words = [
            word for word in words
            if len(word) >= min_word_length
        ]
        
        # Update counter
        word_counter.update(filtered_words)
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{total_chunks} chunks.. .", end='\r')
    
    print(f"   Processed {total_chunks}/{total_chunks} chunks    ")
    
    # Calculate total words
    total_words = sum(word_counter.values())
    
    # Get most common words
    most_common = word_counter.most_common(limit)
    
    print(f"\n✅ Found {len(word_counter)} unique words")
    print(f"📊 Total words: {total_words: ,}")
    print(f"📊 Top {limit} most common words:\n")
    
    return most_common, total_words


def display_results(results: List[Tuple[str, int]], total_words: int, show_bar_chart: bool = True):
    """
    Display the results in a formatted way with percentages.
    
    Args:
        results: List of (word, count) tuples
        total_words: Total number of words in the corpus
        show_bar_chart:  Whether to show a simple text bar chart
    """
    if not results:
        print("No results to display")
        return
    
    # Find max count for scaling bar chart
    max_count = results[0][1] if results else 0
    max_word_length = max(len(word) for word, _ in results) if results else 0
    
    # Calculate cumulative percentage
    cumulative_count = 0
    
    for rank, (word, count) in enumerate(results, 1):
        # Calculate percentage
        percentage = (count / total_words) * 100 if total_words > 0 else 0
        cumulative_count += count
        cumulative_percentage = (cumulative_count / total_words) * 100 if total_words > 0 else 0
        
        # Format rank and word
        rank_str = f"{rank:3d}."
        word_str = f"{word: <{max_word_length}}"
        count_str = f"{count:7,d}"
        pct_str = f"{percentage: 5.2f}%"
        cumulative_str = f"{cumulative_percentage:6.2f}%"
        
        if show_bar_chart:
            # Create a simple bar chart
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            print(f"{rank_str} {word_str}  {count_str}  {pct_str}  {bar}")
        else:
            print(f"{rank_str} {word_str}  {count_str}  {pct_str}  (cumulative: {cumulative_str})")


def main():
    """Main function to run the word frequency analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find the most common words across all chunk content in the vector database"
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        default=50,
        help='Number of top words to display (default: 50)'
    )
    parser.add_argument(
        '-m', '--min-length',
        type=int,
        default=2,
        help='Minimum word length to consider (default: 2)'
    )
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='Disable bar chart display'
    )
    
    args = parser.parse_args()
    
    # Get most common words
    results, total_words = get_most_common_words(
        limit=args.limit,
        min_word_length=args. min_length
    )
    
    # Display results
    display_results(results, total_words, show_bar_chart=not args.no_chart)


if __name__ == "__main__": 
    main()