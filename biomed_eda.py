import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# --- Configuration ---
DATA_DIR = "pubmed_data"
PROCESSED_DATA_FILENAME = "processed_pubmed_articles.json"

# --- EDA Functions ---

def load_processed_articles(filepath):
    """Loads processed article data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}. Please run biomed_engine.py first.")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} articles from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def plot_publication_year_distribution(df):
    """Plots the distribution of publication years."""
    if 'publication_year' not in df.columns or df['publication_year'].isnull().all():
        print("Skipping publication year distribution: 'publication_year' data missing or all NaN.")
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(x='publication_year', data=df.dropna(subset=['publication_year']), palette='viridis')
    plt.title('Distribution of Publication Years')
    plt.xlabel('Publication Year')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_text_length_distribution(df):
    """Plots the distribution of text lengths (title, abstract, processed_text)."""
    df['title_len'] = df['title'].apply(lambda x: len(x) if x else 0)
    df['abstract_len'] = df['abstract'].apply(lambda x: len(x) if x else 0)
    df['processed_text_len'] = df['processed_text'].apply(lambda x: len(x) if x else 0)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df['title_len'], bins=20, kde=True, color='skyblue')
    plt.title('Title Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    sns.histplot(df['abstract_len'], bins=20, kde=True, color='lightcoral')
    plt.title('Abstract Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.histplot(df['processed_text_len'], bins=20, kde=True, color='lightgreen')
    plt.title('Processed Text Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_entity_frequencies(articles, top_n=15):
    """Plots the frequency of top entities by type."""
    all_extracted_entities = []
    for article in articles:
        all_extracted_entities.extend(article.get('all_entities', []))
    
    entity_counts_by_type = {}
    for ent in all_extracted_entities:
        label = ent['label']
        if label == 'CHEMICAL': label = 'DRUG' # Standardize
        
        if label not in entity_counts_by_type:
            entity_counts_by_type[label] = Counter()
        entity_counts_by_type[label][ent['text'].lower()] += 1

    if not entity_counts_by_type:
        print("No entities found for frequency visualization.")
        return

    num_plots = len(entity_counts_by_type)
    cols = min(num_plots, 3) 
    rows = (num_plots + cols - 1) // cols 

    plt.figure(figsize=(cols * 6, rows * 5)) 
    plot_idx = 1
    for label, counts in entity_counts_by_type.items():
        if not counts: continue
        
        top_entities = counts.most_common(top_n)
        if not top_entities: continue
        
        entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
        
        plt.subplot(rows, cols, plot_idx)
        sns.barplot(x='Count', y='Entity', data=entities_df, palette='viridis')
        plt.title(f'Top {top_n} {label} Frequencies')
        plt.xlabel('Count')
        plt.ylabel(label)
        plot_idx += 1
    plt.tight_layout()
    plt.show()

def generate_word_clouds(articles, field='processed_text'):
    """Generates a word cloud from the specified text field."""
    combined_text = " ".join([a.get(field, "") for a in articles if a.get(field)])
    
    if not combined_text.strip():
        print(f"No text available in '{field}' field for word cloud.")
        return

    plt.figure(figsize=(10, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(combined_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {field.replace("_", " ").title()}')
    plt.show()

def analyze_topic_distribution(df):
    """Analyzes and plots the overall distribution of dominant topics."""
    if 'dominant_topic' not in df.columns or df['dominant_topic'].isnull().all():
        print("Skipping topic distribution: 'dominant_topic' data missing or all NaN.")
        return
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='dominant_topic', data=df, palette='cubehelix')
    plt.title('Overall Dominant Topic Distribution')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Articles')
    plt.show()


# --- Main EDA Execution ---
if __name__ == "__main__":
    print("--- Starting Biomedical Literature EDA ---")

    # 1. Load Data
    processed_articles = load_processed_articles(os.path.join(DATA_DIR, PROCESSED_DATA_FILENAME))
    if processed_articles is None or not processed_articles:
        print("Exiting EDA as no processed data could be loaded.")
        exit()

    df = pd.DataFrame(processed_articles)
    
    # 2. Data Overview
    print("\n--- Data Overview ---")
    print(f"Total articles loaded: {len(df)}")
    print(f"Columns available: {df.columns.tolist()}")
    
    if 'publication_year' in df.columns:
        print(f"Publication years range: {df['publication_year'].min()} - {df['publication_year'].max()}")
        print(f"Articles with 'No Abstract': {df[df['abstract'] == 'No Abstract'].shape[0]}")
    
    # 3. Plot Publication Year Distribution
    print("\n--- Publication Year Distribution ---")
    plot_publication_year_distribution(df)

    # 4. Plot Text Length Distributions
    print("\n--- Text Length Distributions ---")
    plot_text_length_distribution(df)

    # 5. Plot Entity Frequencies
    print("\n--- Entity Frequencies ---")
    plot_entity_frequencies(processed_articles) # Pass original list to handle nested entities easily

    # 6. Generate Word Clouds
    print("\n--- Generating Word Clouds ---")
    generate_word_clouds(processed_articles, 'abstract')
    generate_word_clouds(processed_articles, 'processed_text') # Word cloud from cleaned text

    # 7. Analyze Topic Distribution
    print("\n--- Overall Topic Distribution ---")
    analyze_topic_distribution(df)

    print("\n--- EDA Complete ---")
    print("Check your plot windows for visualizations.")