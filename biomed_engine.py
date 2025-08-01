import os
import time
import re
import string
import json
import xml.etree.ElementTree as ET
from collections import Counter
import itertools

# --- NLP Libraries ---
import spacy
from spacy.lang.en import English
from Bio import Entrez
from transformers import pipeline

# --- Data/Search Libraries ---
from elasticsearch import Elasticsearch, helpers

# --- ML/NLP Specific Libraries ---
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
import nltk

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords

# --- Visualization Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network


# --- Global Configurations ---
# Set your email for Entrez. It's required by NCBI.
Entrez.email = "email" # <<< IMPORTANT: CHANGE THIS TO YOUR EMAIL >>>
# Optional: Set an API key if you have one for higher request rates
# Entrez.api_key = "YOUR_NCBI_API_KEY" # <<< OPTIONAL: UNCOMMENT AND ADD YOUR NCBI API KEY >>>

# PubMed Query parameters
SEARCH_TERM = "CRISPR gene editing cancer"
MAX_RECORDS = 50  # Limiting for demonstration purposes. For real project, use more (e.g., 500-5000).
BATCH_SIZE = 10   # How many records to fetch per API call
DB = "pubmed"     # Database to search (e.g., pubmed, pmc, protein)

# Data Storage
DATA_DIR = "pubmed_data"

# Elasticsearch Configuration
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX_NAME = "biomed_literature_engine"

# --- Global NLP Models (loaded once) ---
nlp_sm = None
nlp_sci_ner_sm = None
nlp_sci_ner_bc5cdr = None
ner_pipeline_hf = None
all_stopwords = None

def load_nlp_models():
    """Loads all necessary spaCy, scispaCy, and HuggingFace models."""
    global nlp_sm, nlp_sci_ner_sm, nlp_sci_ner_bc5cdr, ner_pipeline_hf, all_stopwords

    print("Loading NLP models...")
    try:
        nlp_sm = spacy.load("en_core_sci_sm")
        print("Loaded en_core_sci_sm for preprocessing.")
    except OSError:
        print("en_core_sci_sm not found. Please run: python -m spacy download en_core_sci_sm")
        exit()

    try:
        nlp_sci_ner_sm = spacy.load("en_core_sci_sm") # General biomedical entities
        nlp_sci_ner_bc5cdr = spacy.load("en_ner_bc5cdr_md") # CHEMICAL and DISEASE entities
        print("Loaded scispaCy NER models (en_core_sci_sm, en_ner_bc5cdr_md).")
    except OSError:
        print("scispaCy NER models not found. Please run:")
        print("python -m spacy download en_core_sci_sm")
        print("python -m spacy download en_ner_bc5cdr_md")
        exit()

    try:
        # This model identifies entities like Gene, Organism, Cell Line, Cell Type, Chemical, Protein, DNA, RNA.
        # It doesn't explicitly have 'Drug' or 'Disease' as separate labels, but 'Chemical' can cover drugs.
        ner_pipeline_hf = pipeline("ner", model="tner/bionlp13cg", aggregation_strategy="simple")
        print("Loaded HuggingFace NER pipeline with tner/bionlp13cg.")
    except Exception as e:
        print(f"Could not load HuggingFace NER model (tner/bionlp13cg): {e}")
        print("Proceeding without HuggingFace NER for this demo.")
        ner_pipeline_hf = None

    # Load NLTK stopwords and combine with Gensim's
    try:
        nltk_stopwords = set(stopwords.words('english'))
        all_stopwords = nltk_stopwords.union(STOPWORDS)
        print("Loaded NLTK and Gensim stopwords.")
    except LookupError:
        print("NLTK stopwords not found. Please run 'python -c \"import nltk; nltk.download(\'punkt\'); nltk.download(\'stopwords\')\"'")
        exit()


# --- 3. Data Collection: Fetching Biomedical Abstracts from PubMed ---

def search_pubmed(query, max_records):
    """Searches PubMed and returns UIDs."""
    print(f"Searching PubMed for '{query}'...")
    try:
        handle = Entrez.esearch(db=DB, term=query, retmax=max_records)
        record = Entrez.read(handle)
        handle.close()
        print(f"Found {record['Count']} records, fetching up to {len(record['IdList'])}.")
        return record["IdList"]
    except Exception as e:
        print(f"Error during PubMed search: {e}")
        return []

def fetch_abstracts(id_list):
    """Fetches full article details (including abstract) for a list of UIDs."""
    print(f"Fetching details for {len(id_list)} UIDs...")
    records = []
    if not id_list:
        return records

    for i in range(0, len(id_list), BATCH_SIZE):
        batch_ids = id_list[i : i + BATCH_SIZE]
        id_str = ",".join(batch_ids)
        try:
            handle = Entrez.efetch(db=DB, id=id_str, rettype="fasta", retmode="xml")
            xml_data = handle.read()
            handle.close()
            root = ET.fromstring(xml_data)

            for pub_article in root.findall(".//PubmedArticle"):
                pmid = pub_article.find(".//PMID").text if pub_article.find(".//PMID") is not None else "N/A"
                title = pub_article.find(".//ArticleTitle").text if pub_article.find(".//ArticleTitle") is not None else "No Title"

                # --- CORRECTED ABSTRACT EXTRACTION ---
                abstract = "No Abstract"
                abstract_elem = pub_article.find(".//AbstractText")
                if abstract_elem is not None and abstract_elem.text is not None:
                    abstract = abstract_elem.text
                # --- END CORRECTED ABSTRACT EXTRACTION ---
                
                pub_date = pub_article.find(".//PubDate")
                year = None
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None:
                        year = year_elem.text
                    else:
                        medline_date_elem = pub_date.find("MedlineDate")
                        if medline_date_elem is not None and len(medline_date_elem.text) >= 4:
                            year = medline_date_elem.text[:4]

                records.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "publication_year": int(year) if year and year.isdigit() else None
                })
            time.sleep(0.3) # Throttling: ~3 requests per second
        except Exception as e:
            print(f"Error fetching batch {batch_ids}: {e}")
            time.sleep(1) # Longer pause on error

    return records

# --- 4. Text Preprocessing ---

def preprocess_text(text):
    """
    Cleans and normalizes biomedical text using scispaCy.
    """
    if not text or not nlp_sm:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp_sm(text)

    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_stop and not token.like_num and token.text.strip():
            if len(token.text.strip()) > 1 or (len(token.text.strip()) == 1 and token.text.isalpha()):
                tokens.append(token.lemma_)
    
    return " ".join(tokens)

# --- 5. Named Entity Recognition (NER) ---

def extract_entities_scispacy(text):
    """
    Extracts entities using scispaCy's pre-trained models.
    Combines general biomedical entities and specific CHEMICAL/DISEASE entities.
    """
    if not nlp_sci_ner_sm or not nlp_sci_ner_bc5cdr:
        return []

    doc_sm = nlp_sci_ner_sm(text)
    doc_bc5cdr = nlp_sci_ner_bc5cdr(text)

    entities = []
    for ent in doc_sm.ents:
        entities.append({"text": ent.text, "label": ent.label_, "source": "scispacy_sm"})
    
    for ent in doc_bc5cdr.ents:
        if ent.label_ in ["CHEMICAL", "DISEASE"]:
            entities.append({"text": ent.text, "label": ent.label_, "source": "scispacy_bc5cdr"})
            
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent['text'].lower(), ent['label'])
        if key not in seen:
            unique_entities.append(ent)
            seen.add(key)
    return unique_entities


def extract_entities_huggingface(text):
    """
    Extracts entities using a HuggingFace Transformer model.
    """
    if ner_pipeline_hf is None:
        return []
    
    entities = ner_pipeline_hf(text)
    
    extracted = []
    for ent in entities:
        label = ent['entity_group']
        if label == 'CHEMICAL':
            label = 'DRUG' # Map 'CHEMICAL' to 'DRUG' for project's domain
        extracted.append({"text": ent['word'], "label": label, "score": ent['score'], "source": "huggingface"})
    return extracted

# --- 6. Relation Extraction ---

def extract_co_occurrence_relations(article_entities):
    """
    Extracts co-occurrence relations between entities within an article.
    Assumes entities are lists of dicts like {'text': '...', 'label': '...'}.
    """
    relations = []
    unique_entities = {}
    for ent in article_entities:
        unique_entities[ent['text'].lower()] = ent['label']

    entity_names = list(unique_entities.keys())
    for ent1_text, ent2_text in itertools.combinations(entity_names, 2):
        original_ent1 = [e['text'] for e in article_entities if e['text'].lower() == ent1_text][0] if [e['text'] for e in article_entities if e['text'].lower() == ent1_text] else ent1_text
        original_ent2 = [e['text'] for e in article_entities if e['text'].lower() == ent2_text][0] if [e['text'] for e in article_entities if e['text'].lower() == ent2_text] else ent2_text
        
        relations.append({
            "entity1": original_ent1,
            "entity1_type": unique_entities[ent1_text],
            "entity2": original_ent2,
            "entity2_type": unique_entities[ent2_text],
            "relationship_type": "CO_OCCURS_WITH"
        })
    return relations

def build_knowledge_graph(articles, graph_type="co_occurrence", min_links=1):
    """
    Builds a NetworkX graph from extracted relations.
    """
    G = nx.Graph()

    for article in articles:
        pmid = article['pmid']
        entities = [ent for ent in article.get('all_entities', []) if ent['text'].strip()]

        if graph_type == "co_occurrence":
            added_relations = extract_co_occurrence_relations(entities)
            
            for rel in added_relations:
                node1_id = rel['entity1']
                node2_id = rel['entity2']
                
                G.add_node(node1_id, label=node1_id, type=rel['entity1_type'])
                G.add_node(node2_id, label=node2_id, type=rel['entity2_type'])

                if G.has_edge(node1_id, node2_id):
                    G[node1_id][node2_id]['weight'] += 1
                    G[node1_id][node2_id]['articles'].add(pmid)
                else:
                    G.add_edge(node1_id, node2_id, weight=1, relationship=rel['relationship_type'], articles={pmid})
    
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data['weight'] < min_links]
    G.remove_edges_from(edges_to_remove)
    
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G

# --- 7. Search Engine Integration (Elasticsearch) ---

def connect_to_elasticsearch():
    """Establishes connection to Elasticsearch."""
    es_client = None
    try:
        es_client = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")
        if es_client.ping():
            print(f"Connected to Elasticsearch at {ES_HOST}:{ES_PORT}")
            return es_client
        else:
            print(f"Could not connect to Elasticsearch at {ES_HOST}:{ES_PORT}")
            return None
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return None

def create_index(es_client, index_name):
    """Creates the Elasticsearch index with appropriate mappings."""
    if es_client is None: return

    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "pmid": {"type": "keyword"},
                "title": {"type": "text"},
                "abstract": {"type": "text"},
                "processed_text": {"type": "text"},
                "publication_year": {"type": "integer"},
                "entities": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "keyword"},
                        "label": {"type": "keyword"}
                    }
                }
            }
        }
    }
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=settings)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")
        # Optionally, delete and recreate for fresh run:
        # es_client.indices.delete(index=index_name, ignore=[400, 404])
        # es_client.indices.create(index=index_name, body=settings)
        # print(f"Index '{index_name}' recreated.")


def index_documents(es_client, index_name, documents):
    """Indexes processed articles into Elasticsearch."""
    if es_client is None: return

    actions = []
    for doc in documents:
        entity_list = [{"text": e['text'], "label": e['label']} for e in doc.get('all_entities', [])]
        
        action = {
            "_index": index_name,
            "_id": doc["pmid"],
            "_source": {
                "pmid": doc["pmid"],
                "title": doc["title"],
                "abstract": doc["abstract"],
                "processed_text": doc["processed_text"],
                "publication_year": doc["publication_year"],
                "entities": entity_list
            }
        }
        actions.append(action)
    
    if actions:
        success, failed = helpers.bulk(es_client, actions)
        print(f"Successfully indexed {success} documents, {len(failed)} failed.")
        if failed:
            print(f"Failed items: {failed}")

def search_articles(es_client, index_name, query_text, year_filter=None, entity_filter=None, size=5):
    """
    Performs a search query on Elasticsearch.
    Allows full-text search, year filtering, and entity filtering.
    """
    if es_client is None: return []

    search_body = {
        "query": {
            "bool": {
                "must": [],
                "filter": []
            }
        },
        "size": size
    }

    if query_text:
        search_body["query"]["bool"]["must"].append({
            "multi_match": {
                "query": query_text,
                "fields": ["title^3", "abstract", "processed_text"]
            }
        })

    if year_filter:
        search_body["query"]["bool"]["filter"].append({
            "range": {
                "publication_year": {
                    "gte": year_filter[0],
                    "lte": year_filter[1]
                }
            }
        })

    if entity_filter:
        search_body["query"]["bool"]["filter"].append({
            "nested": {
                "path": "entities",
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"entities.text": entity_filter["text"]}},
                            {"match": {"entities.label": entity_filter["label"]}}
                        ]
                    }
                }
            }
        })

    try:
        res = es_client.search(index=index_name, body=search_body)
        print(f"Found {res['hits']['total']['value']} hits for query '{query_text}'.")
        return res['hits']['hits']
    except Exception as e:
        print(f"Error during search: {e}")
        return []

# --- 8. Topic Modeling & Trend Analysis ---

def tokenize_for_lda(text):
    """Simple tokenizer for LDA, removing short words and stopwords."""
    if not all_stopwords: # Fallback if stopwords not loaded
        return [token for token in text.lower().split() if len(token) > 2]
    return [token for token in text.lower().split() if token not in all_stopwords and len(token) > 2]

def run_lda(articles_data, num_topics=5, passes=10):
    """
    Performs LDA topic modeling on processed texts.
    """
    print(f"\nRunning LDA with {num_topics} topics...")
    texts = [tokenize_for_lda(article['processed_text']) for article in articles_data if article.get('processed_text')]
    
    if not texts or len(texts) < num_topics: # Need at least as many documents as topics
        print(f"Not enough processed texts ({len(texts)}) for LDA with {num_topics} topics. Skipping LDA.")
        return None, None, None

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000) 
    corpus = [dictionary.doc2bow(text) for text in texts]

    if not corpus:
        print("Empty corpus after filtering for LDA. Skipping LDA.")
        return None, None, None

    try:
        lda_model = models.LdaMulticore(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=num_topics,
                                        random_state=100,
                                        chunksize=100,
                                        passes=passes,
                                        per_word_topics=True)
        print("LDA model training complete.")
        return lda_model, corpus, dictionary
    except Exception as e:
        print(f"Error during LDA model training: {e}")
        return None, None, None

def get_document_topics(lda_model, corpus):
    """Assigns the dominant topic to each document."""
    doc_topics = []
    if lda_model and corpus:
        for i, doc_bow in enumerate(corpus):
            topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
            dominant_topic = max(topics, key=lambda x: x[1])
            doc_topics.append(dominant_topic[0])
    return doc_topics

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Building a Biomedical Literature Search & Insight Engine ---")
    print("\n--- Initial Setup & Model Loading ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    load_nlp_models()
    
    raw_articles = []
    kg_graph = None
    es = None
    
    # --- 3. Data Collection ---
    print("\n--- 3. Data Collection ---")
    pubmed_ids = search_pubmed(SEARCH_TERM, MAX_RECORDS)
    raw_articles = fetch_abstracts(pubmed_ids)

    print(f"\nFetched {len(raw_articles)} articles.")
    if raw_articles:
        print("Sample Article:")
        print(raw_articles[0])
    else:
        print("No articles fetched. Subsequent steps will be skipped or have limited data.")

    # --- 4. Text Preprocessing ---
    print("\n--- 4. Text Preprocessing ---")
    if not raw_articles:
        print("No articles to preprocess. Skipping.")
    else:
        sample_article = raw_articles[0]
        original_abstract = sample_article["abstract"]
        print(f"Original Abstract (PMID: {sample_article['pmid']}):\n{original_abstract[:300]}...\n")
        
        for article in raw_articles:
            article["processed_text"] = preprocess_text(article["abstract"])
        print(f"Example processed text for PMID {raw_articles[0]['pmid']}: {raw_articles[0]['processed_text'][:200]}...")

    # --- 5. Named Entity Recognition (NER) ---
    print("\n--- 5. Named Entity Recognition (NER) ---")
    if not raw_articles:
        print("No articles to process for NER. Skipping.")
    else:
        sample_article_text = raw_articles[0]["processed_text"]
        original_abstract_for_ner = raw_articles[0]["abstract"] 

        print(f"Original Abstract (for NER context):\n{original_abstract_for_ner[:300]}...\n")

        for i, article in enumerate(raw_articles): # Added 'i' for index to help identify problematic article
            # --- DEBUG PRINT STATEMENT ADDED HERE ---
            print(f"DEBUGGING NER INPUT: Article index {i}, PMID: {article['pmid']}, Abstract Type: {type(article['abstract'])}, Abstract Start: '{str(article['abstract'])[:50]}'...")
            # --- END DEBUG PRINT STATEMENT ---

            scispacy_entities = extract_entities_scispacy(article["abstract"])
            hf_entities = extract_entities_huggingface(article["abstract"])
            
            # Combine and deduplicate entities from both sources
            all_entities_for_article = []
            seen_entities = set()
            for ent in scispacy_entities + hf_entities:
                key = (ent['text'].lower(), ent['label'])
                if key not in seen_entities:
                    all_entities_for_article.append(ent)
                    seen_entities.add(key)
            article["all_entities"] = all_entities_for_article
        
        print(f"Example entities for PMID {raw_articles[0]['pmid']}:")
        for ent in raw_articles[0]["all_entities"][:5]:
            print(f"  - Text: {ent['text']}, Label: {ent['label']}, Source: {ent.get('source', 'N/A')}")
        if len(raw_articles[0]["all_entities"]) > 5:
            print(f"  ... and {len(raw_articles[0]['all_entities']) - 5} more.")

    # --- 6. Relation Extraction ---
    print("\n--- 6. Relation Extraction ---")
    if not raw_articles or not any("all_entities" in a and a["all_entities"] for a in raw_articles):
        print("No articles with entities to process for relation extraction. Skipping.")
    else:
        all_relations = []
        for article in raw_articles:
            if "all_entities" in article:
                article_relations = extract_co_occurrence_relations(article["all_entities"])
                all_relations.extend(article_relations)
                
        print(f"Extracted {len(all_relations)} co-occurrence relations across all articles.")
        if all_relations:
            print("Sample relation:", all_relations[0])

        print("\nBuilding Knowledge Graph...")
        kg_graph = build_knowledge_graph(raw_articles, min_links=1)

        print(f"Knowledge Graph created with {kg_graph.number_of_nodes()} nodes and {kg_graph.number_of_edges()} edges.")
        if kg_graph.number_of_edges() > 0:
            print("\nSample Graph Nodes and Edges:")
            for i, node in enumerate(list(kg_graph.nodes)[:5]):
                print(f"  Node: {node} (Type: {kg_graph.nodes[node].get('type', 'N/A')})")
            for i, edge in enumerate(list(kg_graph.edges(data=True))[:5]):
                print(f"  Edge: {edge[0]} --({edge[2]['relationship']})--> {edge[1]} (Weight: {edge[2]['weight']})")
        else:
            print("No significant relations found to build a graph.")

    # --- 7. Search Engine Integration (Elasticsearch) ---
    print("\n--- 7. Search Engine Integration (Elasticsearch) ---")
    es = connect_to_elasticsearch()
    if es is None:
        print("Elasticsearch is not running or not accessible. Skipping this section and dependent sections.")
    elif not raw_articles:
        print("No articles to index. Skipping Elasticsearch indexing and search.")
    else:
        create_index(es, ES_INDEX_NAME)
        index_documents(es, ES_INDEX_NAME, raw_articles)

        print("\n--- Testing Search ---")
        results = search_articles(es, ES_INDEX_NAME, query_text="CRISPR cancer therapy", size=3)
        print("\nSearch Results for 'CRISPR cancer therapy':")
        for hit in results:
            print(f"  PMID: {hit['_source']['pmid']}, Title: {hit['_source']['title'][:70]}...")

        results = search_articles(es, ES_INDEX_NAME, query_text="gene editing", year_filter=(2019, 2023), size=3)
        print("\nSearch Results for 'gene editing' (2019-2023):")
        for hit in results:
            print(f"  PMID: {hit['_source']['pmid']}, Title: {hit['_source']['title'][:70]}..., Year: {hit['_source']['publication_year']}")
            
        all_labels_in_data = set()
        for article in raw_articles:
            for ent in article.get('all_entities', []):
                all_labels_in_data.add(ent['label'])
        print(f"\nAvailable entity labels in indexed data: {all_labels_in_data}")
        
        if 'GENE' in all_labels_in_data:
            results = search_articles(es, ES_INDEX_NAME, query_text="therapy", entity_filter={"text": "CRISPR", "label": "GENE"}, size=3)
            print("\nSearch Results for 'therapy' mentioning 'CRISPR' (as GENE):")
            for hit in results:
                print(f"  PMID: {hit['_source']['pmid']}, Title: {hit['_source']['title'][:70]}...")
        else:
             print("\nSkipping entity-filtered search: 'GENE' label not found in data for demonstration.")

    # --- 8. Topic Modeling & Trend Analysis ---
    print("\n--- 8. Topic Modeling & Trend Analysis ---")
    if not raw_articles or len(raw_articles) < 10:
        print("Not enough articles for meaningful topic modeling. Skipping.")
        lda_model = None
        corpus = None
        dictionary = None
    else:
        articles_for_lda = [a for a in raw_articles if a.get('processed_text') and a.get('publication_year')]
        lda_model, corpus, dictionary = run_lda(articles_for_lda, num_topics=min(5, len(articles_for_lda) // 2) if len(articles_for_lda) > 1 else 1) # Adjust num_topics dynamically

        if lda_model:
            print("\n--- Discovered Topics ---")
            for idx, topic in lda_model.print_topics(-1):
                print(f"Topic: {idx} \nWords: {topic}\n")
            
            document_topics = get_document_topics(lda_model, corpus)
            for i, article in enumerate(articles_for_lda):
                article['dominant_topic'] = document_topics[i]

            print("\n--- Topic Trend Analysis ---")
            topic_trends = pd.DataFrame([
                {'year': article['publication_year'], 'topic': article['dominant_topic']}
                for article in articles_for_lda if article.get('publication_year') is not None
            ])

            if not topic_trends.empty:
                topic_prevalence = topic_trends.groupby(['year', 'topic']).size().unstack(fill_value=0)

                topic_prevalence = topic_prevalence.div(topic_prevalence.sum(axis=1), axis=0).fillna(0) 

                print("\nTopic prevalence over time (sample):\n", topic_prevalence.head())

                plt.figure(figsize=(12, 6))
                sns.lineplot(data=topic_prevalence)
                plt.title('Topic Evolution Over Time')
                plt.xlabel('Publication Year')
                plt.ylabel('Proportion of Articles')
                plt.legend(title='Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("No enough yearly data for topic trend analysis.")
        else:
            print("LDA model could not be trained. Skipping topic trend visualization.")


    # --- 9. Data Visualization ---
    print("\n--- 9. Data Visualization ---")
    
    # --- 9.1 Entity Frequency ---
    print("\n--- Visualizing Entity Frequencies ---")
    all_extracted_entities_plot = []
    for article in raw_articles:
        all_extracted_entities_plot.extend(article.get('all_entities', []))
    
    entity_texts_by_type = {}
    for ent in all_extracted_entities_plot:
        label = ent['label']
        if label == 'CHEMICAL': label = 'DRUG'
        
        if label not in entity_texts_by_type:
            entity_texts_by_type[label] = []
        entity_texts_by_type[label].append(ent['text'].lower())

    if entity_texts_by_type:
        num_plots = len(entity_texts_by_type)
        cols = min(num_plots, 3) # Max 3 columns
        rows = (num_plots + cols - 1) // cols # Calculate rows needed

        plt.figure(figsize=(cols * 5, rows * 4)) # Adjust figure size based on number of plots
        plot_idx = 1
        for label, texts in entity_texts_by_type.items():
            if not texts: continue
            
            top_entities = Counter(texts).most_common(10)
            if not top_entities: continue
            
            entities_df = pd.DataFrame(top_entities, columns=['Entity', 'Count'])
            
            plt.subplot(rows, cols, plot_idx)
            sns.barplot(x='Count', y='Entity', data=entities_df, palette='viridis')
            plt.title(f'Top 10 {label} Frequencies')
            plt.xlabel('Count')
            plt.ylabel(label)
            plot_idx += 1
        plt.tight_layout()
        plt.show()
    else:
        print("No entities found for frequency visualization.")


    # --- 9.2 Knowledge Graph Visualization (PyVis) ---
    print("\n--- Visualizing Knowledge Graph (Interactive with PyVis) ---")
    if kg_graph and kg_graph.number_of_edges() > 0:
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')
        net.toggle_physics(True)

        for node, data in kg_graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            color = 'lightblue'
            if node_type == 'GENE':
                color = '#FF7F0E' # Orange
            elif node_type == 'DISEASE':
                color = '#2CA02C' # Green
            elif node_type == 'DRUG' or node_type == 'CHEMICAL':
                color = '#D62728' # Red
            
            title = f"Type: {node_type}<br>Name: {node}"
            net.add_node(node, label=node, title=title, color=color, physics=True)

        for u, v, data in kg_graph.edges(data=True):
            weight = data.get('weight', 1)
            relationship = data.get('relationship', 'CO_OCCURS')
            articles_list = ", ".join(list(data.get('articles', {}))) if data.get('articles') else "N/A"
            title = f"Relationship: {relationship}<br>Weight: {weight}<br>Articles: {articles_list}"
            net.add_edge(u, v, title=title, value=weight, label=relationship, physics=True)
            
        output_path = os.path.join(DATA_DIR, "biomed_knowledge_graph.html")
        net.show(output_path)
        print(f"Interactive knowledge graph saved to {output_path}")
        print("Open this HTML file in your browser to view the graph.")
    else:
        print("No knowledge graph to visualize or graph is empty.")

    print("\n--- Project Execution Complete ---")
