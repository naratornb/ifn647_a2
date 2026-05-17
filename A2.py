"""
Assignment 2 - Task 1: Baseline 1 (BM25-based IR Model)

This script implements a BM25 baseline model for ranking documents in each dataset
according to their corresponding topics.

Input:
  - Topics.txt: Topic definitions with IDs and titles
  - Doc_Collection/: Datasets (Dataset101, Dataset102, etc.)

Output:
  - ModelOutputs/Baseline1_R###_Ranking.dat: Ranked document lists per topic

BM25 Formula:
  BM25(D, Q) = Σ_t∈Q log10(1 + (k1+1)*f(t,D) / (K + f(t,D))) * log10(N/n(t))
  where K = k1*((1-b) + b*dl/avdl)
"""

import os
import glob
import string
import re
from collections import defaultdict
from math import log10
from stemming.porter2 import stem


class BowDoc:
    """Bag-of-words document representation."""
    def __init__(self, docid):
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        """Add term occurrence."""
        self.terms[term] = self.terms.get(term, 0) + 1

    def get_docid(self):
        return self.docid

    def get_term_list(self):
        return sorted(self.terms.keys())

    def get_term_freq_dict(self):
        return self.terms

    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, length):
        self.doc_len = length


class BowColl:
    """Collection of BOW documents."""
    def __init__(self):
        self.docs = {}

    def add_doc(self, doc):
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        return self.docs[docid]

    def get_docs(self):
        return self.docs

    def get_num_docs(self):
        return len(self.docs)


def load_stopwords(stopwords_file):
    """Load stop words from file."""
    if not os.path.exists(stopwords_file):
        return set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        data = f.read()
    if ',' in data:
        words = [w.strip().lower() for w in data.split(',') if w.strip()]
    else:
        words = [w.strip().lower() for w in data.split() if w.strip()]
    return set(words)


def parse_dataset(dataset_folder, stop_words):
    """
    Parse all XML documents in a dataset folder.

    Returns:
        BowColl: Collection of parsed documents
    """
    coll = BowColl()

    if not os.path.exists(dataset_folder):
        print(f"Warning: {dataset_folder} not found")
        return coll

    for xml_file in glob.glob(os.path.join(dataset_folder, "*.xml")):
        try:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {xml_file}: {e}")
            continue

        # Extract docid from itemid attribute
        docid_match = re.search(r'itemid="([^"]+)"', content)
        if not docid_match:
            continue
        docid = docid_match.group(1)

        doc = BowDoc(docid)
        doc_len = 0

        # Extract text between <text> tags
        text_matches = re.findall(r'<text>(.*?)</text>', content, re.DOTALL | re.IGNORECASE)

        for text_block in text_matches:
            # Remove XML tags
            text_block = re.sub(r'<[^>]+>', ' ', text_block)
            # Remove digits
            text_block = re.sub(r'\d+', ' ', text_block)
            # Replace punctuation with spaces
            text_block = text_block.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

            # Tokenize and process
            for word in text_block.split():
                doc_len += 1
                word = word.lower()
                if word in stop_words:
                    continue
                term = stem(word)
                if len(term) > 2:
                    doc.add_term(term)

        doc.set_doc_len(doc_len)
        coll.add_doc(doc)

    return coll


def parse_topics(topics_file):
    """
    Parse Topics.txt and extract topic ID and title.

    Returns:
        dict: {topic_id: topic_title}
    """
    topics = {}

    with open(topics_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by </Topic> to get individual topic blocks
    topic_blocks = content.split('</Topic>')

    for block in topic_blocks:
        # Extract topic number
        num_match = re.search(r'<num>\s*Number:\s*(\w+)', block)
        if not num_match:
            continue
        topic_id = num_match.group(1)

        # Extract title
        title_match = re.search(r'<title>\s*([^<\n]+)', block)
        if not title_match:
            continue
        title = title_match.group(1).strip()

        topics[topic_id] = title

    return topics


def queryParser(query, stop_words):
    """
    Parse query string into {term: frequency} dict.

    Applies same preprocessing as documents:
    - Remove digits and punctuation
    - Lowercase
    - Stem with Porter2
    - Filter stop words and short terms
    """
    query_terms = {}

    # Remove digits
    query = re.sub(r'\d+', ' ', query)
    # Replace punctuation with spaces
    query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    for word in query.split():
        word = word.lower()
        if word in stop_words:
            continue
        term = stem(word)
        if len(term) > 2:
            query_terms[term] = query_terms.get(term, 0) + 1

    return query_terms


def compute_df(coll):
    """
    Compute document frequency for each term.

    Returns:
        dict: {term: number_of_docs_containing_term}
    """
    df = defaultdict(int)

    for docid, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            df[term] += 1

    return df


def avg_doc_len(coll):
    """Compute average document length in collection."""
    if coll.get_num_docs() == 0:
        return 0.0
    total = sum(doc.get_doc_len() for doc in coll.get_docs().values())
    return total / coll.get_num_docs()


def bm25_rank(coll, query, df_dict):
    """
    Compute BM25 scores for all documents.

    Parameters:
        coll: BowColl object
        query: query string
        df_dict: {term: document_frequency}

    Returns:
        dict: {docid: bm25_score}

    BM25 parameters:
        k1 = 1.2 (term frequency saturation)
        k2 = 500 (query term frequency saturation)
        b = 0.75 (document length normalization)
    """
    q_terms = queryParser(query, stop_words)

    if not q_terms:
        return {}

    k1 = 1.2
    k2 = 500
    b = 0.75

    N = coll.get_num_docs()
    avdl = avg_doc_len(coll)

    scores = {}

    for docid, doc in coll.get_docs().items():
        K = k1 * ((1 - b) + b * (doc.get_doc_len() / avdl)) if avdl > 0 else k1
        score = 0.0

        for term, qfi in q_terms.items():
            ni = df_dict.get(term, 0)
            fi = doc.terms.get(term, 0)

            if ni == 0 or fi == 0:
                continue

            # BM25 formula: log10(1 + (k1+1)*f(t,D) / (K + f(t,D))) * log10(N/n(t))
            idf = log10(N / ni) if ni > 0 else 0
            tf_component = ((k1 + 1) * fi) / (K + fi)
            score += idf * tf_component

        scores[docid] = score

    return scores


def main():
    """
    Main execution: Baseline 1 (BM25) ranking for all topics.

    For each topic:
      1. Load the corresponding dataset
      2. Compute document frequencies
      3. Parse topic title as query
      4. Rank documents using BM25
      5. Save ranked output
    """
    # Setup
    base_dir = "/Users/niceenb/Documents/647/Assignment2"
    topics_file = os.path.join(base_dir, "Topics.txt")
    doc_collection_base = os.path.join(base_dir, "Doc_Collection")
    output_folder = os.path.join(base_dir, "ModelOutputs")
    stopwords_file = os.path.join(base_dir, "common-english-words.txt")

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Load stop words
    global stop_words
    stop_words = load_stopwords(stopwords_file)
    print(f"Loaded {len(stop_words)} stop words")

    # Parse topics
    topics = parse_topics(topics_file)
    print(f"Loaded {len(topics)} topics from {topics_file}")

    # Process each topic
    for topic_id, topic_title in sorted(topics.items()):
        print(f"\n{'='*60}")
        print(f"Processing Topic {topic_id}: {topic_title}")
        print(f"{'='*60}")

        # Determine dataset folder (R101 → Dataset101, R102 → Dataset102, etc.)
        dataset_num = topic_id[1:]  # Remove 'R' prefix
        dataset_folder = os.path.join(doc_collection_base, f"Dataset{dataset_num}")

        # Parse dataset
        print(f"  Parsing dataset: {dataset_folder}")
        coll = parse_dataset(dataset_folder, stop_words)
        num_docs = coll.get_num_docs()
        print(f"  Loaded {num_docs} documents")

        if num_docs == 0:
            print(f"  Warning: No documents found for {topic_id}")
            continue

        # Compute document frequencies
        df_dict = compute_df(coll)
        print(f"  Computed {len(df_dict)} unique terms")

        # Rank documents using BM25
        print(f"  Running BM25 ranking...")
        scores = bm25_rank(coll, topic_title, df_dict)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Save output with header line matching example format
        output_file = os.path.join(output_folder, f"Baseline1_{topic_id}_Ranking.dat")
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header line matching example: "R101 (Doc_ID BM25_Score):"
            f.write(f"{topic_id} (Doc_ID BM25_Score):\n")
            # Write rankings
            for docid, score in ranked:
                f.write(f"{docid} {score}\n")

        print(f"  Saved ranking to: {output_file}")
        print(f"  Top 5 documents:")
        for i, (docid, score) in enumerate(ranked[:5], 1):
            print(f"    {i}. DocID: {docid}, BM25 Score: {score:.6f}")


if __name__ == "__main__":
    main()