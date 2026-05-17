import glob, os
import string
from math import log10
from stemming.porter2 import stem

def index_docs(inputpath,stop_words):
    Index = {}    # initialize the index
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        start_end = False
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            break
                if line.startswith("<text>"):
                    start_end = True
            elif line.startswith("</text>"):
                break
            else:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                for term in line.split():
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        try:
                            try:
                                Index[term][docid] += 1
                            except KeyError:
                                Index[term][docid]=1
                        except KeyError:
                            Index[term] = {docid:1}
    return Index

def likelihood_JM(I_C, I_D, Q, lamda):  # index I_C ans I_D have the form of {term:{itemId:freq}}
    # calcualte query term frequency in documents indexed by I_D
    L={}    # L is the selected inverted list
    R={}    # R is a directionary of docId:score
    D_len={} # D_len is a directionary of docId:length
    for list in I_D.items():
        for id in list[1].items():
            R[id[0]]=0.0       # get all document IDs and initialize as 0.0 for log accumulation
            D_len[id[0]]=0.5 # initialize a small non-zero value as it will be used as denominator
        if (list[0] in Q):     # select inverted lists based on the query
                L[list[0]]= I_D[list[0]]
    for q_term in Q.items(): # L may not include all query terms
        if not(q_term[0] in L):
            L[q_term[0]]={}
    for list in I_D.items():
        for id in list[1].items(): # Count term occurrences in documents
            D_len[id[0]]= D_len[id[0]] + id[1]
    # calculate query term frequency in I_C
    CF={}
    L_C={}
    for list in I_C.items():
        if (list[0] in Q):     # select inverted lists based on Q in I_C
                L_C[list[0]]= I_C[list[0]]
                CF[list[0]] = 0 # assign 0 to each query term
    for (term, doc) in L_C.items():
        for id in doc.items():
            CF[term] = CF[term] + id[1]
    C_len = 0
    for list in I_C.items():
        for id in list[1].items(): # Count term occurrences in documents
            C_len = C_len + id[1]
    # using the equation
    for (d, sd) in R.items():
        for (term, f) in L.items():
            if not(d in f):
                f[d]=0
            sd = sd + log10((1-lamda)*f[d]/D_len[d]+(lamda*CF[term]/C_len)) # see page 12 in wk5 lecture notes
        R[d] = sd
    return R

class BowDoc:
    """Bag-of-words representation of a document.

    The document has an ID, and an iterable list of terms with their
    frequencies."""

    def __init__(self, docid):
        """Constructor.

        Set the ID of the document, and initiate an empty term dictionary.
        Call add_term to add terms to the dictionary."""
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        """Add a term occurrence to the BOW representation.

        This should be called each time the term occurs in the document."""
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1

    def get_term_count(self, term):
        """Get the term occurrence count for a term.

        Returns 0 if the term does not appear in the document."""
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        """Return dictionary of term:freq pairs."""
        return self.terms

    def get_term_list(self):
        """Get sorted list of all terms occurring in the document."""
        return sorted(self.terms.keys())

    def get_docid(self):
        """Get the ID of the document."""
        return self.docid

    def __iter__(self):
        """Return an ordered iterator over term--frequency pairs.

        Each element is a (term, frequency) tuple.  They are iterated
        in term's frequency descending order."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1], reverse=True))
        """Or in term alphabetical order:
        return iter(sorted(self.terms.iteritems()))"""

    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len


class BowColl:
    """Collection of BOW documents."""

    def __init__(self):
        """Constructor.

        Creates an empty collection."""
        self.docs = {}

    def add_doc(self, doc):
        """Add a document to the collection."""
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        """Return a document by docid.

        Will raise a KeyError if there is no document with that ID."""
        return self.docs[docid]

    def get_docs(self):
        """Get the full list of documents.

        Returns a dictionary, with docids as keys, and docs as values."""
        return self.docs

    def inorder_iter(self):
        """Return an ordered iterator over the documents.

        The iterator will traverse the collection in docid order.  Modifying
        the collection while iterating over it leads to undefined results.
        Each element is a document; to find the id, call doc.get_docid()."""
        return BowCollInorderIterator(self)

    def get_num_docs(self):
        """Get the number of documents in the collection."""
        return len(self.docs)

    def __iter__(self):
        """Iterator interface.

        See inorder_iter."""
        return self.inorder_iter()


class BowCollInorderIterator:
    """Iterator over a collection."""

    def __init__(self, coll):
        """Constructor.

        Takes the collection we're going to iterator over as sole argument."""
        self.coll = coll
        self.keys = sorted(coll.get_docs().keys())
        self.i = 0

    def __iter__(self):
        """Iterator interface."""
        return self

    def next(self):
        """Get next element."""
        if self.i >= len(self.keys):
            raise StopIteration
        doc = self.coll.get_doc(self.keys[self.i])
        self.i += 1
        return doc


def parse_rcv_coll(inputpath, stop_words):
    """Parse an RCV1 data files into a collection.

    inputpath is the folder name of the RCV1 data files.  The parsed collection
    is returned.  NOTE the function performs very limited error checking."""
    # stopwords = open('common-english-words.txt', 'r')

    coll = BowColl()
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        curr_doc = None
        start_end = False
        word_count = 0
        for line in open(file_):
            line = line.strip()
            if (start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            curr_doc = BowDoc(docid)
                            break
                    continue
                if line.startswith("<text>"):
                    start_end = True
            elif line.startswith("</text>"):
                break
            elif curr_doc is not None:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

                for term in line.split():
                    word_count += 1
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        curr_doc.add_term(term)
        if curr_doc is not None:
            curr_doc.set_doc_len(word_count)
            coll.add_doc(curr_doc)
    return coll


def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl / coll.get_num_docs()


def queryParser(query, stop_words):
    """Parse query string into {term: frequency} dict."""
    query_terms = {}
    line = (query.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation))))
    for term in line.split():
        term = stem(term.lower())
        if len(term) > 2 and term not in stop_words:
            try:
                query_terms[term] += 1
            except KeyError:
                query_terms[term] = 1
    return query_terms


def compute_df(coll):
    """Compute document frequency for each term."""
    df = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            df[term] = df.get(term, 0) + 1
    return df


def parse_topics(topics_file):
    """Parse Topics.txt and extract topic ID and title."""
    import re
    topics = {}
    with open(topics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    topic_blocks = content.split('</Topic>')
    for block in topic_blocks:
        num_match = re.search(r'<num>\s*Number:\s*(\w+)', block)
        if not num_match:
            continue
        topic_id = num_match.group(1)
        title_match = re.search(r'<title>\s*([^<\n]+)', block)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        topics[topic_id] = title
    return topics


def bm_25(coll, q, df_):
    # This function calculate BM25 scores for all documents in the coll for a given query q
    # Since no relevance feedback is given, we use R=0 and ri=0
    q_terms = queryParser(q, stop_words)   # Parse query string using queryParser
    # k1 and b values are from the lecture slides
    k1=1.2
    b=0.75
    # k2 values are set in the middle range as 500
    k2=500
    #R=0 since no relevance feedback
    #ri = 0 since no relevance feedback
    N = coll.get_num_docs()

    #N = len(coll) total number of document
    avdl = avg_doc_len(coll) #average length of doc in the collection

    scores = {}

    for docid, doc in coll.get_docs().items():
        K = k1 * ((1 - b) + b * (doc.get_doc_len() / avdl)) # equation from the lecture slides
        score = 0.0

        for term, qfi in q_terms.items():
            ni = df_.get(term, 0)           # number of docs containing term
            fi = doc.terms.get(term, 0)     # term frequency in this doc
            if ni == 0 or fi == 0:
                continue

            # BM25 formula with query-frequency factor k2 per assignment spec:
            # BM25 contribution per term: log10(1 + RSJ * tf_component * q_component)
            # where RSJ = (N - n_t + 0.5) / (n_t + 0.5)
            # tf_component = ((k1+1) * f_tD) / (K + f_tD)
            # q_component = ((k2+1) * qf_t) / (k2 + qf_t)
            tf_component = ((k1 + 1) * fi) / (K + fi)
            q_component = ((k2 + 1) * qfi) / (k2 + qfi)
            rsj = 1 + ((N - ni + 0.5) / (ni + 0.5))
            score += log10(rsj) * tf_component * q_component

        scores[docid] = score

    return scores


if __name__ == '__main__':

    # Setup paths
    base_dir = "/Users/niceenb/Documents/647/Assignment2"
    topics_file = os.path.join(base_dir, "Topics.txt")
    doc_collection_base = os.path.join(base_dir, "Doc_Collection")
    output_folder = os.path.join(base_dir, "ModelOutputs")
    stopwords_file = os.path.join(base_dir, "common-english-words.txt")

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load stop words
    stopwords_f = open(stopwords_file, 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    # Parse topics
    topics = parse_topics(topics_file)
    print(f"Loaded {len(topics)} topics")

    # Process each topic
    for topic_id, topic_title in sorted(topics.items()):
        print(f"\nProcessing {topic_id}: {topic_title}")

        # Determine dataset folder (R101 → Dataset101, R102 → Dataset102, etc.)
        dataset_num = topic_id[1:]  # Remove 'R' prefix
        dataset_folder = os.path.join(doc_collection_base, f"Dataset{dataset_num}")

        # Parse dataset
        print(f"  Parsing dataset...")
        coll = parse_rcv_coll(dataset_folder, stop_words)
        num_docs = coll.get_num_docs()
        print(f"  Loaded {num_docs} documents")

        if num_docs == 0:
            print(f"  Warning: No documents found")
            continue

        # Compute document frequencies
        df_dict = compute_df(coll)

        # Rank documents using BM25
        scores = bm_25(coll, topic_title, df_dict)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Save output with header line
        output_file = os.path.join(output_folder, f"Baseline1_{topic_id}_Ranking.dat")
        with open(output_file, 'w') as f:
            f.write(f"{topic_id} (Doc_ID BM25_Score):\n")
            for docid, score in ranked:
                f.write(f"{docid} {score}\n")

        print(f"  Saved: {output_file}")
        
        # ----------------- Baseline 2: Jelinek-Mercer (JM) -----------------
        try:
            print(f"  Running Jelinek-Mercer (JM) ranking...")
            # build inverted index for this dataset (term -> {docid:freq})
            Index_D = index_docs(dataset_folder, stop_words)
            # use the same collection-level index (collection = dataset) for CF values
            Index_C = Index_D
            # restore working directory
            os.chdir(base_dir)

            # parse topic title into query term-frequency dict
            Q_dict = queryParser(topic_title, stop_words)

            # compute JM scores (lambda = 0.3 as required)
            jm_scores = likelihood_JM(Index_C, Index_D, Q_dict, 0.3)

            # sort and write output in the requested format
            ranked_jm = sorted(jm_scores.items(), key=lambda x: x[1], reverse=True)
            output_file_jm = os.path.join(output_folder, f"Baseline2_{topic_id}_Ranking.dat")
            with open(output_file_jm, 'w', encoding='utf-8') as f2:
                f2.write(f"Query is the topic title = \"{topic_title}\"\n")
                f2.write(f"{sum(Q_dict.values())}\n")
                f2.write("Doc_ID JM_Score\n")
                for docid, score in ranked_jm:
                    f2.write(f"{docid} {score}\n")

            print(f"  Saved JM ranking to: {output_file_jm}")
        except Exception as e:
            print(f"  JM ranking failed for {topic_id}: {e}")
        


