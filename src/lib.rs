use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use sprs::{CsMat, CsVec, TriMat};
mod matops;
use std::time::SystemTime;

fn count_occurrences(document: &[usize]) -> FxHashMap<usize, usize> {
    let mut counts = FxHashMap::default();
    for &c in document {
        *counts.entry(c).or_insert(0) += 1;
    }
    counts
}

pub fn fit_counts(documents: &[Vec<usize>], mut matrix: TriMat<f64>) -> TriMat<f64> {
    for (doc_index, doc) in documents.iter().enumerate() {
        let counts = count_occurrences(doc);

        for (term, count) in counts {
            matrix.add_triplet(doc_index, term, count as f64);
        }
    }

    matrix
}

pub fn create_indices(
    summaries: Vec<Vec<String>>,  // Avoids unnecessary cloning
    stopwords: FxHashSet<String>, // Pass stopwords by reference
) -> DocumentStore {
    let mut term_index = IndexSet::new();
    let mut document_index = IndexSet::new();

    let documents: Vec<Vec<usize>> = summaries
        .iter()
        .filter_map(|doc| {
            let doc_name = doc.first()?; // Gracefully handle empty documents
            document_index.insert_full(doc_name.clone()); // Insert document name

            let terms = doc
                .iter()
                .filter_map(|word| {
                    if stopwords.contains(word) {
                        None // Skip stopwords efficiently
                    } else {
                        Some(term_index.insert_full(word.clone()).0) // Insert and return index
                    }
                })
                .collect::<Vec<_>>();

            Some(terms) // Only collect non-empty documents
        })
        .collect();

    DocumentStore {
        stopwords,
        documents,
        term_index,
        document_index,
    }
}

/// term_index owns the strings of the terms and can map back and forth between indices and Strings
/// other data structues just store indices instead of strings, this way you might store indices twice, but don't have to store Strings twice or fuck with borrowing
pub struct DocumentStore {
    stopwords: FxHashSet<String>,
    pub documents: Vec<Vec<usize>>,
    pub document_index: IndexSet<String>, // Store doc indices instead of documents
    pub term_index: IndexSet<String>, // Store each unique word exactly once with index. get_index() = index -> word | get_index_of() = word -> index
}

impl DocumentStore {
    pub fn new(summaries: Vec<Vec<String>>, stopwords: FxHashSet<String>) -> Self {
        create_indices(summaries, stopwords)
    }
    pub fn tokenize(&self, query: &str) -> Vec<usize> {
        query
            .split_ascii_whitespace()
            .filter_map(|word| {
                if self.stopwords.contains(word) {
                    None
                } else {
                    Some(self.term_index.get_index_of(word)?) // Insert and return index
                }
            })
            .collect()
    }

    pub fn embed(&self, query: &str) -> CsVec<f64> {
        let counts = count_occurrences(&self.tokenize(query));
        // Convert the HashMap to a vector of tuples (index, count)
        let mut pairs: Vec<(usize, f64)> = counts
            .iter()
            .map(|(&index, &count)| (index, count as f64))
            .collect();

        // Sort the pairs by the index (ascending)
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        // Unzip the sorted pairs into two separate vectors: indices and values
        let (indices, values): (Vec<usize>, Vec<f64>) = pairs.into_iter().unzip();

        // Construct the sparse vector (of size vocab_size)
        CsVec::new(self.term_index.len(), indices, values)
    }
}

pub struct TFIDFModel {
    pub index: DocumentStore,
    pub n: f64,
    pub matrix: CsMat<f64>,
}

impl TFIDFModel {
    pub fn new(documents: Vec<Vec<String>>, stopwords: Option<FxHashSet<String>>) -> Self {
        let start = SystemTime::now();
        let document_store = DocumentStore::new(documents, stopwords.unwrap_or_default());
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("Indexing took {} seconds", duration.as_secs_f32());

        let n = document_store.document_index.len();
        let t = document_store.term_index.len();

        let start = SystemTime::now();
        let matrix = fit_counts(&document_store.documents, TriMat::new((n, t))).to_csr();
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("Counting took {} seconds", duration.as_secs_f32());

        Self {
            index: document_store,
            n: n as f64,
            matrix,
        }
    }

    fn document_lengths(&self) -> Vec<f64> {
        self.index
            .documents
            .iter()
            .map(|doc| (doc.len() as f64))
            .collect()
    }

    fn tf(&self, term: &usize, document: &[usize]) -> f64 {
        let d_count = count_occurrences(document);
        let tf = d_count.get(term);

        match tf {
            Some(tf) => *tf as f64 / document.len() as f64,
            None => 0.0,
        }
    }

    fn find_across_documents(&self, term: &usize) -> f64 {
        self.index
            .documents
            .iter()
            .filter(|doc| doc.contains(term))
            .count() as f64
    }

    fn idf(&self, term: &usize) -> f64 {
        let relevant_docs = self.find_across_documents(term);
        (self.n / relevant_docs).log(2.0)
    }

    pub fn tf_idf(&self, term: &usize, document: &[usize]) -> f64 {
        self.tf(term, document) / self.idf(term)
    }

    pub fn precompute_idf(&self) -> Vec<f64> {
        // get all the row indices of the matrix
        // create a vec of len = rows, each item is the number of times that term (col) occurs in any doc (row)
        let mut idf_counts = vec![0.0; self.matrix.cols()];
        self.matrix
            .iter()
            .for_each(|(_val, (_doc_idx, term_idx))| idf_counts[term_idx] += 1.0);

        idf_counts
            .iter()
            .map(|count| (self.n / count).log(2.0))
            .collect()
    }

    pub fn fit(&mut self) {
        let reciprocal_lengths: Vec<f64> = vec![1.0; self.n as usize]
            .iter()
            .zip(self.document_lengths().iter())
            .map(|(&a, &b)| a / b)
            .collect();

        let idf = self.precompute_idf(); // <terms>

        // Scale the original matrix in place
        matops::scale_csmat_by_vector(&mut self.matrix, &reciprocal_lengths); // <docs, terms>

        // Transpose the matrix in place, modify it, and then transpose back
        // The `transpose_mut()` method modifies the matrix directly, avoiding moves
        self.matrix.transpose_mut(); // <terms, docs> for multiplications with <terms>

        matops::scale_csmat_by_vector(&mut self.matrix, &idf);
        self.matrix.transpose_mut(); // <docs, terms>
    }

    pub fn query(&self, query: String) -> &String {
        let query_vec = self.index.embed(&query);

        let numerator = (&self.matrix * &query_vec).to_dense(); // <docs> no longer sparse
        let denom = &matops::diagonal_mul(&self.matrix.view(), &self.matrix.transpose_view())
            .sqrt()
            * query_vec.dot(&query_vec).sqrt();

        let cosines = numerator / denom;
        let argmax = (0..cosines.len()) // Generate indices
            .max_by(|&i, &j| cosines[i].partial_cmp(&cosines[j]).unwrap()) // Compare using partial_cmp
            .unwrap(); // Unwrap since max_by returns an Option

        self.index.document_index.get_index(argmax).unwrap()
    }
}
