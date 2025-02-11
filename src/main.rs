use indexmap::IndexSet;
use sprs::{CsMat, TriMat};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use std::time::SystemTime;

mod matops;

fn read_lines_words(filename: &str) -> io::Result<Vec<Vec<String>>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let result = reader
        .lines()
        .map(|line| {
            line.unwrap() // Handle potential read errors
                .split_whitespace() // Split into words
                .map(|word| {
                    word.to_lowercase()
                        .chars()
                        .filter(|c| !c.is_ascii_punctuation())
                        .collect()
                }) // Convert each word to lowercase
                .collect::<Vec<String>>() // Collect words into Vec<String>
        })
        .collect::<Vec<Vec<String>>>(); // Collect lines into Vec<Vec<String>>

    Ok(result)
}

fn read_lines(filename: &str) -> io::Result<HashSet<String>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let result: HashSet<String> = reader.lines().map(|line| line.unwrap()).collect();

    Ok(result)
}

fn count_occurrences(document: &[usize]) -> HashMap<&usize, usize> {
    let mut counts: HashMap<&usize, usize> = HashMap::new();

    for c in document {
        *counts.entry(c).or_insert(0) += 1;
    }

    counts
}

pub struct TFIDFModel {
    index: DocumentStore,
    documents: Vec<Vec<usize>>,
    N: f64,
    T: f64,
    matrix: TriMat<f64>,
}

impl TFIDFModel {
    pub fn new(index: DocumentStore, documents: Vec<Vec<usize>>) -> Self {
        let n = index.document_index.len();
        let t = index.term_index.len();
        let matrix = TriMat::new((t, n));
        Self {
            index: index,
            documents: documents,
            N: n as f64,
            T: t as f64,
            matrix: matrix,
        }
    }

    fn document_lengths(&self) -> Vec<f64> {
        self.documents
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
        self.documents
            .iter()
            .filter(|doc| doc.contains(term))
            .count() as f64
    }

    fn idf(&self, term: &usize) -> f64 {
        let relevant_docs = self.find_across_documents(&term);
        (self.N / relevant_docs).log(10.0)
    }

    pub fn tf_idf(&self, term: &usize, document: &[usize]) -> f64 {
        self.tf(term, document) / self.idf(term)
    }

    pub fn fit_counts(&mut self) {
        let tokenized_counts: Vec<HashMap<&usize, usize>> = self
            .documents
            .iter()
            .map(|doc| count_occurrences(doc))
            .collect();
        let mut triplets: Vec<(usize, usize, usize)> = Vec::new();

        tokenized_counts.iter().enumerate().for_each(|(i, doc)| {
            doc.into_iter()
                .for_each(|(term, count)| triplets.push((**term, i, *count)))
        });

        for (i, doc, count) in triplets {
            self.matrix.add_triplet(i, doc, count as f64);
        }
    }

    pub fn precompute_idf(&self) -> Vec<f64> {
        let mut newmat: CsMat<f64> = self.matrix.to_csr();

        // get all the row indices of the matrix
        // create a vec of len = rows, each item is the number of times that term (row) occurs in any doc (col)
        let mut nnz_rows_counts = vec![0.0; newmat.rows()];
        newmat
            .iter()
            .map(|(v, (r, c))| r as usize)
            .for_each(|idx| nnz_rows_counts[idx] += 1.0);

        nnz_rows_counts
            .iter()
            .map(|count| (self.N / count).log(2.0))
            .collect()
    }

    pub fn fit(&mut self) -> CsMat<f64> {
        let reciprocal_lengths: Vec<f64> = vec![1.0; self.N as usize]
            .iter()
            .zip(self.document_lengths().iter())
            .map(|(&a, &b)| a / b)
            .collect();

        let tf = matops::scale_csr_by_vector(&self.matrix.to_csr(), &reciprocal_lengths);
        let idf = self.precompute_idf();
        let tfidf = matops::scale_csc_by_vector(&tf.transpose_into(), &idf).transpose_into();

        tfidf
    }
}

/// term_index owns the strings of the terms and can map back and forth between indices and Strings
/// other data structues just store indices instead of strings, this way you might store indices twice, but don't have to store Strings twice or fuck with borrowing
pub struct DocumentStore {
    document_index: IndexSet<String>, // Store doc indices instead of documents
    term_index: IndexSet<String>, // Store each unique word exactly once with index. get_index() = index -> word | get_index_of() = word -> index
}

fn main() {
    println!("Reading data...");
    let summaries = read_lines_words("MovieSummaries/plot_summaries.txt")
        .unwrap()
        .to_vec();
    let stopwords = read_lines("stopwords.txt").unwrap();

    println!("Indexing data...");
    let mut term_index = IndexSet::new();
    let mut document_index = IndexSet::new();

    // in one pass, populate the indices and convert words to term indices in the documents.
    let documents: Vec<Vec<usize>> = summaries
        .iter()
        .map(|doc| {
            document_index.insert_full(doc.get(0).unwrap().clone()); // insert doc into doc index
            doc.iter()
                .filter(|word| !stopwords.contains(*word))
                .map(|word| {
                    let (index, _) = term_index.insert_full(word.clone()); // insert term into term index
                    index // return the term to convert docs into terms
                })
                .collect()
        })
        .collect();

    let store = DocumentStore {
        document_index,
        term_index,
    };

    let mut model = TFIDFModel::new(store, documents);

    println!(
        "Some tfidf score: {:?}",
        model.tf_idf(&2, model.documents.get(0).unwrap())
    );

    println!(
        "Terms {:?} | {:?}",
        model.index.term_index.get_index(20),
        model.index.term_index.len()
    );
    println!(
        "Docs {:?} | {:?}",
        model.index.document_index.get_index(20),
        model.index.document_index.len()
    );

    println!("Fitting data...");
    let start = SystemTime::now();
    model.fit_counts();
    let mut finalmat = model.fit();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("it took {} seconds", duration.as_secs_f32());

    // println!(
    //     "shape: {:?}, non zero entries: {:?}, total {:?}, ratio {:?}",
    //     finalmat.shape(),
    //     finalmat.nnz() as f64,
    //     finalmat.rows() as f64 * finalmat.cols() as f64,
    //     finalmat.nnz() as f64 / (finalmat.rows() as f64 * finalmat.cols() as f64)
    // );

    // let indices: Vec<(&f64, (usize, usize))> = finalmat.iter().collect();
}
