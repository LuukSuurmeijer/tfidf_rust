use indexmap::IndexSet;
use sprs::{CsMat, TriMat, TriMatBase};
use std::collections::HashMap;

use std::fs::File;
use std::io::{self, BufRead};

fn read_lines(filename: &str) -> io::Result<Vec<Vec<String>>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let result = reader
        .lines()
        .map(|line| {
            line.unwrap() // Handle potential read errors
                .split_whitespace() // Split into words
                .map(|word| word.to_lowercase()) // Convert each word to lowercase
                .collect::<Vec<String>>() // Collect words into Vec<String>
        })
        .collect::<Vec<Vec<String>>>(); // Collect lines into Vec<Vec<String>>

    Ok(result)
}

fn count_occurences(document: &[usize]) -> HashMap<&usize, usize> {
    let counts: HashMap<&usize, usize> = document.iter().fold(HashMap::new(), |mut map, c| {
        *map.entry(c).or_insert(0) += 1;
        map
    });

    counts
}

fn get_document_frequencies(documents: &[Vec<usize>]) -> HashMap<&usize, usize> {
    let mut term_frequenices: HashMap<&usize, usize> = HashMap::new();
    for doc in documents {
        term_frequenices.extend(count_occurences(doc));
    }
    term_frequenices
}

/// term_index owns the strings of the terms and can map back and forth between indices and Strings
/// other data structues just store indices instead of strings, this way you might store indices twice, but don't have to store Strings twice or fuck with borrowing
pub struct DocumentStore {
    document_index: IndexSet<String>, // Store doc indices instead of documents
    term_index: IndexSet<String>, // Store each unique word exactly once with index. get_index() = index -> word | get_index_of() = word -> index
}

fn main() {
    let summaries = read_lines("MovieSummaries/plot_summaries.txt").unwrap();

    let mut term_index = IndexSet::new();
    let mut document_index = IndexSet::new();

    // in one pass, populate the indices and convert words to term indices in the documents.
    let documents: Vec<Vec<usize>> = summaries
        .iter()
        .map(|doc| {
            document_index.insert_full(doc.get(0).unwrap().clone()); // insert doc into doc index
            doc.iter()
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

    let counts = get_document_frequencies(&documents);
    let key_with_max_value = counts.values().max_by_key(|&&v| v).unwrap();

    println!(
        "Terms {:?} | {:?}",
        store.term_index.get_index(20),
        store.term_index.len()
    );
    println!(
        "Docs {:?} | {:?}",
        store.document_index.get_index(20),
        store.document_index.len()
    );

    println!("{:?}", store.term_index.get_index(*key_with_max_value))
}
