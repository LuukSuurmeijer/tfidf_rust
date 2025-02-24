use rustc_hash::FxHashSet;
use sprs::{CsMat, DenseVector};
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::time::SystemTime;
use tfidf::*; // Replace `my_project` with your actual crate name
mod matops;

// fn read_lines_words(filename: &str) -> io::Result<Vec<Vec<String>>> {
//     let file = File::open(filename)?;
//     let reader = io::BufReader::new(file);

//     let result = reader
//         .lines()
//         .map(|line| {
//             line.unwrap() // Handle potential read errors
//                 .split_whitespace() // Split into words
//                 .map(|word| {
//                     word.to_lowercase()
//                         .chars()
//                         .filter(|c| !c.is_ascii_punctuation())
//                         .collect()
//                 }) // Convert each word to lowercase
//                 .collect::<Vec<String>>() // Collect words into Vec<String>
//         })
//         .collect::<Vec<Vec<String>>>(); // Collect lines into Vec<Vec<String>>

//     Ok(result)
// }

pub fn read_lines_words(filename: &str) -> io::Result<Vec<Vec<String>>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?; // read entire file to 1 string

    let mut result = Vec::with_capacity(500_000); // Preallocate memory

    for line in contents.lines() {
        let words: Vec<String> = line
            .split_whitespace()
            .map(|word| {
                let mut word = word.to_ascii_lowercase();
                word.retain(|c| !c.is_ascii_punctuation()); // In-place filtering
                word.shrink_to_fit(); // Reduce memory overhead
                word
            })
            .collect();

        result.push(words);
    }

    Ok(result.to_vec())
}

pub fn read_lines(filename: &str) -> io::Result<FxHashSet<String>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);

    let mut result = FxHashSet::default();

    for line in reader.lines() {
        result.insert(line?); // Use `?` instead of `.unwrap()`
    }

    Ok(result)
}

fn argmax(vec: &[f64]) -> Option<usize> {
    vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
}

fn main() {
    let start = SystemTime::now();
    let summaries = read_lines_words("MovieSummaries/plot_summaries.txt").expect("Failed to read");
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("Reading took {} seconds", duration.as_secs_f32());

    let stopwords = read_lines("stopwords.txt").unwrap();

    let mut model = TFIDFModel::new(summaries, Some(stopwords));
    let some_doc = model.index.documents.first().unwrap();
    println!("Some tfidf score: {:?}", model.tf_idf(&2, some_doc));

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

    let start: SystemTime = SystemTime::now();
    model.fit();
    let end: SystemTime = SystemTime::now();
    let duration: std::time::Duration = end.duration_since(start).unwrap();
    println!("Fitting TFIDF took {} seconds", duration.as_secs_f32());

    let start: SystemTime = SystemTime::now();
    println!("{:?}", model.query("Love  is a prostitute looking to get out of the business, but unfortunately her last gig is for three psychotic cultists who've just escaped from a mental hospital and are trawling for victims. Director Sean Cain  pulls out all the stops in this macabre, character-driven nightmare, a gore-filled love letter to midnight movie fans everywhere.".to_string()));
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("Querying took {}", duration.as_secs_f32())
}
