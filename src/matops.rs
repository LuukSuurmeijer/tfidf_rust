use indicatif::ProgressBar;
use rayon::prelude::*;
use sprs::{CsMat, TriMat};

pub fn scale_csc_by_vector(matrix: &CsMat<f64>, col_sums: &[f64]) -> CsMat<f64> {
    assert_eq!(
        matrix.cols(),
        col_sums.len(),
        "Vector length must match the number of columns!"
    );

    let bar = ProgressBar::new(matrix.cols() as u64);
    let mut scaled_matrix = matrix.clone(); // Clone to modify
    for (col_idx, mut col) in scaled_matrix.outer_iterator_mut().enumerate() {
        for (_, value) in col.iter_mut() {
            *value *= col_sums[col_idx]; // Scale by corresponding column sum
        }
        bar.inc(1);
    }

    scaled_matrix
}

pub fn scale_csr_by_vector(matrix: &CsMat<f64>, col_scaling: &[f64]) -> CsMat<f64> {
    assert_eq!(
        matrix.cols(),
        col_scaling.len(),
        "Vector length must match the number of columns!"
    );
    let mut scaled_matrix = matrix.clone(); // Clone to modify
    for mut row in scaled_matrix.outer_iterator_mut() {
        for (col_idx, value) in row.iter_mut() {
            *value *= col_scaling[col_idx]; // Scale each column entry
        }
    }

    scaled_matrix
}
