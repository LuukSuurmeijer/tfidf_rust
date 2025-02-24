use ndarray::Array;
use sprs::{CsMat, CsMatView};

pub fn scale_csmat_by_vector(matrix: &mut CsMat<f64>, scaling_vec: &[f64]) {
    assert_eq!(
        matrix.shape().0,
        scaling_vec.len(),
        "Vector length must match the number of rows!"
    );

    // Modify the matrix in place using a mutable reference
    let iterator = matrix.outer_iterator_mut().enumerate();
    for (i, mut row) in iterator {
        for (_col_idx, value) in row.iter_mut() {
            *value *= scaling_vec[i]; // Scale each row entry
        }
    }
}

pub fn diagonal_mul(
    A: &CsMatView<f64>,
    B: &CsMatView<f64>,
) -> Array<f64, ndarray::Dim<[usize; 1]>> {
    let mut result = Array::zeros(A.shape().0);

    for ((i, A_v), B_v) in A.outer_iterator().enumerate().zip(B.outer_iterator()) {
        result[i] = A_v.dot(B_v);
    }
    result
}
