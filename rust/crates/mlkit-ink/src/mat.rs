//! A row-major dense `f32` matrix, deliberately minimal.
//!
//! Three shapes flow through the pipeline — features `[T, 10]`, hidden
//! activations `[T, 2H]` and logits `[T, C]` — and all we ever do is index rows
//! and multiply by a weight block. A linear-algebra dependency would buy
//! nothing and cost wasm binary size, so this is the whole abstraction.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Panics unless `data.len() == rows * cols`; callers build these from
    /// shapes they just computed, so a mismatch is a bug, not bad input.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrix data does not match {rows}x{cols}"
        );
        Mat { rows, cols, data }
    }

    pub fn from_rows(cols: usize, rows_iter: impl IntoIterator<Item = Vec<f32>>) -> Self {
        let mut data = Vec::new();
        let mut rows = 0;
        for row in rows_iter {
            assert_eq!(row.len(), cols, "row width does not match {cols}");
            data.extend_from_slice(&row);
            rows += 1;
        }
        Mat { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let cols = self.cols;
        &mut self.data[i * cols..(i + 1) * cols]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.cols)
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;

    fn index(&self, (r, c): (usize, usize)) -> &f32 {
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f32 {
        &mut self.data[r * self.cols + c]
    }
}
