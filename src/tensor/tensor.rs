use std::sync::Arc;
use super::storage::Storage;

#[derive(Debug)]
pub enum TensorError {
    DimensionError(String),
    ShapeError(String),
    IndexError(String),
}

#[derive(Clone)]
pub struct Tensor<T> {
    storage: Arc<Storage<T>>,  // Shared ownership of the underlying data
    shape: Vec<usize>,         // Dimensions of the tensor
    strides: Vec<usize>,       // How to traverse the memory
    offset: usize,             // Starting point in storage
}

