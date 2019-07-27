use crate::tensor::Tensor;

pub trait Module<T> {
    fn forward(&self) -> &Tensor<T>;
    fn backward(&self) -> &Tensor<T>;
}
