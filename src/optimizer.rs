use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Optimizer<T>{
    pub parameter_list: Vec<Tensor<T>>,
}
