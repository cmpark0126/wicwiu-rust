use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct MSE<T>{
    result: Tensor<T>
}

impl<T: Numeric + Clone + Display + Debug> MSE<T> {
    pub fn new() -> MSE<T>{
        MSE{result: Tensor::zeros(vec![])}
    }
}

impl<T: Numeric + Clone + Display + Debug> Module<T> for MSE<T> {
    fn forward(&self) -> &Tensor<T>{
        println!("forward for MSE");
        &self.result
    }

    fn backward(&self) -> &Tensor<T>{
        println!("backward for MSE");
        &self.result
    }
}
