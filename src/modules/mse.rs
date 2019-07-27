use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct MSE<T>{
    result: Option<Tensor<T>>
}

impl<T: Numeric + Clone + Display + Debug> MSE<T> {
    pub fn new() -> MSE<T>{
        MSE{result: None}
    }
}

impl<T: Numeric + Clone + Display + Debug> Module<T> for MSE<T> {
    fn forward(&self) -> &Tensor<T>{
        println!("forward for MSE");
        let t: Tensor<T> = Tensor::zeros(None);
        &t
    }

    fn backward(&self) -> &Tensor<T>{
        println!("backward for MSE");
        let t: Tensor<T> = Tensor::zeros(None);
        &t
    }
}
