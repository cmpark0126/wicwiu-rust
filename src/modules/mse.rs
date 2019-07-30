use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

// #[derive(Debug)]
pub struct MSE<T>{
    inputs: Vec<Box<dyn Module<T>>>,
    result: Tensor<T>,
}

impl<T> MSE<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(input: Box<dyn Module<T>>) -> MSE<T>{
        let t = input.result();

        if t.shape.rank > 1 {
            panic!("Rank of input result tensor is less than 2, but got {}.", t.shape.rank);
        }

        let result = input.result().clone();

        MSE{
            inputs: vec![input],
            result: result,
        }
    }
}

impl<T> Module<T> for MSE<T>
where T: Numeric + Clone + Display + Debug
{
    fn forward(&mut self){
        println!("forward for MSE");
    }

    fn forward_prev_node(&mut self){
    }

    fn backward(&mut self){
        println!("backward for MSE");
    }

    fn result(&self) -> &Tensor<T>{
        &self.result
    }

    fn result_mut(&mut self) -> &mut Tensor<T>{
        &mut self.result
    }
}
