use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

// #[derive(Debug)]
pub struct Sigmoid<T>{
    result: Tensor<T>,
    need_to_forward: bool,
    need_to_backward: bool,
}

impl<T> Sigmoid<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(input: Box<dyn Module<T>>) -> Sigmoid<T>{
        Sigmoid{
            result: input.result().clone(),
            need_to_forward: true,
            need_to_backward: false,
        }
    }
}

impl<T> Module<T> for Sigmoid<T>
where T: Numeric + Clone + Display + Debug
{
    fn forward(&mut self){
        println!("forward for Sigmoid");
    }

    fn backward(&mut self){
        println!("backward for Sigmoid");
    }

    fn result(&self) -> &Tensor<T>{
        &self.result
    }

    fn result_mut(&mut self) -> &mut Tensor<T>{
        &mut self.result
    }

    fn need_to_forward(&self) -> &bool{
        &self.need_to_forward
    }
    fn need_to_forward_mut(&mut self) -> &mut bool{
        &mut self.need_to_forward
    }

    fn need_to_backward(&self) -> &bool{
        &self.need_to_backward
    }
    fn need_to_backward_mut(&mut self) -> &mut bool{
        &mut self.need_to_forward
    }
}
