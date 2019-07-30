use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::optimizer::Optimizer;

pub struct NeuralNetwork<T>{
    pub module_list: Vec<Box<dyn Module<T>>>,
    pub optimizer: Optimizer<T>,
}

impl<T> NeuralNetwork<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(optimizer: Optimizer<T>) -> NeuralNetwork<T>{
        NeuralNetwork{
            module_list: Vec::<Box<dyn Module<T>>>::new(),
            optimizer: optimizer,
        }
    }
}
