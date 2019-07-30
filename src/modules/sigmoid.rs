use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct Sigmoid<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Tensor<T>,
}

impl<T> Sigmoid<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(input: Box<dyn Module<T>>) -> Sigmoid<T>{
        let result = input.result().clone();

        Sigmoid{
            inputs: vec![Rc::new(RefCell::new(input))],
            result: result,
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
}
