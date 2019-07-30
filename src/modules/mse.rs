use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct MSE<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Tensor<T>,
}

impl<T> MSE<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>,) -> MSE<T>{
        // let t = ;

        if input.borrow().result().shape.rank > 1 {
            panic!("Rank of input result tensor is less than 2, but got {}.", input.borrow().result().shape.rank);
        }

        let result = input.borrow().result().clone();

        MSE{
            inputs: vec![Rc::clone(input)],
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
