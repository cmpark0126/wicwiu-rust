use num::{Num, Float};
use crate::modules::Module;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct Sigmoid<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Rc<RefCell<Tensor<T>>>,
}

impl<T> Sigmoid<T>
where T: Num + Float + Clone
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>,) -> Sigmoid<T>{
        let result = input.borrow().result().borrow().clone();

        Sigmoid{
            inputs: vec![Rc::clone(input)],
            result: Rc::new(RefCell::new(result)),
        }
    }
}

impl<T> Module<T> for Sigmoid<T>
where T: Num + Float + Clone
{
    fn forward(&mut self){
        println!("forward for Sigmoid");
    }

    fn backward(&mut self){
        println!("backward for Sigmoid");
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }

    // fn result_mut(&mut self) -> &mut Tensor<T>{
    //     &mut self.result
    // }
}
