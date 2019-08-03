use crate::optimizers::Optimizer;
use num::{Num, NumCast, Float, FromPrimitive};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Debug, Display};

pub struct SGD<T> {
    pub parameters: Vec<Rc<RefCell<Tensor<T>>>>,
    pub learning_rate: f32,
}

impl<T> SGD<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive,
{
    pub fn new(parameters: Vec<Rc<RefCell<Tensor<T>>>>, learning_rate: f32) -> SGD<T>{
        SGD{
            parameters: parameters,
            learning_rate: learning_rate,
        }
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive,
{
    fn step(&mut self) {
        println!("SGD step!");
    }
}
