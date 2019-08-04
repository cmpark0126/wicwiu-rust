// use crate::tensor::Tensor;
// use std::rc::Rc;
// use std::cell::RefCell;

pub trait Optimizer<T> {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
