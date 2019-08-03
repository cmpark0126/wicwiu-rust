// use crate::tensor::Tensor;
// use std::rc::Rc;
// use std::cell::RefCell;

pub trait Optimizer<T> {
    fn step(&mut self);
}
