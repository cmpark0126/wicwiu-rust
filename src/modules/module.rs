use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub trait Module<T> {
    fn forward(&mut self);
    fn backward(&mut self);

    fn result(&self) -> Rc<RefCell<Tensor<T>>>;
    // fn result_mut(&mut self) -> &mut Rc<RefCell<Tensor<T>>>;

    fn parameters(&self) -> Vec<&Tensor<T>>{
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>>{
        vec![]
    }

    fn is_tensorholder(&self) -> bool{
        false
    }
}
