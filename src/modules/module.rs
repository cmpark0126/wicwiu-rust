use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

pub trait Module<T> {
    fn forward(&mut self);
    fn backward(&mut self);

    fn result(&self) -> Rc<RefCell<Tensor<T>>>;
    // fn result_mut(&mut self) -> &mut Rc<RefCell<Tensor<T>>>;

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor<T>>>>{
        vec![]
    }

    fn is_tensorholder(&self) -> bool{
        false
    }

    fn set_result(&mut self, result: Rc<RefCell<Tensor<T>>>){
        panic!("Only for tensorholder");
    }
}
