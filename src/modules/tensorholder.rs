use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

#[derive(Debug)]
pub struct Tensorholder<T>{
    tensor: Rc<RefCell<Tensor<T>>>,
}

impl<T> Tensorholder<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(dim: Vec<usize>) -> Tensorholder<T>{
        Tensorholder{
            tensor: Rc::new(RefCell::new(Tensor::<T>::zeros(dim, true))),
        }
    }
}

impl<T> Module<T> for Tensorholder<T>
where T: Numeric + Clone + Display + Debug
{
    fn forward(&mut self){
        // panic!("forward for Tensorholder is unnecessory");
    }

    fn backward(&mut self){
        // panic!("backward for Tensorholder is unnecessory");
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.tensor)
    }

    fn is_tensorholder(&self) -> bool{
        true
    }
}
