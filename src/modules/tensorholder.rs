use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

#[derive(Debug)]
pub struct Tensorholder<T>{
    tensor: Tensor<T>,
}

impl<T> Tensorholder<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(dim: Vec<usize>) -> Tensorholder<T>{
        Tensorholder{
            tensor: Tensor::<T>::zeros(dim),
        }
    }
}

impl<T> Module<T> for Tensorholder<T>
where T: Numeric + Clone + Display + Debug
{
    fn forward(&mut self){
        panic!("forward for Tensorholder is unnecessory");
    }

    fn backward(&mut self){
        panic!("backward for Tensorholder is unnecessory");
    }

    fn result(&self) -> &Tensor<T>{
        &self.tensor
    }

    fn result_mut(&mut self) -> &mut Tensor<T>{
        &mut self.tensor
    }

    fn is_tensorholder(&self) -> bool{
        true
    }
}
