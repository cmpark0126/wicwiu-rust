use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Tensorholder<T>{
    tensor: Tensor<T>,
    need_to_forward: bool,
    need_to_backward: bool,
}

impl<T> Tensorholder<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(dim: Vec<usize>) -> Tensorholder<T>{
        Tensorholder{
            tensor: Tensor::<T>::zeros(dim),
            need_to_forward: false,
            need_to_backward: false,
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

    fn need_to_forward(&self) -> &bool{
        &self.need_to_forward
    }
    fn need_to_forward_mut(&mut self) -> &mut bool{
        &mut self.need_to_forward
    }

    fn need_to_backward(&self) -> &bool{
        &self.need_to_backward
    }
    fn need_to_backward_mut(&mut self) -> &mut bool{
        &mut self.need_to_forward
    }
}
