use crate::tensor::Tensor;

pub trait Module<T> {
    fn forward(&mut self);
    fn backward(&mut self);

    fn result(&self) -> &Tensor<T>;
    fn result_mut(&mut self) -> &mut Tensor<T>;

    fn parameters(&self) -> Option<Vec<&Tensor<T>>>{
        None
    }

    fn parameters_mut(&mut self) -> Option<Vec<&mut Tensor<T>>>{
        None
    }

    fn is_tensorholder(&self) -> bool{
        false
    }
}
