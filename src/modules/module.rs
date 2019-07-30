use crate::tensor::Tensor;

pub trait Module<T> {
    fn forward(&mut self);
    fn forward_prev_node(&mut self);
    fn backward(&mut self);

    fn result(&self) -> &Tensor<T>;
    fn result_mut(&mut self) -> &mut Tensor<T>;

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
