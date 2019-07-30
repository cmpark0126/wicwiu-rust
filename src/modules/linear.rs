use crate::modules::Module;
use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Debug, Display};

// #[derive(Debug)]
pub struct Linear<T> {
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    in_features: usize,
    out_features: usize,
}

impl<T> Linear<T>
where
    T: Numeric + Clone + Display + Debug + Sized,
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>, in_features: usize, out_features: usize) -> Linear<T> {
        let weight = Tensor::<T>::zeros(vec![out_features, in_features]);
        let bias = Tensor::<T>::zeros(vec![out_features]);
        let result = Tensor::<T>::zeros(vec![out_features]);
        Linear {
            inputs: vec![Rc::clone(input)],
            result: result,
            weight: weight,
            bias: bias,
            in_features: in_features,
            out_features: out_features,
        }
    }
}

impl<T> Module<T> for Linear<T>
where
    T: Numeric + Clone + Display + Debug,
{
    fn forward(&mut self) {
        println!("forward for Linear");
    }

    fn backward(&mut self) {
        println!("backward for Linear");
    }

    fn result(&self) -> &Tensor<T> {
        &self.result
    }

    fn result_mut(&mut self) -> &mut Tensor<T> {
        &mut self.result
    }
}
