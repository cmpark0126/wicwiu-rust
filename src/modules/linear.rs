use std::fmt::{Display, Debug};
use crate::numeric::Numeric;
use crate::modules::Module;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Linear<T>{
    result: Tensor<T>
}

impl<T> Linear<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(weight: &Module<T>, input: &Module<T>) -> Linear<T>{
        let w_s = &weight.result().shape;
        let i_s = &input.result().shape;

        if w_s.rank != 2 || i_s.rank != 1 {
            panic!("Rank of weight result tensor must be 2\
                    and rank of input result tensor must be 1\
                    , but got {}, {} respectively."
                    , w_s.rank, i_s.rank);
        }

        let w_d = &w_s.dim;
        let i_d = &i_s.dim;

        if w_d[1] != i_d[0] {
            panic!("");
        }

        Linear{result: Tensor::<T>::zeros(vec![w_d[0].clone()])}
    }
}

impl<T> Module<T> for Linear<T>
where T: Numeric + Clone + Display + Debug
{
    fn forward(&self){
        println!("forward for Linear");
    }

    fn backward(&self){
        println!("backward for Linear");
    }

    fn result(&self) -> &Tensor<T>{
        &self.result
    }
}
