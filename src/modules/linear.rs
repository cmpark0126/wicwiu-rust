use crate::modules::Module;
use crate::impl_tensor::{matmul, add};
use num::{Num, NumCast, Float, FromPrimitive};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

// #[derive(Debug)]
pub struct Linear<T> {
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Rc<RefCell<Tensor<T>>>,
    weight: Rc<RefCell<Tensor<T>>>,
    bias: Rc<RefCell<Tensor<T>>>,
    in_features: usize,
    out_features: usize,
}

impl<T> Linear<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive + Debug,
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>, in_features: usize, out_features: usize) -> Linear<T> {
        let weight = Tensor::<T>::ones(vec![out_features, in_features], true);
        let bias = Tensor::<T>::zeros(vec![out_features], true);
        let result = Tensor::<T>::zeros(vec![out_features], true);
        Linear {
            inputs: vec![Rc::clone(input)],
            result: Rc::new(RefCell::new(result)),
            weight: Rc::new(RefCell::new(weight)),
            bias: Rc::new(RefCell::new(bias)),
            in_features: in_features,
            out_features: out_features,
        }
    }
}

impl<T> Module<T> for Linear<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive + Debug,
{
    fn forward(&mut self) {
        println!("forward for Linear");
        let input = &((&self.inputs[0]).borrow()).result();
        let weight = &self.weight;
        let bias = &self.bias;
        let middle_result = &Rc::new(RefCell::new(bias.borrow().clone()));
        let result = &self.result;

        matmul(weight, input, middle_result);
        add(middle_result, &T::one(), bias, &T::one(), result);
    }

    fn backward(&mut self) {
        println!("backward for Linear");
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor<T>>>>{
        vec![Rc::clone(&self.weight), Rc::clone(&self.bias),]
    }

    // fn result_mut(&mut self) -> &mut Rc<RefCell<Tensor<T>>> {
    //     &mut self.result.borrow_mut()
    // }
}
