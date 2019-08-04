use num::{Num, NumCast, Float, FromPrimitive};
use crate::modules::Module;
use crate::impl_tensor::{sigmoid, sigmoid_backward};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub struct Sigmoid<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Rc<RefCell<Tensor<T>>>,
}

impl<T> Sigmoid<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>,) -> Sigmoid<T>{
        let result = Tensor::zeros_like(input.borrow().result());

        Sigmoid{
            inputs: vec![Rc::clone(input)],
            result: Rc::new(RefCell::new(result)),
        }
    }
}

impl<T> Module<T> for Sigmoid<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    fn forward(&mut self){
        println!("forward for Sigmoid");
        let input = &((&self.inputs[0]).borrow()).result();
        let result = &self.result;

        sigmoid(input, result);
    }

    fn backward(&mut self){
        println!("backward for Sigmoid");
        let result = &self.result;
        let result_t = &result.borrow();
        let result_grad = result_t.gradient.as_ref().unwrap();
        let input = &((&self.inputs[0]).borrow()).result();
        let input_t = &input.borrow();
        let input_grad = input_t.gradient.as_ref().unwrap();

        sigmoid_backward(result, result_grad, input_grad);
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }
}
