use num::{Num, NumCast, Float, FromPrimitive};
use crate::modules::Module;
use crate::impl_tensor::{add, square, sum, mul_with_constant};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub struct MSE<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    subtract: Rc<RefCell<Tensor<T>>>,
    squred: Rc<RefCell<Tensor<T>>>,
    result: Rc<RefCell<Tensor<T>>>,
}

impl<T> MSE<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>,
                target: &Rc<RefCell<Box<dyn Module<T>>>>) -> MSE<T>{
        let input_result = &input.borrow().result();
        let input_result_t = &input_result.borrow();
        let target_result = &target.borrow().result();
        let target_result_t = &target_result.borrow();

        if input_result_t.shape.rank > 1 {
            panic!("Rank of input result tensor is less than 2, but got {}.",
                    input.borrow().result().borrow().shape.rank);
        }

        if target_result_t.shape.rank > 1 {
            panic!("Rank of target result tensor is less than 2, but got {}.",
                    target.borrow().result().borrow().shape.rank);
        }

        let subtract = Tensor::zeros_like(Rc::clone(input_result));

        let squred = Tensor::zeros_like(Rc::clone(input_result));

        let result = Tensor::zeros(vec![], true);

        MSE{
            inputs: vec![Rc::clone(input), Rc::clone(target)],
            subtract: Rc::new(RefCell::new(subtract)),
            squred: Rc::new(RefCell::new(squred)),
            result: Rc::new(RefCell::new(result)),
        }
    }
}

impl<T> Module<T> for MSE<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    fn forward(&mut self){
        // println!("forward for MSE");

        let input = &((&self.inputs[0]).borrow()).result();
        let target = &((&self.inputs[1]).borrow()).result();
        let subtract = &self.subtract;
        let squred = &self.squred;
        let result = &self.result;

        let alpha : &T = &T::from_f32(1.0).unwrap();
        let beta : &T = &T::from_f32(-1.0).unwrap();

        add(input, alpha, target, beta, subtract);
        square(subtract, squred);
        sum(squred, result); // in this time, I did not use * 0.5. I will fix this algo.

    }

    fn backward(&mut self){
        // println!("backward for MSE");
        let result = &self.result;
        let result_t = result.borrow();
        let result_grad = result_t.gradient.as_ref().unwrap();
        let input = &((&self.inputs[0]).borrow()).result();
        let input_t = input.borrow();
        let input_grad = input_t.gradient.as_ref().unwrap();
        // subtract value can be use as a result of diffrenciation.
        let subtract = &self.subtract;

        mul_with_constant(subtract, &result_grad.borrow().longarray[0], input_grad);
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }
}
