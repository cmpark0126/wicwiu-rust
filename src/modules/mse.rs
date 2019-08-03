use num::{Num, NumCast, Float, FromPrimitive};
use crate::modules::Module;
use crate::impl_tensor::{add, sum};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub struct MSE<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Rc<RefCell<Tensor<T>>>,
}

impl<T> MSE<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    pub fn new(input: &Rc<RefCell<Box<dyn Module<T>>>>, target: &Rc<RefCell<Box<dyn Module<T>>>>) -> MSE<T>{
        // let t = ;

        if input.borrow().result().borrow().shape.rank > 1 {
            panic!("Rank of input result tensor is less than 2, but got {}.", input.borrow().result().borrow().shape.rank);
        }

        if target.borrow().result().borrow().shape.rank > 1 {
            panic!("Rank of target result tensor is less than 2, but got {}.", target.borrow().result().borrow().shape.rank);
        }

        let result = Tensor::zeros(vec![], true);

        MSE{
            inputs: vec![Rc::clone(input), Rc::clone(target)],
            result: Rc::new(RefCell::new(result)),
        }
    }
}

impl<T> Module<T> for MSE<T>
where T: Num + NumCast + Float + Clone + FromPrimitive + Debug
{
    fn forward(&mut self){
        println!("forward for MSE");

        let input = &((&self.inputs[0]).borrow()).result();
        let target = &((&self.inputs[1]).borrow()).result();
        let middle_result = &Rc::new(RefCell::new(input.borrow().clone()));
        let result = &self.result;

        let alpha : &T = &T::from_f32(1.0).unwrap();
        let beta : &T = &T::from_f32(-1.0).unwrap();

        add(input, alpha, target, beta, middle_result);
        sum(middle_result, result);

    }

    fn backward(&mut self){
        println!("backward for MSE");
        let result = &self.result.borrow();
        let result_grad = result.gradient.as_ref().unwrap();
        // println!("{:?}", result_grad.borrow());

    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }

    // fn result_mut(&mut self) -> &mut Tensor<T>{
    //     &mut self.result
    // }
}
