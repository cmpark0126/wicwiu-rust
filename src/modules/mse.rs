use num::{Num, NumCast, Float, FromPrimitive};
use crate::modules::Module;
use crate::impl_tensor::{matmul, add};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct MSE<T>{
    inputs: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    result: Rc<RefCell<Tensor<T>>>,
}

impl<T> MSE<T>
where T: Num + NumCast + Float + Clone + FromPrimitive
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
where T: Num + NumCast + Float + Clone + FromPrimitive
{
    fn forward(&mut self){
        println!("forward for MSE");

    }

    fn backward(&mut self){
        println!("backward for MSE");
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.result)
    }

    // fn result_mut(&mut self) -> &mut Tensor<T>{
    //     &mut self.result
    // }
}
