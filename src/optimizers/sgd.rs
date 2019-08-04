use crate::optimizers::Optimizer;
use num::{Num, NumCast, Float, FromPrimitive};
use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::Debug;

pub struct SGD<T> {
    pub parameters: Vec<Rc<RefCell<Tensor<T>>>>,
    pub learning_rate: f32,
}

impl<T> SGD<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive + Debug,
{
    pub fn new(parameters: Vec<Rc<RefCell<Tensor<T>>>>, learning_rate: f32) -> SGD<T>{
        SGD{
            parameters: parameters,
            learning_rate: learning_rate,
        }
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Num + NumCast + Float + Clone + FromPrimitive + Debug,
{
    fn step(&mut self) {
        println!("SGD step!");

        let lr : &T = &T::from_f32(-1.0 * self.learning_rate).unwrap();
        let alpha : &T = &T::from_f32(1.0).unwrap();

        for p in &self.parameters{
            let p = &p;
            let mut p_t = p.borrow_mut();
            let p_g = p_t.gradient.as_ref().unwrap().borrow();

            // this case we can calculate sgd in here, I will change this ambiguousity

        }
    }
}
