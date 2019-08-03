use num::{Num, Float};
use crate::tensor::Tensor;
use crate::modules::Module;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct NeuralNetwork<T>{
    pub module_list: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
}

impl<T> NeuralNetwork<T>
where T: Num + Float + Clone
{
    pub fn new() -> NeuralNetwork<T>{
        NeuralNetwork{
            module_list: Vec::<Rc<RefCell<Box<dyn Module<T>>>>>::new(),
        }
    }

    pub fn push(&mut self, module: Box<dyn Module<T>>) -> Rc<RefCell<Box<dyn Module<T>>>>{
        self.module_list.push(Rc::new(RefCell::new(module)));
        Rc::clone(&self.module_list[self.module_list.len() - 1])
    }
}

impl<T> Module<T> for NeuralNetwork<T>
where
    T: Num + Float + Clone,
{
    fn forward(&mut self){
        for module in &self.module_list{
            let mut module_mut = module.borrow_mut();
            if module_mut.is_tensorholder() == false {
                module_mut.forward();
            }
        }
    }

    fn backward(&mut self){
        let len = self.module_list.len();

        for idx in (0..len).rev(){
            let mut module_mut = self.module_list[idx].borrow_mut();
            if module_mut.is_tensorholder() == false {
                module_mut.backward();
            }
        }
    }

    fn result(&self) -> Rc<RefCell<Tensor<T>>> {
        Rc::clone(&self.module_list[self.module_list.len() - 1].borrow().result())
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor<T>>>>{
        let mut v = Vec::<Rc<RefCell<Tensor<T>>>>::new();

        for module in &self.module_list{
            for p in module.borrow().parameters(){
                v.push(p);
            }
        }

        v
    }
}
