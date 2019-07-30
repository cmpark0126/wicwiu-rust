use crate::numeric::Numeric;
use crate::modules::Module;
use crate::optimizer::Optimizer;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt::{Display, Debug};

pub struct NeuralNetwork<T>{
    pub module_list: Vec<Rc<RefCell<Box<dyn Module<T>>>>>,
    pub optimizer: Optimizer<T>,
}

impl<T> NeuralNetwork<T>
where T: Numeric + Clone + Display + Debug
{
    pub fn new(optimizer: Optimizer<T>) -> NeuralNetwork<T>{
        NeuralNetwork{
            module_list: Vec::<Rc<RefCell<Box<dyn Module<T>>>>>::new(),
            optimizer: optimizer,
        }
    }

    pub fn push(&mut self, module: Box<dyn Module<T>>) -> Rc<RefCell<Box<dyn Module<T>>>>{
        self.module_list.push(Rc::new(RefCell::new(module)));
        Rc::clone(&self.module_list[self.module_list.len() - 1])
    }

    pub fn forward(&mut self){
        for module in &self.module_list{
            module.borrow_mut().forward();
        }
    }
}
