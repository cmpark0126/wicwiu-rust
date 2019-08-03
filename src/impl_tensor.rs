use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use num::Float;

pub fn add<T: Float>(lhs: Rc<RefCell<Tensor<T>>>,
    rhs: Rc<RefCell<Tensor<T>>>,
    out: Rc<RefCell<Tensor<T>>>){
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        let mut out = out.borrow_mut();

        if lhs.shape.rank != rhs.shape.rank{
            panic!("lhs and rhs rank must be same, \
                    but receive lhs shape rank {}, rhs shape rank {}.",
                    lhs.shape.rank,
                    rhs.shape.rank)
        }

        if lhs.shape.rank != out.shape.rank{
            panic!("lhs and out rank must be same, \
                    but receive lhs shape rank {}, out shape rank {}.",
                    lhs.shape.rank,
                    out.shape.rank)
        }

        let capacity = lhs.shape.capacity();

        if capacity != rhs.shape.capacity(){
            panic!("lhs and rhs capacity must be same, \
                    but receive lhs shape capacity {}, rhs shape capacity {}.",
                    capacity,
                    rhs.shape.capacity())
        }

        if capacity!= out.shape.capacity(){
            panic!("lhs and out capacity must be same, \
                    but receive lhs shape capacity {}, out shape capacity {}.",
                    capacity,
                    out.shape.capacity())
        }

        let alpha: T = T::one();

        for i in 0..capacity{
            out.longarray[i] = lhs.longarray[i].mul_add(alpha, rhs.longarray[i]);
        }

    }
