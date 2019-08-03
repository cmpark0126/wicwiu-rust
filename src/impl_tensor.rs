use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use num::Float;

pub fn add<T: Float>(lhs: &Rc<RefCell<Tensor<T>>>,
    rhs: &Rc<RefCell<Tensor<T>>>,
    out: &Rc<RefCell<Tensor<T>>>){
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

pub fn sigmoid<T: Float>(in_t: &Rc<RefCell<Tensor<T>>>,
    out_t: &Rc<RefCell<Tensor<T>>>){
        let in_t = in_t.borrow();
        let mut out_t = out_t.borrow_mut();

        if in_t.shape.rank != out_t.shape.rank{
            panic!("in_t and out_t rank must be same, \
                    but receive in_t shape rank {}, out_t shape rank {}.",
                    in_t.shape.rank,
                    out_t.shape.rank)
        }

        let capacity = in_t.shape.capacity();

        if capacity!= out_t.shape.capacity(){
            panic!("in_t and out_t capacity must be same, \
                    but receive in_t shape capacity {}, out_t shape capacity {}.",
                    capacity,
                    out_t.shape.capacity())
        }

        for i in 0..capacity{
            let divider = in_t.longarray[i].exp() + T::one();
            out_t.longarray[i] = divider.recip();
        }

    }
