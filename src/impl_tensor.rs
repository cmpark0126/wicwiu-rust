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

pub fn matmul<T: Float>(lhs: &Rc<RefCell<Tensor<T>>>,
    rhs: &Rc<RefCell<Tensor<T>>>,
    out: &Rc<RefCell<Tensor<T>>>){
        let lhs = lhs.borrow();
        let rhs = rhs.borrow();
        let mut out = out.borrow_mut();

        if lhs.shape.rank != 2{
            panic!("lhs rank must be 2, \
                    but receive lhs shape rank {}.",
                    lhs.shape.rank)
        }

        if rhs.shape.rank != 1 || out.shape.rank != 1{
            panic!("lhs and out rank must be 1, \
                    but receive lhs shape rank {}, out shape rank {}.",
                    lhs.shape.rank,
                    out.shape.rank)
        }

        if lhs.shape.dim[1] != rhs.shape.dim[0]{
            panic!("lhs's dim[1] and rhs's dim[0] must be same, \
                    but receive lhs's dim[1] {}, rhs's dim[0] {}.",
                    lhs.shape.dim[1],
                    rhs.shape.dim[0])
        }

        if lhs.shape.dim[0] != out.shape.dim[0]{
            panic!("lhs's dim[0] and out's dim[0] must be same, \
                    but receive lhs's dim[0] {}, rhs's dim[0] {}.",
                    lhs.shape.dim[0],
                    out.shape.dim[0])
        }

        let in_features = rhs.shape.dim[0].clone();
        let out_features = out.shape.dim[0].clone();

        let beta : T = T::zero();

        for out_idx in 0..out_features{
            out.longarray[out_idx] = T::zero();
            for in_idx in 0..in_features{
                out.longarray[out_idx] = out.longarray[out_idx] +
                        lhs.longarray[out_idx * in_features + in_idx].mul_add(rhs.longarray[in_idx], beta);
            }
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