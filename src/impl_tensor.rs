use crate::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use num::{Num, NumCast, Float, FromPrimitive};
use std::fmt::Debug;

pub fn add<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    lhs: &Rc<RefCell<Tensor<T>>>,
    alpha: &T,
    rhs: &Rc<RefCell<Tensor<T>>>,
    beta: &T,
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

        for i in 0..capacity{
            out.longarray[i] = (*alpha * lhs.longarray[i]) +
                                (*beta * rhs.longarray[i]);
        }

}

pub fn add_<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    lhs: &Rc<RefCell<Tensor<T>>>,
    alpha: &T,
    rhs: &Rc<RefCell<Tensor<T>>>,
    beta: &T) {
        let mut lhs = lhs.borrow_mut();
        let rhs = rhs.borrow();

        if lhs.shape.rank != rhs.shape.rank{
            panic!("lhs and rhs rank must be same, \
                    but receive lhs shape rank {}, rhs shape rank {}.",
                    lhs.shape.rank,
                    rhs.shape.rank)
        }

        let capacity = lhs.shape.capacity();

        if capacity != rhs.shape.capacity(){
            panic!("lhs and rhs capacity must be same, \
                    but receive lhs shape capacity {}, rhs shape capacity {}.",
                    capacity,
                    rhs.shape.capacity())
        }

        for i in 0..capacity{
            lhs.longarray[i] = (*alpha * lhs.longarray[i]) +
                                (*beta * rhs.longarray[i]);
        }
}

pub fn mul_with_constant<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    lhs: &Rc<RefCell<Tensor<T>>>,
    alpha: &T,
    out: &Rc<RefCell<Tensor<T>>>){
        let lhs = lhs.borrow();
        let mut out = out.borrow_mut();

        if lhs.shape.rank != out.shape.rank{
            panic!("lhs and out rank must be same, \
                    but receive lhs shape rank {}, out shape rank {}.",
                    lhs.shape.rank,
                    out.shape.rank)
        }

        let capacity = lhs.shape.capacity();

        if capacity!= out.shape.capacity(){
            panic!("lhs and out capacity must be same, \
                    but receive lhs shape capacity {}, out shape capacity {}.",
                    capacity,
                    out.shape.capacity())
        }

        for i in 0..capacity{
            out.longarray[i] = *alpha * lhs.longarray[i];
        }
}

pub fn mul_with_constant_remain_out<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    lhs: &Rc<RefCell<Tensor<T>>>,
    alpha: &T,
    out: &Rc<RefCell<Tensor<T>>>){
        let lhs = lhs.borrow();
        let mut out = out.borrow_mut();

        if lhs.shape.rank != out.shape.rank{
            panic!("lhs and out rank must be same, \
                    but receive lhs shape rank {}, out shape rank {}.",
                    lhs.shape.rank,
                    out.shape.rank)
        }

        let capacity = lhs.shape.capacity();

        if capacity!= out.shape.capacity(){
            panic!("lhs and out capacity must be same, \
                    but receive lhs shape capacity {}, out shape capacity {}.",
                    capacity,
                    out.shape.capacity())
        }

        for i in 0..capacity{
            out.longarray[i] = out.longarray[i] + *alpha * lhs.longarray[i];
        }
}

pub fn matmul<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    lhs: &Rc<RefCell<Tensor<T>>>,
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

        for out_idx in 0..out_features{
            out.longarray[out_idx] = T::zero();
            for in_idx in 0..in_features{
                out.longarray[out_idx] = out.longarray[out_idx] +
                                            (lhs.longarray[out_idx * in_features + in_idx] *
                                                rhs.longarray[in_idx]);
            }
        }

}

pub fn matmul_backward_input<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    weight: &Rc<RefCell<Tensor<T>>>,
    delta: &Rc<RefCell<Tensor<T>>>,
    grad: &Rc<RefCell<Tensor<T>>>){
        let weight = weight.borrow();
        let delta = delta.borrow();
        let mut grad = grad.borrow_mut();

        if weight.shape.rank != 2{
            panic!("weight rank must be 2, \
                    but receive weight shape rank {}.",
                    weight.shape.rank)
        }

        if delta.shape.rank != 1 || grad.shape.rank != 1{
            panic!("delta and grad rank must be 1, \
                    but receive delta shape rank {}, grad shape rank {}.",
                    delta.shape.rank,
                    grad.shape.rank)
        }

        if weight.shape.dim[0] != delta.shape.dim[0]{
            panic!("weight's dim[1] and delta's dim[0] must be same, \
                    but receive weight's dim[1] {}, delta's dim[0] {}.",
                    weight.shape.dim[0],
                    delta.shape.dim[0])
        }

        if weight.shape.dim[1] != grad.shape.dim[0]{
            panic!("weight's dim[0] and out's dim[0] must be same, \
                    but receive weight's dim[0] {}, grad's dim[0] {}.",
                    weight.shape.dim[1],
                    grad.shape.dim[0])
        }

        let in_features = grad.shape.dim[0].clone();
        let out_features = delta.shape.dim[0].clone();

        for in_idx in 0..in_features{
            grad.longarray[in_idx] = T::zero();
            for out_idx in 0..out_features{
                grad.longarray[in_idx] = grad.longarray[in_idx] +
                                            (weight.longarray[out_idx * in_features + in_idx] *
                                                delta.longarray[out_idx]);
            }
        }
}

pub fn matmul_backward_weight<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    delta: &Rc<RefCell<Tensor<T>>>,
    input: &Rc<RefCell<Tensor<T>>>,
    grad: &Rc<RefCell<Tensor<T>>>){
        let delta = delta.borrow();
        let input = input.borrow();
        let mut grad = grad.borrow_mut();

        if grad.shape.rank != 2{
            panic!("grad rank must be 2, \
                    but receive grad shape rank {}.",
                    grad.shape.rank)
        }

        if delta.shape.rank != 1 || input.shape.rank != 1{
            panic!("delta and input rank must be 1, \
                    but receive delta shape rank {}, input shape rank {}.",
                    delta.shape.rank,
                    input.shape.rank)
        }

        if grad.shape.dim[1] != input.shape.dim[0]{
            panic!("grad's dim[1] and input's dim[0] must be same, \
                    but receive grad's dim[1] {}, input's dim[0] {}.",
                    grad.shape.dim[1],
                    input.shape.dim[0])
        }

        if grad.shape.dim[0] != delta.shape.dim[0]{
            panic!("grad's dim[0] and delta's dim[0] must be same, \
                    but receive grad's dim[0] {}, delta's dim[0] {}.",
                    grad.shape.dim[0],
                    delta.shape.dim[0])
        }

        let in_features = input.shape.dim[0].clone();
        let out_features = delta.shape.dim[0].clone();

        for out_idx in 0..out_features{
            for in_idx in 0..in_features{
                grad.longarray[out_idx * in_features + in_idx] =
                    delta.longarray[out_idx] * input.longarray[in_idx];
            }
        }
}

pub fn sigmoid<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    in_t: &Rc<RefCell<Tensor<T>>>,
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

pub fn sigmoid_backward<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    out_t: &Rc<RefCell<Tensor<T>>>,
    delta_t: &Rc<RefCell<Tensor<T>>>,
    grad_t: &Rc<RefCell<Tensor<T>>>){
        let out_t = out_t.borrow();
        let delta_t = delta_t.borrow();
        let mut grad_t = grad_t.borrow_mut();

        if out_t.shape.rank != delta_t.shape.rank{
            panic!("out_t and delta_t rank must be same, \
                    but receive out_t shape rank {}, delta_t shape rank {}.",
                    out_t.shape.rank,
                    delta_t.shape.rank)
        }

        if out_t.shape.rank != grad_t.shape.rank{
            panic!("out_t and grad_t rank must be same, \
                    but receive out_t shape rank {}, grad_t shape rank {}.",
                    out_t.shape.rank,
                    grad_t.shape.rank)
        }

        let capacity = out_t.shape.capacity();

        if capacity!= delta_t.shape.capacity(){
            panic!("out_t and delta_t capacity must be same, \
                    but receive out_t shape capacity {}, delta_t shape capacity {}.",
                    capacity,
                    delta_t.shape.capacity())
        }

        if capacity!= grad_t.shape.capacity(){
            panic!("out_t and grad_t capacity must be same, \
                    but receive out_t shape capacity {}, grad_t shape capacity {}.",
                    capacity,
                    grad_t.shape.capacity())
        }

        for i in 0..capacity{
            grad_t.longarray[i] = out_t.longarray[i] *
                                    (T::one() - out_t.longarray[i]) *
                                    delta_t.longarray[i];
        }
}

pub fn square<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    in_t: &Rc<RefCell<Tensor<T>>>,
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
            out_t.longarray[i] = in_t.longarray[i] * in_t.longarray[i];
        }

}

pub fn sum<T: Num + NumCast + Float + Clone + FromPrimitive + Debug>(
    in_t: &Rc<RefCell<Tensor<T>>>,
    out_t: &Rc<RefCell<Tensor<T>>>){
        let in_t = in_t.borrow();
        let mut out_t = out_t.borrow_mut();

        if out_t.shape.rank > 0{
            panic!("out_t rank must be 0, \
                    but receive out_t shape rank {}.",
                    out_t.shape.rank)
        }

        let capacity = in_t.shape.capacity();

        out_t.longarray[0] = T::zero();
        for i in 0..capacity{
            out_t.longarray[0] = out_t.longarray[0] + in_t.longarray[i];
        }
}
