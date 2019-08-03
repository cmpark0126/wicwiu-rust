use wicwiu::tensor::Tensor;
use wicwiu::impl_tensor::*;
use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    let t : Tensor<f32> = Tensor::ones(vec![2, 2, 2], false);
    let t_clone : Tensor<f32> = t.clone();
    let out : Tensor<f32> = Tensor::zeros(vec![2, 2, 2], false);

    let t = Rc::new(RefCell::new(t));
    let t_clone = Rc::new(RefCell::new(t_clone));
    let out = Rc::new(RefCell::new(out));

    println!("{:?}", out.borrow());
    add(&t, &2.0, &t_clone, &3.0, &out);
    println!("{:?}", out.borrow());

    drop(t);
    drop(t_clone);

    let t = out;
    let out = Rc::new(RefCell::new(t.borrow().clone()));

    println!("{:?}", out.borrow());
    sigmoid(&t, &out);
    println!("{:?}", out.borrow());

    drop(t);
    drop(out);

    let lhs : Tensor<f32> = Tensor::ones(vec![3, 4], false);
    let rhs : Tensor<f32> = Tensor::ones(vec![4], false);
    let out : Tensor<f32> = Tensor::ones(vec![3], false);

    let lhs = Rc::new(RefCell::new(lhs));
    let rhs = Rc::new(RefCell::new(rhs));
    let out = Rc::new(RefCell::new(out));

    println!("{:?}", out.borrow());
    matmul(&lhs, &rhs, &out);
    println!("{:?}", out.borrow());
}
