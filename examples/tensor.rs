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

    add(&t, &t_clone, &out);

    println!("{:?}", out.borrow());
}
