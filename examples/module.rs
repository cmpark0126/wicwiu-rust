use wicwiu::modules::{Module, Tensorholder, MSE, Linear};

fn main() {
    let w : &Module<f32> = &Tensorholder::<f32>::new(vec![3, 4]);
    let x : &Module<f32> = &Tensorholder::<f32>::new(vec![4]);
    let linear : &Module<f32> = &Linear::<f32>::new(w, x);
    let mse : &Module<f32> = &MSE::<f32>::new(linear);

    println!("{:?}", w.result());
    println!("{:?}", x.result());
    println!("{:?}", linear.result());
    println!("{:?}", mse.result());
}
