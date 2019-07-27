use wicwiu::modules::*;

fn main() {
    let w : &Module<f32> = &Tensorholder::<f32>::new(vec![3, 4]);
    let x : &Module<f32> = &Tensorholder::<f32>::new(vec![4]);
    let linear : &Module<f32> = &Linear::<f32>::new(w, x);
    let act : &Module<f32> = &Sigmoid::<f32>::new(linear);
    let mse : &Module<f32> = &MSE::<f32>::new(act);

    println!("{:?}, \n{}", w.result(), w.is_tensorholder());
    println!("{:?}, \n{}", x.result(), x.is_tensorholder());
    println!("{:?}, \n{}", linear.result(), linear.is_tensorholder());
    println!("{:?}, \n{}", act.result(), act.is_tensorholder());
    println!("{:?}, \n{}", mse.result(), mse.is_tensorholder());
}
