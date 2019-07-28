use wicwiu::modules::*;

fn main() {
    let x = Tensorholder::<f32>::new(vec![4]);
    let linear = Linear::<f32>::new(Box::new(x), 4, 3);
    let act = Sigmoid::<f32>::new(Box::new(linear));
    let mse : &Module<f32> = &MSE::<f32>::new(Box::new(act));

    // println!("{:?}, \n{}", w.result(), w.is_tensorholder());
    println!("{:?}, \n{}", mse.result(), mse.is_tensorholder());
}
