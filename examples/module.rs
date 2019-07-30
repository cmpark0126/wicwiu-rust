use wicwiu::nn::*;
use wicwiu::optimizer::*;
use wicwiu::modules::*;

fn main() {
    let mut x = Tensorholder::<f32>::new(vec![4]);
    let mut optim: Optimizer<f32> = Optimizer{parameter_list: vec![]};
    let mut nn = NeuralNetwork::<f32>::new(optim);

    let x_ref = nn.push(Box::new(x));
    let linear = nn.push(Box::new(Linear::<f32>::new(&x_ref, 4, 3)));
    let act = nn.push(Box::new(Sigmoid::<f32>::new(&linear)));
    let mse = nn.push(Box::new(MSE::<f32>::new(&act)));

    println!("{:?}, \n{}", mse.borrow().result(), mse.borrow().is_tensorholder());

    nn.forward();
    nn.backward();
}
