use wicwiu::nn::*;
use wicwiu::optimizer::*;
use wicwiu::modules::*;

fn main() {
    let mut x = Tensorholder::<f32>::new(vec![4]);
    let mut optim: Optimizer<f32> = Optimizer{parameter_list: vec![]};
    let mut nn = NeuralNetwork::<f32>::new();

    let x_ref = nn.push(Box::new(x));
    let linear1 = nn.push(Box::new(Linear::<f32>::new(&x_ref, 4, 3)));
    let act1 = nn.push(Box::new(Sigmoid::<f32>::new(&linear1)));
    let linear2 = nn.push(Box::new(Linear::<f32>::new(&act1, 3, 2)));
    let act2 = nn.push(Box::new(Sigmoid::<f32>::new(&linear2)));
    let mse = nn.push(Box::new(MSE::<f32>::new(&act2)));

    println!("{:?}, \n{}", mse.borrow().result().borrow(), mse.borrow().is_tensorholder());

    nn.forward();
    nn.backward();

    for p in nn.parameters(){
        println!("{:?}", p.borrow());
    }
}
