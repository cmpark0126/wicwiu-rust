use wicwiu::nn::*;
use wicwiu::optimizers::*;
use wicwiu::modules::*;
use wicwiu::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

fn create_xor_input(case: usize, with) -> (){

}

fn main() {
    let x = Tensorholder::<f32>::new(vec![4]);
    let t = Tensorholder::<f32>::new(vec![2]);
    let mut nn = NeuralNetwork::<f32>::new();

    let x_ref = nn.push(Box::new(x));
    let mut module_ref = x_ref.borrow_mut();
    let module_result = module_ref.result();
    println!("{:?}", module_result.borrow());
    let new_x = Tensor::<f32>::ones(vec![4], true);
    module_ref.set_result(Rc::new(RefCell::new(new_x)));
    let module_result = module_ref.result();
    println!("{:?}", module_result.borrow());
    drop(module_ref);
    let t_ref = nn.push(Box::new(t));
    let linear1 = nn.push(Box::new(Linear::<f32>::new(&x_ref, 4, 3)));
    let act1 = nn.push(Box::new(Sigmoid::<f32>::new(&linear1)));
    let linear2 = nn.push(Box::new(Linear::<f32>::new(&act1, 3, 2)));
    let act2 = nn.push(Box::new(Sigmoid::<f32>::new(&linear2)));
    let mse = nn.push(Box::new(MSE::<f32>::new(&act2, &t_ref)));

    let optim: &mut Optimizer<f32> = &mut SGD::new(nn.parameters(), 0.01);

    nn.forward();
    nn.backward();

    println!("{:?}, \n{}", mse.borrow().result().borrow(), mse.borrow().is_tensorholder());

    for p in nn.parameters(){
        println!("{:?}", p.borrow());
    }

    optim.step();
}
