use wicwiu::nn::*;
use wicwiu::optimizers::*;
use wicwiu::modules::*;
use wicwiu::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

macro_rules! dtype {
    () => (f32);
}

fn create_xor_input(case: usize) -> (Rc<RefCell<Tensor<dtype!()>>>, Rc<RefCell<Tensor<dtype!()>>>){
    let mut input = Tensor::<f32>::zeros(vec![2], true);
    let mut target = Tensor::<f32>::zeros(vec![2], true);

    match case{
        0 => {
            // 0, 0 => 1, 0
            target.longarray[0] = 1.0;
        },
        1 => {
            // 0, 1 => 0, 1
            input.longarray[1] = 1.0;
            target.longarray[1] = 1.0;
        },
        2 => {
            // 1, 0 => 0, 1
            input.longarray[0] = 1.0;
            target.longarray[1] = 1.0;
        },
        3 => {
            // 1, 1 => 1, 0
            input.longarray[0] = 1.0;
            input.longarray[1] = 1.0;
            target.longarray[0] = 1.0;
        },
        i => {
            panic!("{} is out of case!", i);
        }
    }

    (Rc::new(RefCell::new(input)), Rc::new(RefCell::new(target)))
}

fn main() {
    let x = Tensorholder::<dtype!()>::new(vec![2]);
    let t = Tensorholder::<dtype!()>::new(vec![2]);
    let mut nn = NeuralNetwork::<dtype!()>::new();


    let x_ref = nn.push(Box::new(x));
    let t_ref = nn.push(Box::new(t));
    let linear1 = nn.push(Box::new(Linear::<dtype!()>::new(&x_ref, 2, 3)));
    let act1 = nn.push(Box::new(Sigmoid::<dtype!()>::new(&linear1)));
    let linear2 = nn.push(Box::new(Linear::<dtype!()>::new(&act1, 3, 2)));
    let act2 = nn.push(Box::new(Sigmoid::<dtype!()>::new(&linear2)));
    let mse = nn.push(Box::new(MSE::<dtype!()>::new(&act2, &t_ref)));

    let optim: &mut Optimizer<dtype!()> = &mut SGD::new(nn.parameters(), 0.001);
    let cnt : usize = 0;

    // train
    loop{
        let case_num = cnt % 4;
        {
            let (g_i, g_t) = create_xor_input(case_num);

            let mut input_ref = x_ref.borrow_mut();
            input_ref.set_result(g_i);
            // let input_result = input_ref.result();
            // println!("{:?}", input_result);

            let mut target_ref = t_ref.borrow_mut();
            target_ref.set_result(g_t);
            // let target_result = target_ref.result();
            // println!("{:?}", target_result);
        }
        optim.zero_grad();
        nn.forward();
        nn.backward();
        optim.step();

        print!("loss {:?}, \r", mse.borrow().result().borrow().longarray);

        if cnt == 1000 {
            break;
        }
    }
}
