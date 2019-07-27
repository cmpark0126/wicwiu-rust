use wicwiu::modules::{Module, Tensorholder, MSE};

fn main() {
    let th = Tensorholder::<f32>::new(vec![]);
    let mse = MSE::<f32>::new(&th);

    let th_module: &Module<f32> = &th;
    let mse_module: &Module<f32> = &mse;

    println!("{:?}", th_module.result());
    println!("{:?}", mse_module.result());
}
