use wicwiu::modules::*;

fn main() {
    let linear = Linear::new(true);
    let sig = Sigmoid::new();
    let mse = MSE::new();

    linear.forward();
    linear.backward();
    sig.forward();
    sig.backward();
    mse.forward();
    mse.backward();
}
