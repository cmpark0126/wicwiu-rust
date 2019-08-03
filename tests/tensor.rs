use wicwiu::tensor::Tensor;

#[test]
fn tensor_define() {
    let t : Tensor<f32> = Tensor::zeros(vec![], false);
    assert_eq!(t.shape.rank, 0);
    assert_eq!(t.shape.dim, vec![]);
    assert_eq!(t.longarray, vec![0.,]);

    let t : Tensor<f32> = Tensor::zeros(vec![1, 2, 3], false);
    assert_eq!(t.shape.rank, 3);
    assert_eq!(t.shape.dim, vec![1, 2, 3]);
    assert_eq!(t.longarray, vec![0., 0., 0., 0., 0., 0.,]);
}

#[test]
fn tensor_clone() {
    let t : Tensor<f32> = Tensor::zeros(vec![], false);
    let t_clone = t.clone();

    drop(t);

    assert_eq!(t_clone.shape.rank, 0);
    assert_eq!(t_clone.shape.dim, vec![]);
    assert_eq!(t_clone.longarray, vec![0.,]);

    let t : Tensor<f32> = Tensor::zeros(vec![1], false);
    let t_clone = t.clone();

    drop(t);

    assert_eq!(t_clone.shape.rank, 1);
    assert_eq!(t_clone.shape.dim, vec![1]);
    assert_eq!(t_clone.longarray, vec![0.,]);
}
