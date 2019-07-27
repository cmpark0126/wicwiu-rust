use wicwiu::tensor::Tensor;

#[test]
fn tensor_define() {
    let t : Tensor<f32> = Tensor::zeros(None);
    assert_eq!(t.shape.rank, 0);
    assert_eq!(t.shape.dim, None);
    assert_eq!(t.longarray, vec![0.,]);

    let t : Tensor<f32> = Tensor::zeros(Some(vec![1, 2, 3]));
    assert_eq!(t.shape.rank, 3);
    assert_eq!(t.shape.dim, Some(vec![1, 2, 3]));
    assert_eq!(t.longarray, vec![0., 0., 0., 0., 0., 0.,]);
}

#[test]
fn tensor_clone() {
    let t : Tensor<f32> = Tensor::zeros(None);
    let t_clone = t.clone();

    drop(t);

    assert_eq!(t_clone.shape.rank, 0);
    assert_eq!(t_clone.shape.dim, None);
    assert_eq!(t_clone.longarray, vec![0.,]);

    let t : Tensor<f32> = Tensor::zeros(Some(vec![1]));
    let t_clone = t.clone();

    drop(t);

    assert_eq!(t_clone.shape.rank, 1);
    assert_eq!(t_clone.shape.dim, Some(vec![1]));
    assert_eq!(t_clone.longarray, vec![0.,]);
}
