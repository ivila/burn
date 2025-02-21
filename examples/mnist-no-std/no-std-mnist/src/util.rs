use alloc::vec::Vec;

use crate::{MlpConfig, MnistConfig, MnistImage, Model, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH};

use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder, RecorderError};
use burn::tensor::ElementComparison;

// Convert an image into Tensor
// copy from example of `mnist-inference-web`
pub fn image_to_tensor<B: Backend>(device: &B::Device, image: &MnistImage) -> Tensor<B, 3> {
    let tensor = TensorData::from(image.as_slice()).convert::<B::FloatElem>();
    let tensor = Tensor::<B, 1>::from_data(tensor, device);
    let tensor = tensor.reshape([1, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT]);

    // Normalize input: make between [0,1] and make the mean=0 and std=1
    // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
    // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
    ((tensor / 255) - 0.1307) / 0.3081
}

pub fn images_to_tensors<B: Backend>(device: &B::Device, images: &[MnistImage]) -> Tensor<B, 3> {
    let tensors = images.iter().map(|v| image_to_tensor(device, v)).collect();
    Tensor::cat(tensors, 0)
}

pub fn labels_to_tensors<B: Backend>(device: &B::Device, labels: &[u8]) -> Tensor<B, 1, Int> {
    let targets = labels
        .iter()
        .map(|item| Tensor::<B, 1, Int>::from_data([(*item as i64).elem::<B::IntElem>()], device))
        .collect();
    Tensor::cat(targets, 0)
}

pub fn export<B: Backend>(model: Model<B>) -> Result<Vec<u8>, RecorderError> {
    let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
    recorder.record(model.into_record(), ())
}

pub fn import<B: Backend>(device: &B::Device, data: Vec<u8>) -> Result<Model<B>, RecorderError> {
    let mlp_config = MlpConfig::new();
    let mnist_config = MnistConfig::new(mlp_config);
    let model = Model::new(&mnist_config, device);

    let recorder = burn::record::BinBytesRecorder::<FullPrecisionSettings>::new();
    let record = recorder.load(data, device)?;
    Ok(model.load_record(record))
}

pub fn decode_infer_result<B: Backend>(tensor: Tensor<B, 2, Float>) -> u8 {
    let output = burn::tensor::activation::softmax(tensor, 1).into_data();
    assert!(output.num_elements() == 10);
    output
        .iter::<f32>()
        .enumerate()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(index, _)| index)
        .unwrap() as u8
}
