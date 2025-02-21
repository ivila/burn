#![no_std]
extern crate alloc;

use burn::backend::{Autodiff, NdArray};
use burn::backend::ndarray::NdArrayDevice;
pub use burn::record::RecorderError;
pub use burn_no_std_tests::{mlp::MlpConfig, model::{MnistConfig, Model}};

pub mod util;
mod trainer;

pub const MNIST_IMAGE_HEIGHT: usize = 28;
pub const MNIST_IMAGE_WIDTH: usize = 28;
pub const MNIST_IMAGE_SIZE: usize = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;
pub type MnistImage = [u8; MNIST_IMAGE_SIZE];

pub use trainer::{Trainer, Output};
pub type NoStdTrainer = Trainer<Autodiff<NdArray>>;
pub type NoStdModel = Model<NdArray>;
pub const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
