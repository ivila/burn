use crate::{
    util::{export, images_to_tensors, labels_to_tensors},
    MnistConfig, MnistImage, Model,
};
use alloc::vec::Vec;
use burn::record::RecorderError;
use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLoss,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};

pub struct Trainer<B: AutodiffBackend> {
    model: Model<B>,
    device: B::Device,
    optim: OptimizerAdaptor<Adam, Model<B>, B>,
    lr: f64,
}

pub struct Output {
    pub loss: f32,
    pub accuracy: f32,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(device: B::Device, model_config: &MnistConfig) -> Self {
        let config_optimizer = AdamConfig::new();

        B::seed(model_config.seed);

        Self {
            optim: config_optimizer.init(),
            model: Model::new(model_config, &device),
            device,
            lr: 1e-4,
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    pub fn train(&mut self, images: &[MnistImage], labels: &[u8]) -> Output {
        let images = images_to_tensors(&self.device, &images);
        let targets = labels_to_tensors(&self.device, &labels);
        let model = self.model.clone();

        let output = model.forward(images);
        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());
        let accuracy = accuracy(output, targets);

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);
        // Update the model using the optimizer.
        self.model = self.optim.step(self.lr, model, grads);

        Output {
            loss: loss.into_scalar().to_f32(),
            accuracy,
        }
    }

    pub fn valid(&self, images: &[MnistImage], labels: &[u8]) -> Output {
        // Get the model without autodiff.
        let model_valid = self.model.valid();

        let images = images_to_tensors(&self.device, images);
        let targets = labels_to_tensors(&self.device, labels);

        let output = model_valid.forward(images);
        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), targets.clone());
        let accuracy = accuracy(output, targets);

        Output {
            loss: loss.into_scalar().to_f32(),
            accuracy,
        }
    }

    pub fn export(&self) -> Result<Vec<u8>, RecorderError> {
        export(self.model.clone())
    }
}

/// Create out own accuracy metric calculation.
fn accuracy<B: Backend>(output: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> f32 {
    let predictions = output.argmax(1).squeeze(1);
    let num_predictions: usize = targets.dims().iter().product();
    let num_corrects = predictions.equal(targets).int().sum().into_scalar();

    num_corrects.elem::<f32>() / num_predictions as f32 * 100.0
}
