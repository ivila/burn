use no_std_mnist::{MlpConfig, MnistConfig, MnistImage, NoStdTrainer, DEVICE, MNIST_IMAGE_SIZE};
use rand::{seq::SliceRandom, Rng};

#[derive(clap::Args, Debug)]
pub struct Args {
    #[arg(short, long, default_value_t = 6)]
    num_epochs: usize,
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 0.0001)]
    learning_rate: f64,
    #[arg(short, long, default_value = "model.bin")]
    output_path: String,
}

fn prepare_data_sets(images: &[u8], labels: &[u8]) -> Vec<(MnistImage, u8)> {
    let mut datasets: Vec<(MnistImage, u8)> = images
        .chunks_exact(MNIST_IMAGE_SIZE)
        .map(|v| super::load_image(v))
        .zip(labels.iter().map(|v| *v))
        .collect();
    datasets.shuffle(&mut rand::rng());
    datasets
}

pub fn run(args: &Args) {
    // Download mnist data
    let data = super::new_mnist_data();
    // Crate a trainer
    let seed: u64 = rand::rng().random();
    let model_config = MnistConfig::new(MlpConfig::new()).with_seed(seed);
    let mut trainer =
        NoStdTrainer::new(DEVICE.clone(), &model_config).with_learning_rate(args.learning_rate);
    // Prepare datasets
    let train_datasets = prepare_data_sets(&data.trn_img, &data.trn_lbl);
    let valid_datasets = prepare_data_sets(&data.val_img, &data.val_lbl);
    // Train
    for epoch in 1..args.num_epochs + 1 {
        for (iteration, data) in train_datasets.chunks(args.batch_size).enumerate() {
            let images: Vec<MnistImage> = data.iter().map(|v| v.0).collect();
            let labels: Vec<u8> = data.iter().map(|v| v.1).collect();
            let output = trainer.train(images.as_slice(), &labels);
            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch, iteration, output.loss, output.accuracy,
            );
        }

        for (iteration, data) in valid_datasets.chunks(args.batch_size).enumerate() {
            let images: Vec<MnistImage> = data.iter().map(|v| v.0).collect();
            let labels: Vec<u8> = data.iter().map(|v| v.1).collect();
            let output = trainer.valid(images.as_slice(), &labels);
            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch, iteration, output.loss, output.accuracy,
            );
        }
    }

    let record = trainer.export().unwrap();
    let output_path = std::path::absolute(&args.output_path).unwrap();
    println!("Export record to \"{}\"", output_path.display());
    std::fs::write(&output_path, &record).unwrap();
}
