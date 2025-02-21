use no_std_mnist::{util, NoStdModel, DEVICE, MNIST_IMAGE_SIZE};

#[derive(clap::Args, Debug)]
pub struct Args {
    #[arg(short, long, default_value = "model.bin")]
    model_path: String,
    #[arg(short, long)]
    image_path: String,
}

pub fn run(args: &Args) {
    let import_path = std::path::absolute(&args.model_path).unwrap();
    println!("Import record from \"{}\"", import_path.display());
    let record = std::fs::read(&import_path).unwrap();
    let model: NoStdModel = util::import(&DEVICE, record).unwrap();

    let image_path = std::path::absolute(&args.image_path).unwrap();
    println!("Load image from \"{}\"", image_path.display());
    let image = std::fs::read(&image_path).unwrap();
    assert!(image.len() == MNIST_IMAGE_SIZE);

    let tensor = util::image_to_tensor(&DEVICE, &super::load_image(&image));
    let output = model.forward(tensor);
    println!("Inference number is: {}", util::decode_infer_result(output));
}
