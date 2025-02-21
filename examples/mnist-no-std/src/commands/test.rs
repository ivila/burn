use mnist::Mnist;
use no_std_mnist::{util, NoStdModel, DEVICE, MNIST_IMAGE_SIZE};

#[derive(clap::Args, Debug)]
pub struct Args {
    #[arg(short, long, default_value = "model.bin")]
    model_path: String,
}

pub fn run(args: &Args) {
    let (images, labels) = get_datasets();

    let import_path = std::path::absolute(&args.model_path).unwrap();
    println!("Import record from \"{}\"", import_path.display());
    let record = std::fs::read(&import_path).unwrap();
    let model: NoStdModel = util::import(&DEVICE, record).unwrap();

    let total = labels.len();
    println!("Test on datasets, total: {}", total);
    println!("process: {{match}}/{{checked}}/{{total}}, {{current_accuracy}}");
    let mut success = 0;
    for (index, (image, label)) in images
        .chunks_exact(MNIST_IMAGE_SIZE)
        .zip(labels.iter())
        .enumerate()
    {
        let tensor = util::image_to_tensor(&DEVICE, &super::load_image(image));
        let output = model.forward(tensor);
        if util::decode_infer_result(output) == *label {
            success += 1;
        }
        // Print the process
        let num = index + 1;
        if num % 1000 == 0 || index == total {
            println!(
                "process: {}/{}/{}, {:.3} %",
                success,
                num,
                total,
                (success as f64) / (num as f64) * 100.0
            );
        }
    }

    println!(
        "Result: total {}, accuracy {:.3} %",
        total,
        (success as f64) / (total as f64) * 100.0
    );
}

fn get_datasets() -> (Vec<u8>, Vec<u8>) {
    let Mnist {
        mut trn_img,
        mut trn_lbl,
        mut tst_img,
        mut tst_lbl,
        mut val_img,
        mut val_lbl,
    } = super::new_mnist_data();

    trn_img.append(&mut tst_img);
    trn_img.append(&mut val_img);

    trn_lbl.append(&mut tst_lbl);
    trn_lbl.append(&mut val_lbl);

    (trn_img, trn_lbl)
}
