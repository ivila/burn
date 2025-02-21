pub mod inference;
pub mod test;
pub mod train;

use std::convert::TryInto;

use no_std_mnist::MnistImage;

fn new_mnist_data() -> mnist::Mnist {
    // Keep the same URL with burn do.
    const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    mnist::MnistBuilder::new()
        .base_url(BASE_URL)
        .download_and_extract()
        .finalize()
}

fn load_image(data: &[u8]) -> MnistImage {
    data.try_into().unwrap()
}
