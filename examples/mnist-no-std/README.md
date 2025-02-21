# MNIST no-std

This example demonstrates how to train and perform inference in a `no-std`
environment.

## Design

This example consists of two crates:

### 1. no-std-mnist

This crate contains everything related to `burn`. It exports `NoStdTrainer` for
training and `NoStdModel` for inference. Since the crate is `no-std`, it can be
used in `no-std` environment (e.g., Trustzone).

### 2. mnist-no-std

This crate serves as the example itself. It does not depend on `burn`, proving
the correctness of `no-std-mnist`.

The example provides 4 subcommands:

1. train -- Trains a new model and exports it to the given path.
2. test -- load a model from the given path and tests it with the MNIST dataset.
3. inference -- Load a model from the given path, tests it with a given image,
                and prints the inference result.
4. help -- Prints help message (provided by `clap` by default).

## Running

1. Train

    ``` shell
    cargo run --release -- train
    ```

    This command downloads the MNIST dataset, trains a new model, and outputs it
    to the given path(default: `model.bin`).

2. Test

    ```shell
    cargo run --release -- test
    ```

    This command loads the model from the given path(default: `model.bin`) and
    tests it with the MNIST dataset.

3. Inference

    ```shell
    cargo run --release -- inference --image-path=images/8.bin
    ```

    This command loads the model the model from given path(default: `model.bin`)
    and tests it with the given image, and prints the inference result.
    For convenience, you can use the sample images in the `images` folder.
