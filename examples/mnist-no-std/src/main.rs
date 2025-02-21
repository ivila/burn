mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Train a new model and export it to the given path")]
    Train(commands::train::Args),
    #[command(about = "Load model from given path and test it with mnist dataset")]
    Test(commands::test::Args),
    #[command(about = "Load model from given path and test it with given image")]
    Inference(commands::inference::Args),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(arg) => commands::train::run(&arg),
        Commands::Test(arg) => commands::test::run(&arg),
        Commands::Inference(arg) => commands::inference::run(&arg),
    }
}
