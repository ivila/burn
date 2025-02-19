mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

// no-std
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";
const ARM_NO_ATOMIC_PTR_TARGET: &str = "thumbv6m-none-eabi";
const NO_STD_CRATES: &[&str] = &[
    "burn",
    // cannot run no-std tests on burn-autodiff as no-std tests run with target
    // `thumbv7m-none-eabi`(a 32-bit environment), and burn-autodiff need
    // `core::sync::atomic::AtomicU64`, which doesn't live in a 32-bit
    // environment.
    // Mark it here for notification.
    // "burn-autodiff",
    "burn-core",
    "burn-common",
    "burn-tensor",
    "burn-ndarray",
    "burn-no-std-tests",
];

#[macros::base_commands(
    Bump,
    Check,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Run commands to manage Burn Books.
    Books(commands::books::BooksArgs),
    /// Build Burn in different modes.
    Build(commands::build::BurnBuildCmdArgs),
    /// Test Burn.
    Test(commands::test::BurnTestCmdArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>()?;

    if args.execution_environment == ExecutionEnvironment::NoStd {
        // Install additional targets for no-std execution environments
        rustup_add_target(WASM32_TARGET)?;
        rustup_add_target(ARM_TARGET)?;
        rustup_add_target(ARM_NO_ATOMIC_PTR_TARGET)?;
    }

    match args.command {
        Command::Books(cmd_args) => cmd_args.parse(),
        Command::Build(cmd_args) => {
            commands::build::handle_command(cmd_args, args.execution_environment)
        }
        Command::Doc(cmd_args) => commands::doc::handle_command(cmd_args),
        Command::Test(cmd_args) => {
            commands::test::handle_command(cmd_args, args.execution_environment)
        }
        Command::Validate(cmd_args) => {
            commands::validate::handle_command(&cmd_args, &args.execution_environment)
        }
        _ => dispatch_base_commands(args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
