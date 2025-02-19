#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # Burn Autodiff
//!
//! This autodiff library is a part of the Burn project. It is a standalone crate
//! that can be used to perform automatic differentiation on tensors. It is
//! designed to be used with the Burn Tensor crate, but it can be used with any
//! tensor library that implements the `Backend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Checkpoint module.
pub mod checkpoint;
/// Gradients module.
pub mod grads;
/// Operation module.
pub mod ops;

pub(crate) mod graph;
// Exported for backend extension
pub use graph::NodeID;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;

pub(crate) mod runtime;

pub use backend::*;

#[cfg(feature = "export_tests")]
mod tests;

/// A facade around all the types we need from the `std`, `core`, and `alloc`
/// crates. This avoids elaborate import wrangling having to happen in every
/// module.
mod libs {
    cfg_block::cfg_block! {
        if #[cfg(feature = "std")] {
            pub use std::collections::{HashMap, HashSet};
            pub use std::sync::Arc;
            pub use std::{vec, vec::Vec};
            pub use std::string::String;
            pub use std::boxed::Box;
            pub use std::format;
            pub use std::borrow::ToOwned;
        } else {
            pub use hashbrown::{HashMap, HashSet};
            pub use alloc::sync::Arc;
            pub use alloc::{vec, vec::Vec};
            pub use alloc::string::String;
            pub use alloc::boxed::Box;
            pub use alloc::format;
            pub use alloc::borrow::ToOwned;
        }
    }
}
