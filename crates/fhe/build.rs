//! Builds the native subgraph kernels of `mxx-fhe` when the `gpu` feature is
//! enabled, against the subgraph-kernel header `mxx-backends` publishes.

use std::{env, path::PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }
    println!("cargo::rerun-if-changed=cuda");
    println!("cargo::rerun-if-env-changed=CUDA_ARCH");
    let include = env::var("DEP_MXX_BACKENDS_CUDA_INCLUDE")
        .expect("mxx-backends publishes its CUDA include directory with the gpu feature");
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "89".to_string());
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    if env::var("NVCC").is_err() {
        let nvcc = PathBuf::from(format!("{cuda_home}/bin/nvcc"));
        if nvcc.exists() {
            // SAFETY: the build script is single-threaded here.
            unsafe { env::set_var("NVCC", nvcc) };
        }
    }
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("cuda/tfhe_blind_rotation.cu")
        .include(include)
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .flag(format!("-arch=sm_{cuda_arch}"));
    if env::var("DEBUG").is_ok_and(|debug| debug != "true") {
        build.flag("-lineinfo");
    }
    build.compile("mxx_fhe_kernels");
}
