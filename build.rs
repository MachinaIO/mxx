use std::{env, path::PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=src/main.rs");

    // linking openFHE
    println!("cargo::rustc-link-arg=-L/usr/local/lib");
    println!("cargo::rustc-link-arg=-lOPENFHEpke");
    println!("cargo::rustc-link-arg=-lOPENFHEbinfhe");
    println!("cargo::rustc-link-arg=-lOPENFHEcore");

    // linking OpenMP
    println!("cargo::rustc-link-arg=-fopenmp");

    // necessary to avoid LD_LIBRARY_PATH
    println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/local/lib");

    if env::var("CARGO_FEATURE_GPU").is_ok() {
        println!("cargo::rerun-if-changed=cuda/src/Runtime.cu");
        println!("cargo::rerun-if-changed=cuda/src/ChaCha.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/Matrix.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixUtils.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixNTT.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixData.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixArith.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixDecompose.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixSampling.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixTrapdoor.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixSerde.cu");
        println!("cargo::rerun-if-changed=cuda/include/Runtime.cuh");
        println!("cargo::rerun-if-changed=cuda/include/ChaCha.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/Matrix.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixUtils.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixData.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixArith.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixDecompose.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixNTT.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixSampling.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixTrapdoor.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/MatrixSerde.cuh");

        let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "89".to_string());
        let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let cuda_lib_dir =
            env::var("CUDA_LIB_DIR").unwrap_or_else(|_| format!("{cuda_home}/lib64"));
        if env::var("NVCC").is_err() {
            let nvcc_path = format!("{cuda_home}/bin/nvcc");
            if PathBuf::from(&nvcc_path).exists() {
                unsafe {
                    env::set_var("NVCC", nvcc_path);
                }
            }
        }

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .file("cuda/src/Runtime.cu")
            .file("cuda/src/matrix/Matrix.cu")
            .include("cuda/include")
            .flag("-std=c++17")
            .flag("-lineinfo")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag(&format!("-arch=sm_{cuda_arch}"));
        build.compile("gpupoly");

        println!("cargo::rustc-link-search=native={cuda_lib_dir}");
        println!("cargo::rustc-link-lib=cudart");
        println!("cargo::rustc-link-lib=cudadevrt");
    }
}
