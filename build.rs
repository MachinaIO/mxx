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
        println!("cargo::rerun-if-changed=cuda/GpuPoly.cu");
        println!("cargo::rerun-if-changed=cuda/GpuPoly.h");

        let fides_root = env::var("FIDESLIB_ROOT").unwrap_or_else(|_| "third_party/FIDESlib".to_string());
        let fides_include =
            env::var("FIDESLIB_INCLUDE_DIR").unwrap_or_else(|_| format!("{fides_root}/include"));
        let fides_lib_dir =
            env::var("FIDESLIB_LIB_DIR").unwrap_or_else(|_| "/usr/local/lib".to_string());

        if !PathBuf::from(&fides_include).exists() {
            println!(
                "cargo::warning=FIDESlib include dir not found at {fides_include} (set FIDESLIB_INCLUDE_DIR)"
            );
        }

        let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "70".to_string());
        let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let cuda_lib_dir =
            env::var("CUDA_LIB_DIR").unwrap_or_else(|_| format!("{cuda_home}/lib64"));

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .file("cuda/GpuPoly.cu")
            .include(&fides_include)
            .flag("-std=c++17")
            .flag("-lineinfo")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag(&format!("-arch=sm_{cuda_arch}"));
        build.compile("gpupoly");

        println!("cargo::rustc-link-search=native={fides_lib_dir}");
        println!("cargo::rustc-link-lib=fideslib");

        println!("cargo::rustc-link-search=native={cuda_lib_dir}");
        println!("cargo::rustc-link-lib=cudart");
    }
}
