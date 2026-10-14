//! Builds the CUDA-only TFHE subgraph kernel using the exact backend compiler
//! selection published by `mxx-backends`. HIP BGV never compiles or links it.
use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(mxx_gpu_backend, values(\"cuda\", \"hip\"))");
    for name in [
        "GPU_INCLUDE",
        "GPU_BACKEND",
        "GPU_ARCH",
        "GPU_COMPILER",
        "GPU_COMPILER_VERSION",
        "GPU_SDK",
        "GPU_NATIVE_REVISION",
    ] {
        println!("cargo::rerun-if-env-changed=DEP_MXX_BACKENDS_{name}");
    }
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }
    let metadata = |name: &str| {
        env::var(format!("DEP_MXX_BACKENDS_{name}"))
            .unwrap_or_else(|_| panic!("mxx-backends must publish {name} with the gpu feature"))
    };
    let backend = metadata("GPU_BACKEND");
    assert!(backend == "cuda" || backend == "hip", "unknown mxx-backends GPU backend");
    println!("cargo::rustc-cfg=mxx_gpu_backend=\"{backend}\"");
    if backend == "hip" {
        return;
    }
    println!("cargo::rerun-if-changed=cuda");
    let mut hasher = DefaultHasher::new();
    for name in
        ["GPU_NATIVE_REVISION", "GPU_COMPILER", "GPU_COMPILER_VERSION", "GPU_ARCH", "GPU_SDK"]
    {
        metadata(name).hash(&mut hasher);
    }
    fs::read("cuda/tfhe_blind_rotation.cu").expect("read TFHE native source").hash(&mut hasher);
    fs::read("build.rs").expect("read TFHE build script").hash(&mut hasher);
    println!(
        "cargo::rustc-env=MXX_FHE_GPU_NATIVE_REVISION=tfhe-native-build-{:016x}",
        hasher.finish()
    );
    let include = metadata("GPU_INCLUDE");
    // Headers are external to this crate, so track them explicitly too.
    println!("cargo::rerun-if-changed={include}");
    let compiler = metadata("GPU_COMPILER");
    unsafe {
        env::set_var("NVCC", &compiler);
    }
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .define("MXX_GPU_BACKEND_CUDA", "1")
        .file("cuda/tfhe_blind_rotation.cu")
        .include(include)
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .flag(format!("-arch=sm_{}", metadata("GPU_ARCH")));
    if env::var("DEBUG").is_ok_and(|value| value != "true") {
        build.flag("-lineinfo");
    }
    build.compile("mxx_fhe_kernels");
}
