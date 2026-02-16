use std::{env, path::PathBuf, process::Command};

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
        println!("cargo::rerun-if-changed=gpu-setup.sh");
        println!("cargo::rerun-if-changed=cuda/src/ChaCha.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/Matrix.cu");
        println!("cargo::rerun-if-changed=cuda/src/Runtime.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixUtils.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixData.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixArith.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixDecompose.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixSampling.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixTrapdoor.cu");
        println!("cargo::rerun-if-changed=cuda/src/matrix/MatrixSerde.cu");
        println!("cargo::rerun-if-changed=cuda/include/Runtime.cuh");
        println!("cargo::rerun-if-changed=cuda/include/matrix/Matrix.h");
        println!("cargo::rerun-if-changed=cuda/include/ChaCha.cuh");

        let manifest_dir =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
        let skip_setup = env::var("FIDESLIB_SKIP_SETUP").ok().as_deref() == Some("1");
        if skip_setup {
            println!("cargo::warning=Skipping gpu-setup.sh (FIDESLIB_SKIP_SETUP=1)");
        } else {
            let mut setup_cmd = Command::new("bash");
            setup_cmd
                .current_dir(&manifest_dir)
                .env("FIDESLIB_SKIP_INSTALL", "1")
                .arg("gpu-setup.sh");
            if let Ok(skip_submodule_update) = env::var("FIDESLIB_SKIP_SUBMODULE_UPDATE") {
                setup_cmd.env("FIDESLIB_SKIP_SUBMODULE_UPDATE", skip_submodule_update);
            }
            if let Ok(cuda_arch) = env::var("CUDA_ARCH") {
                setup_cmd.env("CUDA_ARCH", cuda_arch);
            }

            let status = setup_cmd.status().expect("failed to run gpu-setup.sh");
            if !status.success() {
                panic!("gpu-setup.sh failed with status {status}");
            }
        }

        let fides_include = manifest_dir.join("third_party/FIDESlib/include");
        let fides_static_lib = manifest_dir.join("third_party/FIDESlib/build/fideslib.a");
        let openfhe_include = env::var("OPENFHE_INCLUDE_DIR")
            .unwrap_or_else(|_| "/usr/local/include/openfhe".to_string());
        let openfhe_core_include = format!("{openfhe_include}/core");
        let openfhe_pke_include = format!("{openfhe_include}/pke");
        let openfhe_binfhe_include = format!("{openfhe_include}/binfhe");

        if !fides_include.exists() {
            println!(
                "cargo::warning=FIDESlib include dir not found at {}",
                fides_include.display()
            );
        }
        if !PathBuf::from(&openfhe_include).exists() {
            println!(
                "cargo::warning=OpenFHE include dir not found at {openfhe_include} (set OPENFHE_INCLUDE_DIR)"
            );
        }
        if !PathBuf::from(&openfhe_core_include).exists() {
            println!(
                "cargo::warning=OpenFHE core include dir not found at {openfhe_core_include} (set OPENFHE_INCLUDE_DIR)"
            );
        }
        if !PathBuf::from(&openfhe_pke_include).exists() {
            println!(
                "cargo::warning=OpenFHE pke include dir not found at {openfhe_pke_include} (set OPENFHE_INCLUDE_DIR)"
            );
        }
        if !PathBuf::from(&openfhe_binfhe_include).exists() {
            println!(
                "cargo::warning=OpenFHE binfhe include dir not found at {openfhe_binfhe_include} (set OPENFHE_INCLUDE_DIR)"
            );
        }

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
            .file("cuda/src/matrix/Matrix.cu")
            .include("cuda/include")
            .include(&fides_include)
            .include(&openfhe_include)
            .include(&openfhe_core_include)
            .include(&openfhe_pke_include)
            .include(&openfhe_binfhe_include)
            .flag("-std=c++17")
            .flag("-lineinfo")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag(&format!("-arch=sm_{cuda_arch}"));
        build.compile("gpupoly");

        if !fides_static_lib.exists() {
            panic!(
                "FIDESlib static library not found at {}. run gpu-setup.sh before building with --features gpu",
                fides_static_lib.display()
            );
        }
        println!("cargo::rustc-link-arg={}", fides_static_lib.display());

        println!("cargo::rustc-link-search=native={cuda_lib_dir}");
        println!("cargo::rustc-link-lib=cudart");
        println!("cargo::rustc-link-lib=cudadevrt");
    }
}
