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

        let manifest_dir =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
        let status = Command::new("bash")
            .current_dir(&manifest_dir)
            .env("FIDESLIB_SKIP_INSTALL", "1")
            .arg("gpu-setup.sh")
            .status()
            .expect("failed to run gpu-setup.sh");
        if !status.success() {
            panic!("gpu-setup.sh failed with status {status}");
        }

        let fides_root =
            env::var("FIDESLIB_ROOT").unwrap_or_else(|_| "third_party/FIDESlib".to_string());
        let fides_include =
            env::var("FIDESLIB_INCLUDE_DIR").unwrap_or_else(|_| format!("{fides_root}/include"));
        let fides_build_lib_dir = format!("{fides_root}/build");
        let fides_build_lib = PathBuf::from(&fides_build_lib_dir).join("fideslib.a");
        let fides_lib_dir = env::var("FIDESLIB_LIB_DIR").unwrap_or_else(|_| {
            if fides_build_lib.exists() {
                fides_build_lib_dir
            } else {
                "/usr/local/lib".to_string()
            }
        });
        let openfhe_include = env::var("OPENFHE_INCLUDE_DIR")
            .unwrap_or_else(|_| "/usr/local/include/openfhe".to_string());
        let openfhe_core_include = format!("{openfhe_include}/core");
        let openfhe_pke_include = format!("{openfhe_include}/pke");
        let openfhe_binfhe_include = format!("{openfhe_include}/binfhe");

        if !PathBuf::from(&fides_include).exists() {
            println!(
                "cargo::warning=FIDESlib include dir not found at {fides_include} (set FIDESLIB_INCLUDE_DIR)"
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
            .file("cuda/src/Runtime.cu")
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

        println!("cargo::rustc-link-search=native={fides_lib_dir}");
        let fideslib_prefixed = PathBuf::from(&fides_lib_dir).join("libfideslib.a");
        let fideslib_unprefixed = PathBuf::from(&fides_lib_dir).join("fideslib.a");
        if fideslib_prefixed.exists() {
            println!("cargo::rustc-link-lib=static=fideslib");
        } else if fideslib_unprefixed.exists() {
            println!("cargo::rustc-link-arg={}", fideslib_unprefixed.display());
        } else {
            println!(
                "cargo::warning=FIDESlib library not found in {fides_lib_dir} (expected libfideslib.a or fideslib.a)"
            );
            println!("cargo::rustc-link-lib=fideslib");
        }

        println!("cargo::rustc-link-search=native={cuda_lib_dir}");
        println!("cargo::rustc-link-lib=cudart");
        println!("cargo::rustc-link-lib=cudadevrt");
    }
}
