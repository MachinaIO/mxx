use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

fn native_kernel_build_revision(gpu_dir: &Path, identity: &[String]) -> String {
    fn collect(path: &Path, paths: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(path).expect("read GPU source directory") {
            let path = entry.expect("GPU source entry").path();
            if path.is_dir() {
                collect(&path, paths);
            } else {
                paths.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    collect(gpu_dir, &mut paths);
    paths.sort();
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    fs::read("build.rs").expect("read build script").hash(&mut hasher);
    for path in paths {
        path.strip_prefix(gpu_dir).expect("GPU source prefix").hash(&mut hasher);
        fs::read(&path).expect("read GPU source").hash(&mut hasher);
    }
    format!("gpu-native-build-{:016x}", hasher.finish())
}

fn compiler_version(compiler: &str) -> String {
    let output = std::process::Command::new(compiler)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("GPU compiler {compiler} is unavailable: {error}"));
    assert!(output.status.success(), "GPU compiler {compiler} --version failed");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=native");
    println!("cargo::rerun-if-changed=src/poly/dcrt/native.rs");
    cxx_build::bridge("src/poly/dcrt/native.rs")
        .file("native/ExactBasis.cc")
        .include("native")
        .include("/usr/local/include")
        .include("/usr/local/include/openfhe")
        .include("/usr/local/include/openfhe/core")
        .include("/usr/local/include/openfhe/pke")
        .include("/usr/local/include/openfhe/binfhe")
        .include("/usr/local/include/openfhe/third-party/include")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-fopenmp")
        .compile("mxx_exact_basis");

    // Native library metadata propagates through mxx-backends to its users.
    println!("cargo::rustc-link-search=native=/usr/local/lib");
    println!("cargo::rustc-link-lib=dylib=OPENFHEpke");
    println!("cargo::rustc-link-lib=dylib=OPENFHEbinfhe");
    println!("cargo::rustc-link-lib=dylib=OPENFHEcore");
    println!("cargo::rustc-link-lib=dylib=gomp");

    println!("cargo::rustc-check-cfg=cfg(mxx_gpu_backend, values(\"cuda\", \"hip\"))");
    for name in [
        "MXX_GPU_BACKEND",
        "CUDA_HOME",
        "CUDA_LIB_DIR",
        "NVCC",
        "CUDA_ARCH",
        "ROCM_PATH",
        "HIPCC",
        "HIP_ARCH",
        "HIP_PLATFORM",
        "HIP_PATH",
        "HIP_CLANG_PATH",
        "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPCC_LINK_FLAGS_APPEND",
        "CXX",
        "CXXFLAGS",
        "CC",
        "CFLAGS",
        "AR",
        "ARFLAGS",
        "HOST",
        "TARGET",
        "DEBUG",
        "OPT_LEVEL",
        "PATH",
    ] {
        println!("cargo::rerun-if-env-changed={name}");
    }
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }
    let backend = env::var("MXX_GPU_BACKEND").unwrap_or_else(|_| "cuda".into());
    assert!(backend == "cuda" || backend == "hip", "MXX_GPU_BACKEND must be cuda or hip");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let gpu_dir = manifest.join("gpu");
    let include = gpu_dir.join("include");
    println!("cargo::rerun-if-changed={}", gpu_dir.display());
    let debug = env::var("DEBUG").is_ok_and(|value| value == "true");
    let (sdk, compiler, arch, lib_dir) = if backend == "cuda" {
        let sdk = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let compiler = env::var("NVCC").unwrap_or_else(|_| format!("{sdk}/bin/nvcc"));
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "89".into());
        assert!(
            !arch.is_empty() && arch.bytes().all(|byte| byte.is_ascii_digit()),
            "CUDA_ARCH must be an SM number, e.g. 89"
        );
        let lib_dir = env::var("CUDA_LIB_DIR").unwrap_or_else(|_| format!("{sdk}/lib64"));
        (sdk, compiler, arch, lib_dir)
    } else {
        assert!(
            env::var("HIP_PLATFORM").map_or(true, |value| value == "amd"),
            "HIP_PLATFORM must be amd"
        );
        let sdk = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        let compiler = env::var("HIPCC").unwrap_or_else(|_| format!("{sdk}/bin/hipcc"));
        let arch = env::var("HIP_ARCH").expect("HIP_ARCH must explicitly name the target, e.g. gfx1100; device detection is not required");
        assert!(
            arch.starts_with("gfx") &&
                arch.len() > 3 &&
                arch[3..].bytes().all(|byte| byte.is_ascii_alphanumeric()),
            "HIP_ARCH must be a gfx target, e.g. gfx1100"
        );
        let lib_dir = if Path::new(&format!("{sdk}/lib")).is_dir() {
            format!("{sdk}/lib")
        } else {
            format!("{sdk}/lib64")
        };
        (sdk, compiler, arch, lib_dir)
    };
    let version = compiler_version(&compiler);
    let sdk_version =
        ["version.json", ".info/version", ".info/version-dev", "include/hip/hip_version.h"]
            .iter()
            .filter_map(|name| {
                let path = Path::new(&sdk).join(name);
                println!("cargo::rerun-if-changed={}", path.display());
                fs::read_to_string(path).ok()
            })
            .collect::<Vec<_>>()
            .join(" | ")
            .replace('\n', " | ");
    let mut identity = vec![
        backend.clone(),
        sdk.clone(),
        compiler.clone(),
        version.clone(),
        sdk_version.clone(),
        arch.clone(),
        include.display().to_string(),
        lib_dir.clone(),
        debug.to_string(),
        "c++17,fPIC,no-rdc".into(),
    ];
    for (name, value) in env::vars().filter(|(name, _)| {
        name.contains("FLAGS") ||
            name.starts_with("CC_") ||
            name.starts_with("CXX_") ||
            name.starts_with("AR_") ||
            ["HOST", "TARGET", "OPT_LEVEL", "HIP_PLATFORM", "HIP_PATH", "HIP_CLANG_PATH"]
                .contains(&name.as_str())
    }) {
        println!("cargo::rerun-if-env-changed={name}");
        identity.push(format!("{name}={value}"));
    }
    identity.sort();
    let revision = native_kernel_build_revision(&gpu_dir, &identity);
    println!("cargo::rustc-env=MXX_NATIVE_KERNEL_BUILD_REVISION={revision}");
    println!("cargo::rustc-env=MXX_GPU_BACKEND={backend}");
    let identity_arch = if backend == "cuda" { format!("sm_{arch}") } else { arch.clone() };
    println!("cargo::rustc-env=MXX_GPU_ARCH={identity_arch}");
    println!("cargo::rustc-cfg=mxx_gpu_backend=\"{backend}\"");
    for (name, value) in [
        ("gpu_include", include.display().to_string()),
        ("gpu_backend", backend.clone()),
        ("gpu_arch", arch.clone()),
        ("gpu_compiler", compiler.clone()),
        ("gpu_compiler_version", version.replace('\n', " | ")),
        ("gpu_sdk", sdk.clone()),
        ("gpu_sdk_version", sdk_version),
        ("gpu_native_revision", revision),
    ] {
        println!("cargo::metadata={name}={value}");
    }
    let mut build = cc::Build::new();
    build
        .file("gpu/src/Runtime.cu")
        .file("gpu/src/Primitive.cu")
        .file("gpu/src/Control.cu")
        .file("gpu/src/Real.cu")
        .file("gpu/src/matrix/Matrix.cu")
        .include(&include)
        .flag("-std=c++17");
    if backend == "cuda" {
        // SAFETY: no worker threads have been started by this build script.
        unsafe {
            env::set_var("NVCC", &compiler);
        }
        build
            .cuda(true)
            .define("MXX_GPU_BACKEND_CUDA", "1")
            .flag("-Xcompiler")
            .flag("-fPIC")
            .flag(format!("-arch=sm_{arch}"));
        if !debug {
            build.flag("-lineinfo");
        }
    } else {
        // hipcc generates a self-contained host/device object for each translation unit;
        // no relocatable device code or separate device link is used.
        unsafe {
            env::set_var("HIP_PLATFORM", "amd");
        }
        build
            .cpp(true)
            .compiler(&compiler)
            .define("MXX_GPU_BACKEND_HIP", "1")
            .include(format!("{sdk}/include"))
            .flag("-x")
            .flag("hip")
            .flag("-fPIC")
            .flag("-fno-gpu-rdc")
            .flag(format!("--offload-arch={arch}"));
    }
    build.compile("gpupoly");
    println!("cargo::rustc-link-search=native={lib_dir}");
    if backend == "cuda" {
        println!("cargo::rustc-link-lib=cudart");
        println!("cargo::rustc-link-lib=cudadevrt");
    } else {
        println!("cargo::rustc-link-lib=dylib=amdhip64");
    }
}
