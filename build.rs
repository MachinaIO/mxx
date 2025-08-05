fn main() {
    println!("cargo::rerun-if-changed=src/lib.rs");

    // Linking OpenFHE libraries.
    println!("cargo::rustc-link-arg=-L/usr/local/lib");
    println!("cargo::rustc-link-arg=-lOPENFHEpke");
    println!("cargo::rustc-link-arg=-lOPENFHEbinfhe");
    println!("cargo::rustc-link-arg=-lOPENFHEcore");

    // Linking OpenMP for macOS ARM64.
    // Use the actual path where libomp is installed via Homebrew.
    println!("cargo::rustc-link-arg=-L/opt/homebrew/Cellar/libomp/19.1.7/lib");
    println!("cargo::rustc-link-arg=-lomp");

    // Add rpath for runtime library loading on macOS.
    // This ensures the dynamic libraries can be found at runtime.
    println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    println!("cargo::rustc-link-arg=-Wl,-rpath,/opt/homebrew/Cellar/libomp/19.1.7/lib");

    // Standard library paths for C++ standard library on macOS.
    println!("cargo::rustc-link-arg=-lc++");
}
