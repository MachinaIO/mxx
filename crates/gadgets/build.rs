use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn reject_unreviewed_constructs(path: &Path) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("read Lean proof directory") {
            reject_unreviewed_constructs(&entry.expect("read Lean proof entry").path());
        }
        return;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("lean") {
        return;
    }
    let source = fs::read_to_string(path).expect("read Lean proof source");
    let mut comment_depth = 0usize;
    for (line_index, line) in source.lines().enumerate() {
        let code = lean_code_without_comments(line, &mut comment_depth);
        assert!(
            !code
                .split(|character: char| { !character.is_ascii_alphanumeric() && character != '_' })
                .any(|token| token == "axiom"),
            "unreviewed Lean axiom declaration in {}:{}",
            path.display(),
            line_index + 1
        );
        for forbidden in ["sorry", "admit"] {
            assert!(
                !code
                    .split(|character: char| {
                        !character.is_ascii_alphanumeric() && character != '_'
                    })
                    .any(|token| token == forbidden),
                "Lean proof hole `{forbidden}` is not allowed in mxx-gadgets at {}:{}",
                path.display(),
                line_index + 1
            );
        }
        assert!(
            !code.contains("native_decide"),
            "native_decide is not allowed in the mxx-gadgets proof at {}:{}",
            path.display(),
            line_index + 1
        );
    }
}

fn lean_code_without_comments(line: &str, comment_depth: &mut usize) -> String {
    let bytes = line.as_bytes();
    let mut output = String::new();
    let mut index = 0;
    while index < bytes.len() {
        if *comment_depth == 0 &&
            index + 1 < bytes.len() &&
            bytes[index] == b'-' &&
            bytes[index + 1] == b'-'
        {
            break;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'-' {
            *comment_depth += 1;
            index += 2;
        } else if *comment_depth > 0 &&
            index + 1 < bytes.len() &&
            bytes[index] == b'-' &&
            bytes[index + 1] == b'/'
        {
            *comment_depth -= 1;
            index += 2;
        } else {
            if *comment_depth == 0 {
                output.push(bytes[index] as char);
            }
            index += 1;
        }
    }
    output
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let workspace = manifest.parent().and_then(|path| path.parent()).expect("workspace root");
    let lean = workspace.join("lean");
    println!("cargo:rerun-if-changed={}", workspace.join("lean/Mxx").display());
    println!("cargo:rerun-if-changed={}", manifest.join("lean/MxxGadgets").display());
    println!("cargo:rerun-if-changed={}", manifest.join("lean/MxxGadgets.lean").display());
    println!("cargo:rerun-if-env-changed=MXX_REGENERATE_CORRECTNESS");
    if env::var_os("MXX_REGENERATE_CORRECTNESS").is_some() {
        return;
    }
    reject_unreviewed_constructs(&manifest.join("lean"));
    let status = Command::new("lake")
        .args(["build", "MxxGadgets"])
        .current_dir(&lean)
        .status()
        .expect("failed to start the Lean input-injector proof build for mxx-gadgets");
    assert!(status.success(), "Lean input-injector proof build for mxx-gadgets failed");
}
