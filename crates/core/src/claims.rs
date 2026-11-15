//! Claims about proof images, generated from a protocol specification.
//!
//! A protocol specification is a Lean package that defines a primitive's correctness and security
//! with the algorithms left as parameters, and a manifest ([`SpecManifest`]) naming them. Given
//! the proof images of one implementation and the claimed parameters, failure probability,
//! security notions, and assumptions, [`generate`] writes two Lean modules:
//!
//! - `Images`: one `MathcalA.Bytes` definition per algorithm, the image the core flattened;
//! - `Progs`: one `MathcalA.Prog` per algorithm over its image, with the instructions of its
//!   longest decodable prefix (glue, copy routine, and leaf code), for proofs about the run;
//! - `Claim`: `impl` (the images with their step bounds and memory), `params`, `CorrectnessClaim`,
//!   and one `SecurityClaim_<notion>` per notion, whose hypotheses are `MathcalA.Hard` of each
//!   assumption.
//!
//! The output is deterministic. Apart from the image bytes, `Claim` is a few lines that a reader
//! checks directly; the generator itself is not trusted.

use crate::flatten::ProofImage;
use std::fmt::Write;

#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    #[error("manifest line {0}: {1}")]
    Manifest(usize, String),
    #[error("the manifest lacks `{0}`")]
    MissingKey(&'static str),
    #[error(
        "the implementation gives {given:?}, but the specification's algorithms are {expected:?}"
    )]
    Algorithms { expected: Vec<String>, given: Vec<String> },
    #[error("notion `{0}` is not in the specification")]
    UnknownNotion(String),
}

/// The manifest of a specification package (`spec.toml`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecManifest {
    /// The module that defines the specification.
    pub module: String,
    /// The structure of one implementation: one `MxxCore.Algorithm` field per algorithm.
    pub impl_type: String,
    /// The algorithms, in field order.
    pub algorithms: Vec<String>,
    pub params: String,
    /// `Correct (I : Impl) (p : Params) (δ : ℝ) : Prop`.
    pub correct: String,
    /// Each a function of `Impl`, `Params`, `MathcalA.Bounds`, and `ℝ` to `MathcalA.Target`.
    pub notions: Vec<String>,
}

impl SpecManifest {
    /// Parses the flat subset of TOML a manifest uses: `key = "string"` and
    /// `key = ["string", ...]` lines, and `#` comments.
    pub fn parse(text: &str) -> Result<Self, ClaimError> {
        let mut strings = std::collections::BTreeMap::new();
        let mut lists = std::collections::BTreeMap::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let fail = |message: &str| ClaimError::Manifest(number + 1, message.to_owned());
            let (key, value) =
                line.split_once('=').ok_or_else(|| fail("expected `key = value`"))?;
            let (key, value) = (key.trim().to_owned(), value.trim());
            let unquote = |s: &str| {
                s.trim()
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .map(str::to_owned)
                    .ok_or_else(|| fail("expected a string"))
            };
            if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                let items = inner
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(unquote)
                    .collect::<Result<Vec<_>, _>>()?;
                lists.insert(key, items);
            } else {
                strings.insert(key, unquote(value)?);
            }
        }
        let string =
            |key: &'static str| strings.get(key).cloned().ok_or(ClaimError::MissingKey(key));
        let list = |key: &'static str| lists.get(key).cloned().ok_or(ClaimError::MissingKey(key));
        Ok(Self {
            module: string("module")?,
            impl_type: string("impl")?,
            algorithms: list("algorithms")?,
            params: string("params")?,
            correct: string("correct")?,
            notions: list("notions")?,
        })
    }
}

/// One algorithm's proof image.
pub struct AlgorithmImage<'a> {
    pub name: String,
    pub image: &'a ProofImage,
}

/// An assumption a security claim relies on: its module and a Lean term of type
/// `MathcalA.Target` built from a library assumption.
#[derive(Clone, Debug)]
pub struct AssumptionUse {
    pub module: String,
    pub target: String,
}

/// One security claim: a notion of the specification, the adversary's resources and the claimed
/// advantage as Lean terms, and the assumptions it holds under, in order.
#[derive(Clone, Debug)]
pub struct SecurityRequest {
    pub notion: String,
    pub bounds: String,
    pub advantage: String,
    pub assumptions: Vec<AssumptionUse>,
}

/// What a scheme claims: the parameters and failure probability as Lean terms, and its security.
#[derive(Clone, Debug)]
pub struct ClaimRequest {
    pub params: String,
    pub failure_probability: String,
    pub security: Vec<SecurityRequest>,
}

/// The `Images`, `Progs`, and `Claim` modules, as `(module name, source)`, in namespace
/// `namespace`.
pub fn generate(
    namespace: &str,
    manifest: &SpecManifest,
    algorithms: &[AlgorithmImage<'_>],
    request: &ClaimRequest,
) -> Result<Vec<(String, String)>, ClaimError> {
    let given: Vec<String> = algorithms.iter().map(|a| a.name.clone()).collect();
    if given != manifest.algorithms {
        return Err(ClaimError::Algorithms { expected: manifest.algorithms.clone(), given });
    }
    for security in &request.security {
        if !manifest.notions.contains(&security.notion) {
            return Err(ClaimError::UnknownNotion(security.notion.clone()));
        }
    }
    let mut images = String::new();
    writeln!(images, "-- Generated by mxx-core. Do not edit.").unwrap();
    writeln!(images, "import MathcalA.Trusted.Bytes\n").unwrap();
    writeln!(images, "namespace {namespace}.Images\n").unwrap();
    for algorithm in algorithms {
        writeln!(images, "/-- The proof image of `{}`. -/", algorithm.name).unwrap();
        writeln!(images, "def {} : MathcalA.Bytes := [", algorithm.name).unwrap();
        let lines: Vec<String> = algorithm
            .image
            .bytes
            .chunks(16)
            .map(|chunk| {
                format!(
                    "  {}",
                    chunk.iter().map(|b| format!("0x{b:02x}")).collect::<Vec<_>>().join(", ")
                )
            })
            .collect();
        writeln!(images, "{}]\n", lines.join(",\n")).unwrap();
    }
    writeln!(images, "end {namespace}.Images").unwrap();

    let mut progs = String::new();
    writeln!(progs, "-- Generated by mxx-core. Do not edit.").unwrap();
    writeln!(progs, "import MathcalA.Lib.Prog").unwrap();
    writeln!(progs, "import {namespace}.Images\n").unwrap();
    writeln!(progs, "namespace {namespace}.Progs\n").unwrap();
    writeln!(progs, "open MathcalA RiscvZkvm.Rv64\n").unwrap();
    for algorithm in algorithms {
        let instrs = crate::lean::code_prefix(&algorithm.image.bytes);
        writeln!(
            progs,
            "{}",
            crate::lean::prog_definition(
                &algorithm.name,
                &format!("The decodable code of `{}` in the memory it runs with.", algorithm.name),
                &format!("Images.{}", algorithm.name),
                algorithm.image.mem_size,
                &instrs
            )
        )
        .unwrap();
    }
    writeln!(progs, "end {namespace}.Progs").unwrap();

    let mut claim = String::new();
    writeln!(claim, "-- Generated by mxx-core. Do not edit.").unwrap();
    writeln!(claim, "import {}", manifest.module).unwrap();
    let mut modules: Vec<&str> = request
        .security
        .iter()
        .flat_map(|s| s.assumptions.iter().map(|a| a.module.as_str()))
        .collect();
    modules.sort_unstable();
    modules.dedup();
    for module in modules {
        writeln!(claim, "import {module}").unwrap();
    }
    writeln!(claim, "import {namespace}.Images\n").unwrap();
    writeln!(claim, "namespace {namespace}\n").unwrap();
    writeln!(claim, "open MathcalA MxxCore\n").unwrap();
    let fields: Vec<String> = algorithms
        .iter()
        .map(|a| {
            format!(
                "{} := ⟨Images.{}, ⟨{}, {}, {}⟩⟩",
                a.name,
                a.name,
                a.image.step_bound,
                a.image.mem_size,
                a.image.bytes.len()
            )
        })
        .collect();
    writeln!(
        claim,
        "/-- The implementation: each algorithm's image, step bound, memory, and size. -/"
    )
    .unwrap();
    writeln!(claim, "def impl : {} :=\n  {{ {} }}\n", manifest.impl_type, fields.join(",\n    "))
        .unwrap();
    writeln!(claim, "def params : {} := {}\n", manifest.params, request.params).unwrap();
    writeln!(
        claim,
        "def CorrectnessClaim : Prop :=\n  {} impl params ({})\n",
        manifest.correct, request.failure_probability
    )
    .unwrap();
    for security in &request.security {
        let short = security.notion.rsplit('.').next().unwrap_or(&security.notion);
        let hypotheses: String = security
            .assumptions
            .iter()
            .map(|a| format!("  MathcalA.Hard ({}) →\n", a.target))
            .collect();
        writeln!(
            claim,
            "def SecurityClaim_{short} : Prop :=\n{hypotheses}  MathcalA.Hard ({} impl params ({}) ({}))\n",
            security.notion, security.bounds, security.advantage
        )
        .unwrap();
    }
    writeln!(claim, "end {namespace}").unwrap();
    Ok(vec![
        (format!("{namespace}.Images"), images),
        (format!("{namespace}.Progs"), progs),
        (namespace.to_owned(), claim),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::Bindings,
        flatten::{FlattenOptions, flatten},
        leaf::Builtin,
        program::{DefinitionBuilder, ProgramBuilder},
        types::CoreType,
        validate::validate,
    };
    use std::sync::Arc;

    #[test]
    fn test_generated_claim_of_the_identity_example() {
        let manifest_path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/lean/MxxCore/Examples/Identity.toml");
        let manifest =
            SpecManifest::parse(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.algorithms, vec!["run".to_owned()]);
        // The identity on 8 bytes, as a window of the whole input.
        let mut root = DefinitionBuilder::new("root", vec![CoreType::bytes(8)]);
        let input = root.param(0);
        let output = root.builtin(
            Builtin::Window {
                offset: 0.into(),
                runs: 1.into(),
                length: 8.into(),
                stride: 0.into(),
            },
            vec![input],
        );
        let program = ProgramBuilder::new("identity").finish(
            vec!["x".into()],
            root.finish(vec![output]),
            vec!["x".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: "8".into(),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let modules = generate(
            "Generated.Identity",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap();
        let claim = &modules[2].1;
        assert!(claim.contains(
            "def CorrectnessClaim : Prop :=\n  MxxCore.Examples.Identity.Correct impl params (0)"
        ));
        // Written for the Lean check of `scripts/check_core_lean.sh`.
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        std::fs::create_dir_all(&directory).unwrap();
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
        // The run, composed from the glue alone: no leaves.
        let (module, source) = crate::compose::run_module(
            "Generated.Identity",
            "run",
            &image,
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
        let path = directory.join(format!("{file}.lean"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
        let second = generate(
            "Generated.Identity",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap();
        assert_eq!(second, modules, "generation is deterministic");
        let wrong = generate(
            "G",
            &manifest,
            &[AlgorithmImage { name: "other".into(), image: &image }],
            &request,
        );
        assert!(matches!(wrong, Err(ClaimError::Algorithms { .. })));
    }
}
