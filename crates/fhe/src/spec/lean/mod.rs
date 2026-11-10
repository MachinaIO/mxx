//! Lean statements of a public-key homomorphic encryption scheme and a circuit.
//!
//! [`export`] builds the graphs of a [`Fhe`] scheme and one circuit from the scheme's
//! algorithms, and writes one module per graph and per simulator graph of the given
//! [`Reduction`]s, each defining the graph's relation and its computation `generatedRoot.comp`
//! ([`mxx_ir_core::lean::graph_modules`]), and three modules on top of them:
//!
//! * `Scheme.lean`, the statements: it assembles the computations into a `MxxFhe.Scheme` and the
//!   circuit's `MxxFhe.Evaluation` and states the claims of a [`Statement`],
//!   `GeneratedScheme.CorrectnessClaim` (`MxxFhe.WorstCaseCorrect` or `MxxFhe.AverageCaseCorrect`
//!   for the circuit) and `GeneratedScheme.BoundedSecurityClaim` (`MxxFhe.SecureWithin` relative to
//!   its assumptions and cost bound);
//! * `Reductions.lean`, for proofs: each reduction's simulator as a computation and
//!   `GeneratedScheme.SimulatedSecurity`, IND-CPA security with those simulators (`MxxFhe.Secure`);
//! * `Simulators.lean`, for proofs: each simulator as a pair of IR programs.
//!
//! No statement imports the last two, so the simulators are evidence, not part of a claim.
//!
//! The modules are `Keygen`, `Encrypt`, `Eval` of the circuit on fresh ciphertexts, `Plain`,
//! its `plain_eval`, and `Decrypt_{i}` for each ciphertext type: `0` for the type `Enc`
//! returns and, when it differs, `1` for the type `Eval` returns. The scheme's ciphertext
//! type has one constructor `ct{i}` per type; `Eval` on an argument that is not fresh
//! returns its first argument, since correctness only evaluates fresh ciphertexts.
//!
//! Both definitions are written once in `crates/fhe/lean/spec`; this module only names graphs,
//! roles, and parameter values.

mod reductions;
mod root;
mod scheme;

use super::{
    BindingError, Circuit, Correctness, Fhe, Reduction, SecurityStatement, Statement, roles,
};
use mxx_ir_core::{ParamEnv, lean::graph_modules::export_sampled_graphs};
use reductions::{render_reductions, simulators};
use scheme::render;
use std::{error::Error, fs, path::Path};

/// Writes the graph modules, `Backend.lean`, `Scheme.lean` with the claims of `statement`,
/// and the proof modules `Reductions.lean` and `Simulators.lean` of `reductions`, one per
/// assumption of `statement`, into `directory`.
pub fn export<S: Fhe>(
    scheme: &S,
    circuit: &Circuit,
    statement: &Statement,
    reductions: &[Reduction],
    directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let assumptions = statement.security.as_ref().map_or(&[][..], |s| &s.assumptions[..]);
    for assumption in assumptions {
        assumption.validate()?;
    }
    if reductions.len() != assumptions.len() {
        return Err(BindingError::Reductions {
            assumptions: assumptions.len(),
            reductions: reductions.len(),
        }
        .into());
    }
    let inputs = circuit.input_count();
    let mut graphs = super::graphs(scheme, circuit)?;
    let types = graphs.iter().filter(|(name, _)| name.starts_with("Decrypt_")).count();
    let (_, message_space) = scheme.message_space();
    for (index, reduction) in reductions.iter().enumerate() {
        graphs.push((format!("Sim_{index}_setup"), reduction.setup.clone()));
        graphs.push((format!("Sim_{index}_encrypt"), reduction.challenge.clone()));
    }
    let graphs = graphs.iter().map(|(name, graph)| (name.clone(), graph)).collect::<Vec<_>>();
    let artifacts = export_sampled_graphs(&graphs, &ParamEnv::default(), directory)?;
    let source = render(&message_space, inputs, types, statement, &artifacts)?;
    fs::write(directory.join("Scheme.lean"), source)?;
    if let Some(security) = &statement.security {
        fs::write(directory.join("Reductions.lean"), render_reductions(security, &artifacts)?)?;
        fs::write(directory.join("Simulators.lean"), simulators(reductions)?)?;
    }
    Ok(())
}
