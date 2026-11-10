//! The proof modules `Reductions.lean` and `Simulators.lean`.

use super::{
    BindingError, Reduction, SecurityStatement, roles,
    root::{Root, is_simulator, names},
};
use mxx_ir_core::{
    Graph, ParamEnv,
    lean::{LeanArtifact, program::program_term},
    security::AssumptionId,
};
use std::{collections::BTreeMap, error::Error, fmt::Write as _};

/// `Reductions.lean`: each reduction's simulator as a computation, the reductions as
/// `MxxFhe.AssumptionUse`s, and IND-CPA security with them, for proofs of the claims of
/// `Scheme.lean`. No statement imports it.
pub(super) fn render_reductions(
    statement: &SecurityStatement,
    artifacts: &BTreeMap<String, LeanArtifact>,
) -> Result<String, Box<dyn std::error::Error>> {
    let root = |module: &str| -> Result<Root<'_>, String> {
        let artifact = artifacts.get(module).ok_or(format!("missing module {module}"))?;
        Ok(Root { module: &artifact.module_name, root: &artifact.root })
    };
    let mut source = String::from("import Scheme\n");
    for module in artifacts.keys().filter(|module| is_simulator(module)) {
        writeln!(source, "import {module}")?;
    }
    source.push_str(
        "\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\n\
         /-! The simulators of the reductions of `GeneratedScheme.BoundedSecurityClaim` as\n\
         computations, and IND-CPA security with them, for its proof. Generated deterministically;\n\
         never edit by hand. -/\n\n\
         namespace GeneratedScheme\n\nopen MxxSecurity MxxFhe\n\n",
    );
    for (_, artifact) in artifacts.iter().filter(|(name, _)| is_simulator(name)) {
        source.push_str(
            &Root { module: &artifact.module_name, root: &artifact.root }.params_definition(),
        );
    }
    source.push('\n');
    for (index, assumption) in statement.assumptions.iter().enumerate() {
        source.push_str(&render_reduction(index, assumption.id(), &root)?);
    }
    let uses = statement
        .assumptions
        .iter()
        .enumerate()
        .map(|(index, assumption)| {
            format!("⟨{}, params_{index}, simulator_{index}⟩", assumption.id().lean_name())
        })
        .collect::<Vec<_>>()
        .join(",\n    ");
    write!(
        source,
        "/-- The reductions. -/\nnoncomputable def uses : List (MxxFhe.AssumptionUse scheme) :=\n  [{uses}]\n\n\
         /-- IND-CPA security with these reductions, without a statistical term. -/\n\
         def SimulatedSecurity : Prop :=\n  MxxFhe.Secure scheme uses 0\n\n\
         end GeneratedScheme\n"
    )?;
    Ok(source)
}

fn render_reduction<'a>(
    index: usize,
    id: AssumptionId,
    root: &impl Fn(&str) -> Result<Root<'a>, String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let setup = root(&format!("Sim_{index}_setup"))?;
    let challenge = root(&format!("Sim_{index}_encrypt"))?;
    let components = id.challenge_components();
    setup.expect_inputs(&names(components))?;
    for required in [roles::PUBLIC_KEY, roles::EVALUATION_KEY] {
        if !setup.root.outputs.contains_key(required) {
            return Err(BindingError::MissingOutput {
                graph: setup.module.into(),
                missing: required.into(),
            }
            .into());
        }
    }
    challenge.expect_outputs(&names(&[roles::CIPHERTEXT]))?;
    let projections: &[&str] = match components.len() {
        2 => &["challenge.1", "challenge.2"],
        _ => &["challenge.1.1", "challenge.1.2", "challenge.2.1", "challenge.2.2"],
    };
    let setup_inputs = components
        .iter()
        .zip(projections)
        .map(|(name, projection)| ((*name).to_owned(), (*projection).to_owned()))
        .collect::<BTreeMap<_, _>>();
    let mut challenge_inputs = BTreeMap::new();
    for name in challenge.root.inputs.keys() {
        let value =
            if name == roles::MESSAGE { "m".to_owned() } else { setup.output(name, "state")? };
        challenge_inputs.insert(name.clone(), value);
    }
    Ok(format!(
        "noncomputable def simulator_{index} : MxxFhe.Simulator scheme (MxxSecurity.AssumptionId.Challenge {} params_{index}) where\n  \
         State := {}\n  \
         setup challenge := do\n    let state ← {} {}\n    pure ({}, {}, state)\n  \
         encrypt state m := do\n    let outputs ← {} {}\n    pure (.ct0 {})\n\n",
        id.lean_name(),
        setup.root.output_type,
        setup.comp(),
        setup.inputs(&setup_inputs)?,
        setup.output(roles::PUBLIC_KEY, "state")?,
        setup.output(roles::EVALUATION_KEY, "state")?,
        challenge.comp(),
        challenge.inputs(&challenge_inputs)?,
        challenge.output(roles::CIPHERTEXT, "outputs")?,
    ))
}

/// `Simulators.lean`: each reduction's simulator as a pair of IR programs, for a proof of
/// `BoundedSecurityClaim`. No statement imports it.
pub(super) fn simulators(reductions: &[Reduction]) -> Result<String, Box<dyn Error>> {
    let mut source = String::from(
        "import MxxFhe\n\n/-! The simulators of the declared reductions as IR programs, for a proof of\n\
         `GeneratedScheme.BoundedSecurityClaim`. Generated deterministically; never edit by hand. -/\n\n\
         namespace GeneratedSimulators\n\n",
    );
    for (index, reduction) in reductions.iter().enumerate() {
        let term = |graph: &Graph| -> Result<String, Box<dyn Error>> {
            Ok(program_term(&mxx_ir_core::validate(graph, &ParamEnv::default())?)?)
        };
        writeln!(
            source,
            "/-- The simulator of reduction {index}. -/\n\
             def simulator_{index} : MxxFhe.ProgramSimulator where\n  \
             setup := {}\n  encrypt := {}\n",
            term(&reduction.setup)?,
            term(&reduction.challenge)?
        )?;
    }
    source.push_str("end GeneratedSimulators\n");
    Ok(source)
}
