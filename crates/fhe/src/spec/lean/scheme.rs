//! `Scheme.lean`: the scheme of the exported graphs and the claims of a statement.

use super::{
    Correctness, Statement, roles,
    root::{Root, is_simulator, names},
};
use mxx_ir_core::{
    ParamEnv,
    lean::{LeanArtifact, claim::input_contract_predicate, protocol::input_contract},
};
use std::{collections::BTreeMap, fmt::Write as _};

pub(super) fn render(
    message_space: &mxx_ir_core::protocol::InputValueContract,
    inputs: usize,
    types: usize,
    statement: &Statement,
    artifacts: &BTreeMap<String, LeanArtifact>,
) -> Result<String, Box<dyn std::error::Error>> {
    let root = |module: &str| -> Result<Root<'_>, String> {
        let artifact = artifacts.get(module).ok_or(format!("missing module {module}"))?;
        Ok(Root { module: &artifact.module_name, root: &artifact.root })
    };
    let keygen = root("Keygen")?;
    keygen.expect_inputs(&[])?;
    keygen.expect_outputs(&names(&[
        roles::PUBLIC_KEY,
        roles::EVALUATION_KEY,
        roles::SECRET_KEY,
    ]))?;
    let encrypt = root("Encrypt")?;
    encrypt.expect_inputs(&names(&[roles::PUBLIC_KEY, roles::MESSAGE]))?;
    encrypt.expect_outputs(&names(&[roles::CIPHERTEXT]))?;
    let mut source = String::from("import Backend\n");
    for module in artifacts.keys().filter(|module| !is_simulator(module)) {
        writeln!(source, "import {module}")?;
    }
    source.push_str(
        "import MxxFhe\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\n\
         /-! The public-key homomorphic encryption scheme of the exported graphs and its\n\
         correctness and security statements. Generated deterministically; never edit by hand. -/\n\n\
         namespace GeneratedScheme\n\nopen MxxSecurity MxxFhe\n\n",
    );
    for (_, artifact) in artifacts.iter().filter(|(name, _)| !is_simulator(name)) {
        source.push_str(
            &Root { module: &artifact.module_name, root: &artifact.root }.params_definition(),
        );
    }
    let contract = input_contract(message_space)?;
    let message = encrypt.root.inputs.get(roles::MESSAGE).ok_or("Encrypt has no message")?;
    let valid =
        input_contract_predicate(&contract, &message.wire_type, &ParamEnv::default(), "m", 0)?;
    write!(
        source,
        "\nabbrev Msg := {}\nabbrev PK := {}\nabbrev EK := {}\nabbrev SK := {}\n\n",
        message.lean_type,
        keygen.output_type(roles::PUBLIC_KEY)?,
        keygen.output_type(roles::EVALUATION_KEY)?,
        keygen.output_type(roles::SECRET_KEY)?,
    )?;
    source.push_str("/-- A ciphertext of each type. -/\ninductive Ciphertext where\n");
    for index in 0..types {
        let decrypt = root(&format!("Decrypt_{index}"))?;
        decrypt.expect_inputs(&names(&[roles::SECRET_KEY, roles::CIPHERTEXT]))?;
        decrypt.expect_outputs(&names(&[roles::MESSAGE]))?;
        writeln!(source, "  | ct{index} (value : {})", decrypt.input_type(roles::CIPHERTEXT)?)?;
    }
    let values = |pairs: &[(&str, &str)]| {
        pairs.iter().map(|(name, value)| ((*name).to_owned(), (*value).to_owned())).collect()
    };
    write!(
        source,
        "\ninstance : Nonempty Ciphertext := ⟨.ct0 (Classical.arbitrary _)⟩\n\n\
         noncomputable def keyGen : Comp (PK × EK × SK) := do\n  let outputs ← {} ()\n  pure ({}, {}, {})\n\n\
         noncomputable def enc (pk : PK) (m : Msg) : Comp Ciphertext := do\n  let outputs ← {} {}\n  pure (.ct0 {})\n\n\
         noncomputable def dec (sk : SK) : Ciphertext → Comp Msg\n",
        keygen.comp(),
        keygen.output(roles::PUBLIC_KEY, "outputs")?,
        keygen.output(roles::EVALUATION_KEY, "outputs")?,
        keygen.output(roles::SECRET_KEY, "outputs")?,
        encrypt.comp(),
        encrypt.inputs(&values(&[(roles::PUBLIC_KEY, "pk"), (roles::MESSAGE, "m")]))?,
        encrypt.output(roles::CIPHERTEXT, "outputs")?,
    )?;
    for index in 0..types {
        let decrypt = root(&format!("Decrypt_{index}"))?;
        write!(
            source,
            "  | .ct{index} value => do\n    let outputs ← {} {}\n    pure {}\n",
            decrypt.comp(),
            decrypt.inputs(&values(&[(roles::SECRET_KEY, "sk"), (roles::CIPHERTEXT, "value")]))?,
            decrypt.output(roles::MESSAGE, "outputs")?,
        )?;
    }
    write!(
        source,
        "\n/-- The scheme. -/\nnoncomputable def scheme : MxxFhe.Scheme where\n  Msg := Msg\n  ValidMsg m := {valid}\n  PK := PK\n  EK := EK\n  SK := SK\n  CT := Ciphertext\n  keyGen := keyGen\n  enc := enc\n  dec := dec\n\n\
         instance : DecidableEq scheme.Msg := inferInstanceAs (DecidableEq Msg)\n\n"
    )?;
    let eval = root("Eval")?;
    let mut allowed = (0..inputs).map(roles::ciphertext).collect::<Vec<_>>();
    allowed.push(roles::EVALUATION_KEY.to_owned());
    eval.expect_inputs_within(&allowed)?;
    let keyed = eval.reads(roles::EVALUATION_KEY);
    eval.expect_outputs(&names(&[roles::CIPHERTEXT]))?;
    let mut supplied =
        (0..inputs).map(|i| (roles::ciphertext(i), format!("c{i}"))).collect::<BTreeMap<_, _>>();
    if keyed {
        supplied.insert(roles::EVALUATION_KEY.to_owned(), "ek".into());
    }
    let plain = root("Plain")?;
    plain.expect_inputs_within(&(0..inputs).map(roles::message).collect::<Vec<_>>())?;
    plain.expect_outputs(&names(&[roles::MESSAGE]))?;
    let messages = (0..inputs)
        .map(|i| (roles::message(i), format!("messages ⟨{i}, by decide⟩")))
        .collect::<BTreeMap<_, _>>();
    let backend = if plain.root.requires_backend { " Backend.backend" } else { "" };
    let arguments =
        (0..inputs).map(|i| format!("cts ⟨{i}, by decide⟩")).collect::<Vec<_>>().join(", ");
    let patterns = (0..inputs).map(|i| format!(".ct0 c{i}")).collect::<Vec<_>>().join(", ");
    write!(
        source,
        "/-- The circuit as the scheme evaluates it. `Eval` on an argument that is not fresh \
         returns\nits first argument. -/\n\
         noncomputable def evaluation : MxxFhe.Evaluation scheme {inputs} where\n  \
         plain messages :=\n    let outputs := MxxSecurity.determined fun outputs : {} ↦\n      \
         {}.generatedRoot{backend} {} {} outputs\n    {}\n  \
         eval ek cts :=\n    match {arguments} with\n    | {patterns} => do\n      \
         let outputs ← {} {}\n      pure (.ct{} {})\n",
        plain.root.output_type,
        plain.module,
        plain.params_name(),
        plain.inputs(&messages)?,
        plain.output(roles::MESSAGE, "outputs")?,
        eval.comp(),
        eval.inputs(&supplied)?,
        types - 1,
        eval.output(roles::CIPHERTEXT, "outputs")?,
    )?;
    if types > 1 {
        let wildcard = vec!["_"; inputs].join(", ");
        writeln!(source, "    | {wildcard} => pure (cts ⟨0, by decide⟩)")?;
    }
    source.push('\n');
    let security = statement.security.as_ref();
    for (index, assumption) in security.iter().flat_map(|s| s.assumptions.iter()).enumerate() {
        writeln!(
            source,
            "def params_{index} : MxxSecurity.AssumptionId.Params {} := {}\n",
            assumption.id().lean_name(),
            assumption.lean()
        )?;
    }
    let correctness = match statement.correctness {
        Correctness::WorstCase => {
            "/-- Every execution decrypts the circuit's evaluation correctly. -/\n\
             def CorrectnessClaim : Prop :=\n  MxxFhe.WorstCaseCorrect scheme evaluation"
                .to_owned()
        }
        Correctness::AverageCase { failure_probability_log2: k } => format!(
            "/-- Decryption of the circuit's evaluation fails with probability at most `2^-{k}`. \
             -/\ndef CorrectnessClaim : Prop :=\n  \
             MxxFhe.AverageCaseCorrect scheme evaluation ((2 : ENNReal)⁻¹ ^ {k})"
        ),
    };
    writeln!(source, "{correctness}\n")?;
    if let Some(security) = security {
        let references = security
            .assumptions
            .iter()
            .enumerate()
            .map(|(index, assumption)| format!("⟨{}, params_{index}⟩", assumption.id().lean_name()))
            .collect::<Vec<_>>()
            .join(",\n    ");
        let fresh = root("Decrypt_0")?.input_type(roles::CIPHERTEXT)?;
        let bound = security.cost_bound;
        write!(
            source,
            "/-- The conversions between the scheme's values and IR values; a challenge ciphertext is\n\
             fresh. -/\n\
             noncomputable def codec : MxxFhe.Codec scheme where\n  \
             msg := MxxSecurity.IR.ToValue.encode (α := Msg)\n  \
             pk := MxxSecurity.IR.FromValue.decode (α := PK)\n  \
             ek := MxxSecurity.IR.FromValue.decode (α := EK)\n  \
             ct value := (MxxSecurity.IR.FromValue.decode value : Option ({fresh})).map Ciphertext.ct0\n\n\
             /-- The declared assumptions. -/\nnoncomputable def assumptions : List MxxFhe.AssumptionRef :=\n  [{references}]\n\n\
             /-- IND-CPA security relative to the declared assumptions, without a statistical term, with\n\
             reductions whose simulators are IR programs of cost at most {bound}, for every backend\n\
             and hash function the programs run with. -/\n\
             def BoundedSecurityClaim : Prop :=\n  ∀ backend : MxxSecurity.IR.Backend,\n    \
             MxxFhe.SecureWithin backend scheme codec assumptions {bound} 0\n\n"
        )?;
    }
    source.push_str("end GeneratedScheme\n");
    Ok(source)
}
