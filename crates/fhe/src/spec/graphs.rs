//! The graphs of a scheme and a circuit, with the input and output names of [`super::roles`].

use super::{Circuit, Depth, Fhe, evaluate, plain_eval, roles};
use mxx_dsl::GraphValue;
use mxx_ir_core::Graph;

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum BindingError {
    #[error("a circuit needs at least one input")]
    NoInputs,
    #[error("the circuit has depth {circuit}, beyond the scheme's {scheme:?}")]
    Depth { circuit: usize, scheme: Depth },
    #[error("{graph} must read exactly {expected:?}, but reads {found:?}")]
    Inputs { graph: String, expected: Vec<String>, found: Vec<String> },
    #[error("{graph} must write exactly {expected:?}, but writes {found:?}")]
    Outputs { graph: String, expected: Vec<String>, found: Vec<String> },
    #[error("{graph} must write {missing}")]
    MissingOutput { graph: String, missing: String },
    #[error("the statement has {assumptions} assumptions but {reductions} reductions")]
    Reductions { assumptions: usize, reductions: usize },
}

/// The graphs of `scheme` and `circuit`, by the module names of [`super::lean::export`]: `Keygen`,
/// `Encrypt`, `Eval` of the circuit on fresh ciphertexts, `Decrypt_{i}` for each ciphertext type
/// (`0` for the type `Enc` returns and, when it differs, `1` for the type `Eval` returns), and
/// `Plain`, the circuit's `plain_eval`. These are the graphs a runtime executes and the statements
/// name.
pub fn graphs<S: Fhe>(
    scheme: &S,
    circuit: &Circuit,
) -> Result<Vec<(String, Graph)>, Box<dyn std::error::Error>> {
    use mxx_dsl::DslContext;
    let build = |context: DslContext| -> Result<Graph, Box<dyn std::error::Error>> {
        Ok(context.build()?.graph)
    };
    let inputs = circuit.input_count();
    if inputs == 0 {
        return Err(BindingError::NoInputs.into());
    }
    if !scheme.depth().admits(circuit.depth()) {
        return Err(BindingError::Depth { circuit: circuit.depth(), scheme: scheme.depth() }.into());
    }
    let (message_schema, _) = scheme.message_space();
    let (public_key, evaluation_key, secret_key) = scheme.key_generation();
    let (public_key_schema, evaluation_key_schema, secret_key_schema) =
        (public_key.schema(), evaluation_key.schema(), secret_key.schema());
    let mut graphs: Vec<(String, Graph)> = vec![(
        "Keygen".into(),
        build(
            DslContext::new("keygen")
                .output(roles::SECRET_KEY, secret_key)?
                .output(roles::PUBLIC_KEY, public_key)?
                .output(roles::EVALUATION_KEY, evaluation_key)?,
        )?,
    )];
    let context = DslContext::new("encrypt");
    let message: S::Message = context.input(roles::MESSAGE, message_schema.clone())?;
    let public_key: S::PublicKey = context.input(roles::PUBLIC_KEY, public_key_schema)?;
    let ciphertext = scheme.encrypt(&public_key, &message);
    let mut types = vec![ciphertext.schema()];
    graphs.push(("Encrypt".into(), build(context.output(roles::CIPHERTEXT, ciphertext)?)?));
    let context = DslContext::new("eval");
    let ciphertexts = (0..inputs)
        .map(|i| context.input(roles::ciphertext(i), types[0].clone()))
        .collect::<Result<Vec<S::Ciphertext>, _>>()?;
    let evaluation_key: S::EvaluationKey =
        context.input(roles::EVALUATION_KEY, evaluation_key_schema)?;
    let output = evaluate(scheme, &evaluation_key, circuit, &ciphertexts);
    if output.schema() != types[0] {
        types.push(output.schema());
    }
    graphs.push(("Eval".into(), build(context.output(roles::CIPHERTEXT, output)?)?));
    for (index, schema) in types.iter().enumerate() {
        let context = DslContext::new("decrypt");
        let ciphertext: S::Ciphertext = context.input(roles::CIPHERTEXT, schema.clone())?;
        let secret_key: S::SecretKey =
            context.input(roles::SECRET_KEY, secret_key_schema.clone())?;
        let message = scheme.decrypt(&secret_key, &ciphertext);
        graphs.push((format!("Decrypt_{index}"), build(context.output(roles::MESSAGE, message)?)?));
    }
    let context = DslContext::new("plain-eval");
    let messages = (0..inputs)
        .map(|i| context.input(roles::message(i), message_schema.clone()))
        .collect::<Result<Vec<S::Message>, _>>()?;
    let output = plain_eval(scheme, circuit, &messages);
    graphs.push(("Plain".into(), build(context.output(roles::MESSAGE, output)?)?));
    Ok(graphs)
}
