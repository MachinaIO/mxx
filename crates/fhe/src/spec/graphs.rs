//! The graphs of a scheme and a circuit, with the input and output names of [`super::roles`].

use super::{Circuit, Depth, Fhe, evaluate, plain_eval, roles};
use mxx_ir_core::Graph;
use mxx_lattice_dsl::GraphValue;

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum BindingError {
    #[error("a circuit needs at least one input")]
    NoInputs,
    #[error("the circuit has depth {circuit}, beyond the scheme's {scheme:?}")]
    Depth { circuit: usize, scheme: Depth },
    #[error("{graph} must read {expected:?}, but reads {found:?}")]
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
    use mxx_lattice_dsl::DslContext;
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

#[cfg(test)]
mod tests {
    use super::{super::roles, graphs};
    use crate::{
        bgv::{
            fixtures::{bgv_params, modswitch_steps},
            scheme::BgvScheme,
        },
        spec::Circuit,
        tfhe::scheme::TfheScheme,
        utils,
    };
    use mxx_core::{expr::Bindings, host, validate::ValidatedProgram};
    use mxx_ir_core::{
        Graph, ParamEnv,
        lower::{int_words, lower, lower_with_int_words},
        validate::validate,
    };
    use std::{collections::BTreeMap, sync::Arc};

    /// Every graph of the BGV and TFHE fixtures lowers to a core program, and the run of its
    /// proof image is generated.
    #[test]
    fn test_fhe_graphs_lower_to_core_programs() {
        let bgv = bgv_params();
        let bgv_scheme = BgvScheme { params: &bgv, modswitch_steps: modswitch_steps() };
        let tfhe = utils::tfhe_params();
        let tfhe_scheme = TfheScheme { params: &tfhe };
        let mut all = graphs(&bgv_scheme, &Circuit::new(3, vec![[1, 2], [0, 3]], 4).unwrap())
            .unwrap()
            .into_iter()
            .map(|(name, graph)| (format!("bgv/{name}"), graph))
            .collect::<Vec<_>>();
        all.extend(
            graphs(&tfhe_scheme, &Circuit::new(2, vec![[0, 1]], 2).unwrap())
                .unwrap()
                .into_iter()
                .map(|(name, graph)| (format!("tfhe/{name}"), graph)),
        );
        // Each lowered program also validates and flattens to a proof image whose code a program
        // logic can list.
        let failures: Vec<String> = all
            .iter()
            .filter_map(|(name, graph)| {
                let validated = validate(graph, &ParamEnv::default()).unwrap();
                let program = match lower(&validated) {
                    Ok(program) => program,
                    Err(error) => return Some(format!("{name}: {error}")),
                };
                let program = match mxx_core::validate::validate(Arc::new(program), Bindings::new())
                {
                    Ok(program) => program,
                    Err(error) => return Some(format!("{name}: {error}")),
                };
                let image = match mxx_core::flatten::flatten(&program, &Default::default()) {
                    Ok(image) => image,
                    Err(error) => return Some(format!("{name}: {error}")),
                };
                // Its run is generated, with placeholder Lean names for the leaves.
                let mut leaf_names = Vec::new();
                if let Ok((events, _)) = &image.trace {
                    leaf_names_of(events, &mut leaf_names);
                }
                let leaves = leaf_names
                    .into_iter()
                    .map(|n| {
                        let lean = mxx_core::compose::LeanLeaf {
                            module: "M".into(),
                            code_at: "At".into(),
                            spec: "spec".into(),
                        };
                        (n, lean)
                    })
                    .collect();
                if let Err(error) = mxx_core::compose::run_module("G", "run", &image, &leaves) {
                    return Some(format!("{name}: run: {error}"));
                }
                // All code is one run of decodable instructions, as a program logic lists it.
                let code_end = image
                    .libraries
                    .iter()
                    .map(|library| library.first_word + library.words)
                    .max()
                    .unwrap_or(image.glue_words);
                let decodable = mxx_core::lean::code_prefix(&image.bytes).len() as u64;
                (decodable < code_end).then(|| {
                    format!("{name}: code ends at word {code_end}, decodable up to {decodable}")
                })
            })
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// The names of the leaves an image's glue calls.
    fn leaf_names_of(events: &[mxx_core::flatten::GlueEvent], out: &mut Vec<String>) {
        for event in events {
            match event {
                mxx_core::flatten::GlueEvent::Leaf { name, .. } => out.push(name.clone()),
                mxx_core::flatten::GlueEvent::Loop { body, .. } => leaf_names_of(body, out),
                _ => {}
            }
        }
    }

    /// The integer width of a scheme's programs: the widest any of its graphs needs.
    fn scheme_words(graphs: &[(String, Graph)]) -> u32 {
        graphs
            .iter()
            .map(|(_, graph)| int_words(&validate(graph, &ParamEnv::default()).unwrap()))
            .max()
            .unwrap()
    }

    /// Lowers `graph` with `words`-word integers and runs it on the host with inputs by name,
    /// giving outputs by name.
    fn run(
        graph: &Graph,
        words: u32,
        inputs: &BTreeMap<String, Vec<u8>>,
    ) -> BTreeMap<String, Vec<u8>> {
        let validated = validate(graph, &ParamEnv::default()).unwrap();
        let lowered = lower_with_int_words(&validated, words).unwrap();
        let program: ValidatedProgram =
            mxx_core::validate::validate(Arc::new(lowered), Bindings::new()).unwrap();
        let ordered = program
            .program
            .inputs
            .iter()
            .map(|(name, _)| inputs.get(name).unwrap_or_else(|| panic!("input {name}")).clone())
            .collect();
        let outputs = host::run(&program, ordered, None).unwrap();
        program.program.outputs.iter().map(|(name, _)| name.clone()).zip(outputs).collect()
    }

    /// The parts of the value `role` among `outputs` (`role`, or `role.port` for a composite
    /// value), renamed to `as_role`.
    fn parts(
        outputs: &BTreeMap<String, Vec<u8>>,
        role: &str,
        as_role: &str,
    ) -> Vec<(String, Vec<u8>)> {
        let parts: Vec<_> = outputs
            .iter()
            .filter_map(|(name, bytes)| {
                let rest = name.strip_prefix(role)?;
                (rest.is_empty() || rest.starts_with('.'))
                    .then(|| (format!("{as_role}{rest}"), bytes.clone()))
            })
            .collect();
        assert!(!parts.is_empty(), "no output {role}");
        parts
    }

    /// Runs the lowered graphs of `graphs` under one key generation on each case of `cases` (one
    /// message per circuit input): encryption, the circuit, and decryption, and checks the result
    /// against `Plain`.
    fn round_trips(graphs: &[(String, Graph)], cases: &[Vec<Vec<i128>>]) {
        let words = scheme_words(graphs);
        let graph = |name: &str| &graphs.iter().find(|(n, _)| n == name).unwrap().1;
        let run = |name: &str, inputs: Vec<(String, Vec<u8>)>| {
            run(graph(name), words, &inputs.into_iter().collect())
        };
        let message = |name: String, values: &[i128]| {
            (name, values.iter().flat_map(|v| host::write_int(*v, words)).collect::<Vec<u8>>())
        };
        let keys = run("Keygen", Vec::new());
        let decrypt =
            if graphs.iter().any(|(n, _)| n == "Decrypt_1") { "Decrypt_1" } else { "Decrypt_0" };
        for messages in cases {
            let mut eval_inputs = parts(&keys, roles::EVALUATION_KEY, roles::EVALUATION_KEY);
            let mut plain_inputs = Vec::new();
            for (index, values) in messages.iter().enumerate() {
                let mut inputs = parts(&keys, roles::PUBLIC_KEY, roles::PUBLIC_KEY);
                inputs.push(message(roles::MESSAGE.into(), values));
                let encrypted = run("Encrypt", inputs);
                eval_inputs.extend(parts(&encrypted, roles::CIPHERTEXT, &roles::ciphertext(index)));
                plain_inputs.push(message(roles::message(index), values));
            }
            let evaluated = run("Eval", eval_inputs);
            let mut inputs = parts(&evaluated, roles::CIPHERTEXT, roles::CIPHERTEXT);
            inputs.extend(parts(&keys, roles::SECRET_KEY, roles::SECRET_KEY));
            let decrypted = run(decrypt, inputs);
            let expected = run("Plain", plain_inputs);
            assert_eq!(decrypted[roles::MESSAGE], expected[roles::MESSAGE], "{messages:?}");
        }
    }

    /// The lowered BGV programs evaluate `x0 (x1 x2)` on random slots.
    #[test]
    fn test_lowered_bgv_programs_decrypt_their_circuit() {
        use rand::Rng;
        let bgv = bgv_params();
        let scheme = BgvScheme { params: &bgv, modswitch_steps: modswitch_steps() };
        let graphs = graphs(&scheme, &Circuit::new(3, vec![[1, 2], [0, 3]], 4).unwrap()).unwrap();
        let mut rng = rand::rng();
        let slots = mxx_backends::poly::PolyParams::ring_dimension(&bgv.common.ring) as usize;
        let messages: Vec<Vec<i128>> = (0..3)
            .map(|_| {
                (0..slots).map(|_| i128::from(rng.random_range(0..bgv.plaintext_modulus))).collect()
            })
            .collect();
        round_trips(&graphs, &[messages]);
    }

    /// The lowered TFHE programs evaluate one NAND gate on every pair of bits.
    #[test]
    fn test_lowered_tfhe_programs_decrypt_their_gate() {
        let tfhe = utils::tfhe_params();
        let scheme = TfheScheme { params: &tfhe };
        let graphs = graphs(&scheme, &Circuit::new(2, vec![[0, 1]], 2).unwrap()).unwrap();
        let cases: Vec<Vec<Vec<i128>>> =
            [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(x, y)| vec![vec![x], vec![y]]).to_vec();
        round_trips(&graphs, &cases);
    }
}
