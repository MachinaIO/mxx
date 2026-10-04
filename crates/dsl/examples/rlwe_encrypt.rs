//! Encrypts and decrypts one bit with Ring-LWE on the GPU, and exports the Lean statement that
//! every execution decrypts its message to `crates/dsl/examples/rlwe/generated`, where
//! `lake build` checks its proof.
//!
//! Regenerate without a GPU with `cargo run -p mxx-dsl --example rlwe_encrypt --
//! --export-lean <directory>`. This mode only constructs and exports the protocol.
//!
//! Run with `cargo run -r -p mxx-dsl --example rlwe_encrypt --features gpu`.

use bigdecimal::BigDecimal;
use mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound;
#[cfg(feature = "gpu")]
use mxx_backends::{
    GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::dcrt::gpu::GpuDCRTPolyParams,
};
use mxx_dsl::{BuiltGraph, DslContext, DslError, HashTag, IdealSpec, Int, IntType, Ring};
#[cfg(feature = "gpu")]
use mxx_ir_core::generate_crt_basis;
use mxx_ir_core::{
    IntExpr, ParamEnv, Rational, RealExpr,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OutputRef, ParameterDecl, ParameterKind,
        ProtocolDecl, ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId,
        ProtocolPreconditionSpec, ProtocolStage, StageId, StageInputName, Workflow,
    },
};
use num_bigint::BigInt;
use std::collections::BTreeMap;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Describes the protocol. Named parameters stay symbolic until they are bound.
fn rlwe_program(ring_dimension: u32) -> Result<BuiltGraph, DslError> {
    // R_Q = Z_Q[X]/(X^N + 1), where Q is a product of `crt_depth` primes of `crt_bits` bits.
    let ring = Ring::new(
        IntExpr::Var("crt_bits".into()),
        IntExpr::Var("crt_depth".into()),
        ring_dimension,
    );
    // Gaussian parameters: the width sigma and the bound on every sampled coefficient.
    let sigma = RealExpr::Var("sigma".into());
    let cutoff = IntExpr::Var("cutoff".into());
    let context = DslContext::new("rlwe-encrypt")
        .int_parameter("crt_bits")
        .int_parameter("crt_depth")
        .int_parameter("gadget_base_bits")
        .int_parameter("cutoff")
        .real_parameter("sigma");

    // Inputs supplied at run time: a 32-byte public seed and the message bit (0 or 1).
    let seed = ring.bytes_input("seed", 32);
    let message: Int = context.input("message", IntType)?;

    // The public element a is derived from the seed, so another party can recompute it.
    let a = ring.hash_matrix(seed, HashTag::from(b"rlwe-example/a".as_slice()), (1, 1));
    let s = ring.gaussian((1, 1), sigma.clone(), cutoff.clone());
    let e = ring.gaussian((1, 1), sigma, cutoff);

    // Encrypt the message as the constant term: b = a*s + e + floor(Q/2) * message.
    let delta = ring.polynomial([ring.modulus().floor_div(2)]);
    let plaintext = message.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let b = &a * &s + e + &delta * &plaintext;

    // Decrypt: round the constant term of b - a*s to the nearest multiple of Q/2.
    let decrypted = (b.clone() - &a * &s).threshold_decode_bools(2, 1).remove(0);

    context.output("ciphertext", b)?.output("decrypted", decrypted)?.build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let export_directory = match args.as_slice() {
        [] => None,
        [flag, directory] if flag == "--export-lean" => Some(std::path::PathBuf::from(directory)),
        _ => return Err("usage: rlwe_encrypt [--export-lean <directory>]".into()),
    };
    if export_directory.is_none() && !cfg!(feature = "gpu") {
        return Err("GPU execution requires --features gpu; use --export-lean <directory> for host-only export".into());
    }
    let ring_dimension = 4096;
    let (crt_bits, crt_depth, gadget_base_bits) = (60, 3, 20);
    let sigma = 4;
    let cutoff = hard_cutoff_from_sigma_bound(&BigDecimal::from(sigma)); // 6.5 sigma

    // Bind the parameters, then check the whole program under them.
    let bindings = ParamEnv {
        integers: BTreeMap::from([
            ("crt_bits".to_owned(), BigInt::from(crt_bits)),
            ("crt_depth".to_owned(), BigInt::from(crt_depth)),
            ("gadget_base_bits".to_owned(), BigInt::from(gadget_base_bits)),
            ("cutoff".to_owned(), BigInt::from(cutoff)),
        ]),
        reals: BTreeMap::from([("sigma".to_owned(), Rational::from_integer(BigInt::from(sigma)))]),
        ..ParamEnv::default()
    };
    let program = rlwe_program(ring_dimension)?;

    // State the program's correctness in Lean: for every seed and message bit, the decrypted bit
    // is the message. `lake build` in `crates/dsl/examples/rlwe` checks the proof.
    let lean = export_directory.clone().unwrap_or_else(|| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/rlwe/generated")
    });
    mxx_ir_core::lean::protocol::export(&correctness_protocol(&program, &bindings)?, &lean)?;

    if export_directory.is_some() {
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    {
        let program = program.validate(&bindings)?;

        // Register the same ring with a GPU backend.
        let moduli = generate_crt_basis(ring_dimension, crt_depth, crt_bits)?;
        let gpu_params = GpuDCRTPolyParams::new(ring_dimension, moduli, gadget_base_bits, None);
        let mut runtime = GpuRuntime::new(gpu_backend([gpu_params]))?;

        let inputs = |seed: u8, bit: bool| {
            BTreeMap::from([
                ("seed".to_owned(), RuntimeValue::Bytes(Arc::from([seed; 32]))),
                ("message".to_owned(), RuntimeValue::Int(BigInt::from(bit))),
            ])
        };

        // Planning picks the parallelism and memory schedule that fit this GPU. It needs inputs of
        // the right types, so it gets dummy ones.
        let mut plan = runtime.plan(program, &inputs(0, false))?;

        // The plan then runs on real inputs without being planned again.
        for (seed, bit) in [(1, false), (2, true)] {
            let result = runtime.execute(&mut plan, inputs(seed, bit))?;
            assert_eq!(runtime.download_bool(&result["decrypted"])?, bit);
            println!("bit {bit} decrypted correctly");
        }
    }
    Ok(())
}

/// The program as a one-stage protocol whose ideal functionality returns the message bit.
fn correctness_protocol(
    program: &BuiltGraph,
    bindings: &ParamEnv,
) -> Result<ProtocolDecl, Box<dyn std::error::Error>> {
    // Every graph of a protocol declares the same parameters.
    let params = [
        ("crt_bits", ParameterKind::Integer),
        ("crt_depth", ParameterKind::Integer),
        ("gadget_base_bits", ParameterKind::Integer),
        ("cutoff", ParameterKind::Integer),
        ("sigma", ParameterKind::Rational),
    ]
    .into_iter()
    .map(|(name, kind)| ParameterDecl { name: name.to_owned(), kind })
    .collect();
    let ideal = DslContext::new("rlwe-ideal")
        .int_parameter("crt_bits")
        .int_parameter("crt_depth")
        .int_parameter("gadget_base_bits")
        .int_parameter("cutoff")
        .real_parameter("sigma");
    let message: Int = ideal.input("message", IntType)?;
    let ideal =
        IdealSpec::new(ideal.output("decrypted", Int::constant(0).less(message))?.build()?.graph)?;

    let stage = StageId("rlwe".to_owned());
    let endpoint = EndpointSpecId::Exact;
    let destination = |input: &str| ProtocolInputDestination::WorkflowStage {
        stage: stage.clone(),
        input: StageInputName(input.to_owned()),
    };
    let (contracts, input_bindings) = [
        ("seed", InputValueContract::Bytes { length: 32.into() }, vec![destination("seed")]),
        (
            "message",
            InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() },
            vec![
                destination("message"),
                ProtocolInputDestination::Ideal { input: "message".to_owned() },
            ],
        ),
    ]
    .into_iter()
    .map(|(name, value, destinations)| {
        (
            InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
            ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
        )
    })
    .unzip();
    Ok(ProtocolDecl::new(ProtocolDecl {
        params,
        bindings: bindings.clone(),
        // The Gaussian samples are truncated, so every execution decrypts correctly.
        failure_probability_log2: None,
        bundle: ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![ProtocolStage {
                    id: stage.clone(),
                    graph: program.graph.clone(),
                    bindings: Vec::new(),
                }],
                entrypoint: stage.clone(),
            },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decrypted".to_owned(),
                    ideal_input: "decrypted".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef { stage, output: "decrypted".to_owned() },
                    ideal_output: "decrypted".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings,
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })?)
}
