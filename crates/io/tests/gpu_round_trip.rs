#![cfg(feature = "gpu")]

//! The Diamond iO 2 round trip on the GPU runtime: a two-step streaming function is obfuscated
//! (one obfuscation graph, executed once) with random obfuscator chunks and initial state, and
//! its obfuscation is evaluated (one evaluation graph per input) on every evaluator input against
//! the clear function. Round trips run only on the GPU.
//!
//! The parameters are the smallest noise-free ones of the implementation's first validation
//! (`crates/io/DIO2-SPEC.md`, §13): the preprocessing ciphertexts have no error, so the only
//! noise is the fresh CBD(1) noise of every refresh and the smudges of the decoders.
//!
//! | Variable (`MXX_IO_DIO2_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `CRT_BITS`, `BASE_BITS` | 50, 12 | CRT prime and gadget base widths |
//! | `PRIMES`, `LEVELS` | 5, 6 | Primes of `Q_0` and levels of the chain (every one of `Q_0`) |
//! | `SMUDGE_DIGITS` | 1 | Smudge digits of every decoder |
//! | `TRAPDOOR_SIGMA` | 4 | Width of the anchor trapdoor |

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, MemoryArtifactStore,
    backend::poly_gpu::gpu_backend_on,
    openfhe_guard::gen_modulus_and_warmup,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};
use mxx_exponent_lut::{ExpLayout, refresh::PrfSettings};
use mxx_io::{
    Obfuscation,
    dio2::{Dio2, Dio2Function, Dio2Parameters, StreamingFunction},
};
use serial_test::serial as sequential;

fn read(name: &str, default: &str) -> String {
    std::env::var(format!("MXX_IO_DIO2_{name}")).unwrap_or_else(|_| default.to_owned())
}

/// Ring dimension 8, the PRF on `Q = 16`, `P = 8` with one LWR coordinate and weight one; a
/// function of one state bit, one evaluator bit and one obfuscator bit on the state layout
/// `(1, 2)` (`mono(e) = X^(2 e)`, `e < 4`) with two table steps.
fn toy() -> (Dio2Parameters, StreamingFunction) {
    let crt_bits = read("CRT_BITS", "50").parse().unwrap();
    let primes = gen_modulus_and_warmup(8, read("PRIMES", "5").parse().unwrap(), crt_bits, None)
        .expect("CRT basis");
    let parameters = Dio2Parameters {
        ring_dimension: 8,
        crt_bits,
        base_bits: read("BASE_BITS", "12").parse().unwrap(),
        primes,
        levels: read("LEVELS", "6").parse().unwrap(),
        prf: PrfSettings {
            lwr_dimension: 1,
            support_weight: 1,
            input_modulus: 16,
            output_modulus: 8,
            bucket_width: 0,
        },
        smudge_digits: read("SMUDGE_DIGITS", "1").parse().unwrap(),
        trapdoor_sigma: read("TRAPDOOR_SIGMA", "4").parse().unwrap(),
        encoding_error_sigma: BigDecimal::from(0),
    };
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 });
    let function = [vec![1, 0, 1, 0], vec![0, 1, 1, 0]]
        .into_iter()
        .fold(function.clone(), |acc, table| acc.with_step_row(table).unwrap());
    (parameters, function)
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_dio2_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let (parameters, function) = toy();
    let runtime = GpuRuntime::new(gpu_backend_on(
        parameters.rings().unwrap().into_iter().map(|ring| {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        }),
        detected_gpu_device_ids(),
    ))
    .expect("GPU runtime");
    let mut dio2 = Dio2::new(parameters, runtime, MemoryArtifactStore::default()).unwrap();
    let steps = function.stream_length();
    let y = (0..steps)
        .map(|_| (0..function.obf_chunk_bits).map(|_| rand::random()).collect())
        .collect::<Vec<Vec<bool>>>();
    let initial_state = rand::random::<u8>() as usize % (1 << function.state_bits);
    let started = std::time::Instant::now();
    let obfuscation = dio2
        .obfuscate(&Dio2Function { function: function.clone(), y: y.clone(), initial_state })
        .unwrap();
    eprintln!("obfuscation: {:?}", started.elapsed());
    let bits = steps * function.eval_chunk_bits;
    for input in 0..1usize << bits {
        let x = (0..steps)
            .map(|step| {
                (0..function.eval_chunk_bits)
                    .map(|bit| input >> (step * function.eval_chunk_bits + bit) & 1 == 1)
                    .collect()
            })
            .collect::<Vec<Vec<bool>>>();
        let started = std::time::Instant::now();
        let output = dio2.evaluate(&obfuscation, &x).unwrap();
        eprintln!("evaluation of {x:?}: {:?}", started.elapsed());
        assert_eq!(output, function.evaluate(initial_state, &x, &y), "input {x:?}");
    }
}
