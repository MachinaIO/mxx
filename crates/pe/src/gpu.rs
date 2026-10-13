//! GPU round trip of the scheme: every stage on the GPU runtime.
use crate::{
    BranchingProgram, PeClass, decrypt, encrypt, keygen, setup,
    test_support::{equals, parameters},
};
use mxx_backends::{
    GpuRuntime, MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
};

/// The GPU parameters of every ring of `parameters`.
fn gpu_rings(parameters: &crate::PeParameters) -> Vec<GpuDCRTPolyParams> {
    parameters
        .rings()
        .unwrap()
        .into_iter()
        .map(|ring| {
            GpuDCRTPolyParams::new(
                ring.ring_dimension(),
                ring.moduli().to_vec(),
                ring.base_bits(),
                None,
            )
        })
        .collect()
}

/// One setup, one ciphertext of a random message, and keys for an
/// authorized and an unauthorized equality predicate, all on the GPU, with
/// parameters whose simulated bound guarantees correctness (by default the
/// noise-free test parameters): the authorized key decrypts the message,
/// and the unauthorized decryption returns a rounded phase, which this test
/// does not constrain.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_authorized_key_decrypts_on_the_gpu() {
    let class = PeClass::new(2, 2, 2, 2).unwrap();
    authorized_key_decrypts_on_the_gpu(class, parameters(class));
}

/// [`test_gpu_authorized_key_decrypts_on_the_gpu`] over a statistical trapdoor with a baby-step
/// split and compact encodings: the composed branch `25 = 17 * 9` reads compact helpers and
/// companions in both switches, all on the GPU with the `q` and `pq` rings registered.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_compact_encodings_with_a_split_decrypt_on_the_gpu() {
    let class = PeClass::new(2, 2, 2, 2).unwrap();
    let parameters = crate::test_support::correct_parameters(class, true, |parameters| {
        parameters.with_baby_steps(2).unwrap().with_compact_encodings().unwrap()
    });
    authorized_key_decrypts_on_the_gpu(class, parameters);
}

fn authorized_key_decrypts_on_the_gpu(class: PeClass, parameters: crate::PeParameters) {
    let level = parameters.level();
    let mut execution = GpuRuntime::new(gpu_backend(gpu_rings(&parameters))).expect("GPU runtime");
    let bound = crate::noise::simulate(&parameters, &BranchingProgram::longest(class)).unwrap();
    assert!(bound.decrypts(), "the parameters must guarantee correctness");
    let mut store = MemoryArtifactStore::default();
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    let attribute = [0; 2].map(|_| rand::random::<u32>() as usize % 2);
    let message = (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let ciphertext =
        encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
    let complement = attribute.map(|bit| 1 - bit);
    let predicates: [BranchingProgram; 2] = [equals(class, &attribute), equals(class, &complement)];
    for (predicate, value) in predicates.iter().zip([0, 1]) {
        assert_eq!(predicate.evaluate(&attribute).unwrap(), value);
        let key = keygen(&public_key, &master, predicate, &mut execution, &mut store).unwrap();
        let decrypted =
            decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
        if value == 0 {
            assert_eq!(decrypted, message);
        }
    }
}

/// The measured noise of one authorized decryption against the simulated bounds, at parameters
/// read from `MXX_PE_NOISE_*` (statistical, average case, no estimator): for each of `TRIALS`
/// fresh ciphertexts and keys of a random maximal-length program, the evaluation error
/// `e_F = c_F - A_F^(2) - s_bar A_F^(1)` of the output row (the secret is known to the test) and
/// the phase error `phase - floor(q/2) m`, each as `log2` of its largest and root-mean-square
/// centered coefficient, next to `log2 E_F`, `log2 E_PE`, and the average-case bias and standard
/// deviation of [`crate::noise::simulate_average`].
#[test]
#[ignore = "requires a CUDA GPU; measures noise"]
fn test_gpu_measured_noise_against_the_bounds() {
    use crate::{
        PeParameterRequest, PeParameters,
        budget::{DEFAULT_FLOODING_BITS, DEFAULT_STATISTICAL_BITS, KeyWidthPolicy},
        decryption::{OUTPUT_ROW, evaluate, phase_graph},
        encryption::encrypt_with_secret,
        params::TrapdoorMode,
    };
    use mxx_backends::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::Poly,
    };
    use mxx_dsl::{DslContext, Mat};
    use mxx_exponent_lut::api::{self, StageExecution, StageGraph, StageOutputs};
    use mxx_ir_core::{ParamEnv, artifact::ArtifactAvailability, node::ConcatAxis};
    use num_bigint::BigUint;
    use std::collections::BTreeMap;

    fn read<T: std::str::FromStr>(name: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        std::env::var(format!("MXX_PE_NOISE_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    }
    let log2 = |value: &BigUint| -> f64 {
        let bits = value.bits();
        let shift = bits.saturating_sub(52);
        let top = (value >> shift).to_string().parse::<f64>().unwrap();
        top.log2() + shift as f64
    };
    // `log2` of the largest and of the root-mean-square centered coefficient.
    let statistics = |q: &BigUint, values: &[BigUint]| -> (f64, f64) {
        let centered = values.iter().map(|value| value.clone().min(q - value)).collect::<Vec<_>>();
        let max = centered.iter().max().unwrap().clone();
        let energy = centered.iter().map(|value| value * value).sum::<BigUint>();
        let mean = log2(&energy) - (centered.len() as f64).log2();
        (log2(&max), mean / 2.0)
    };

    let class = PeClass::new(
        read("ATTRIBUTE_LENGTH", 1),
        read("ALPHABET_SIZE", 2),
        read("STATE_WIDTH", 1),
        read("MAX_STEPS", 1),
    )
    .unwrap();
    let request = PeParameterRequest {
        class,
        ring_dimension: read("RING_DIMENSION", 8192),
        crt_bits: read("CRT_BITS", 32),
        base_bits: read("BASE_BITS", 16),
        crt_depths: read("MIN_CRT_DEPTH", 1)..=16,
        trapdoor_instantiation: TrapdoorMode::Statistical,
        key_width_policy: KeyWidthPolicy::Practical,
        trapdoor_sigma: read("TRAPDOOR_SIGMA", 64.0),
        error_sigma: "3.19".parse().unwrap(),
        flooding_sigma: "3.19".parse().unwrap(),
        enforce_flooding: read("FLOODING", "none".to_owned()) == "enforce",
        programming_sigma: read("PROGRAMMING_SIGMA", "256".to_owned()).parse().unwrap(),
        randomness_sigma: read("RANDOMNESS_SIGMA", "4096".to_owned()).parse().unwrap(),
        statistical_security_bits: DEFAULT_STATISTICAL_BITS,
        flooding_statistical_bits: DEFAULT_FLOODING_BITS,
        key_queries: 1 << 20,
        max_anchor_columns: 1024,
        lut_baby_steps: Some(read("BABY_STEPS", 0usize)).filter(|steps| *steps > 0),
        compact_encodings: read("COMPACT", 0usize) != 0,
        failure_exponent: Some(128),
        lattice_security_bits: None,
    };
    let (parameters, report) = PeParameters::select(&request).unwrap();
    let level = parameters.level();
    let dimension = level.ring_dimension() as usize;
    let q = parameters.modulus();
    let (_, _, moments) = report.average.clone().unwrap();
    let rational = |value: &mxx_exponent_lut::noise_utils::NonnegativeRational| {
        log2(&value.numerator) - log2(&value.denominator)
    };
    eprintln!(
        "parameters: n = {dimension}, CRT {} x {} (log2 q = {}), base 2^{}, m = {}, m_B = {}, W = \
         {}, T = {}; bounds: log2 E_F = {:.1}, log2 E_PE = {:.1}, average bias {:.1}, sd {:.1}",
        level.crt_depth(),
        request.crt_bits,
        q.bits(),
        level.base_bits(),
        parameters.state_columns(),
        parameters.anchor_columns(),
        class.lookup_width(),
        class.max_steps(),
        log2(&report.evaluation),
        log2(&report.decryption),
        rational(&moments.mean_radius),
        rational(&moments.variance) / 2.0,
    );
    let mut execution = GpuRuntime::new(gpu_backend(gpu_rings(&parameters))).expect("GPU runtime");
    let mut store = match read("STORAGE", "device".to_owned()).as_str() {
        "device" => MemoryArtifactStore::on_device(),
        "host" => MemoryArtifactStore::default(),
        other => panic!("unknown storage {other}"),
    };
    let download = |execution: &GpuRuntime, outputs: &StageOutputs, name: &str| {
        <GpuRuntime as StageExecution<MemoryArtifactStore>>::download_matrix(
            execution,
            &outputs.outputs[name],
        )
        .unwrap()
    };
    let coefficients = |matrix: &DCRTPolyMatrix| {
        (0..matrix.col_size())
            .flat_map(|column| matrix.entry(0, column).coeffs_biguints())
            .collect::<Vec<_>>()
    };
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    let random = |bound: usize| rand::random::<u32>() as usize % bound;
    for trial in 0..read("TRIALS", 3usize) {
        let attribute = (0..class.attribute_length())
            .map(|_| random(class.alphabet_size()))
            .collect::<Vec<_>>();
        // A random maximal-length program with `F(x) = 0` whose tables are not constant (a
        // constant table carries no noise forward; with one state, intermediate tables are).
        let entries = class.state_width() * class.alphabet_size();
        let steps = class.max_steps();
        let reads = (0..steps).map(|_| random(class.attribute_length())).collect::<Vec<_>>();
        let mut tables = (0..steps)
            .map(|step| {
                let bound = if step + 1 == steps { 2 } else { class.state_width() };
                let mut table = (0..entries).map(|_| random(bound)).collect::<Vec<_>>();
                if bound > 1 && table.iter().all(|value| *value == table[0]) {
                    table[1] = (table[0] + 1) % bound;
                }
                table
            })
            .collect::<Vec<_>>();
        let mut state = 0;
        for (read, table) in reads.iter().zip(&tables).take(steps - 1) {
            state = table[state + class.state_width() * attribute[*read]];
        }
        let reached = state + class.state_width() * attribute[reads[steps - 1]];
        tables[steps - 1][reached] = 0;
        if tables[steps - 1].iter().all(|value| *value == 0) {
            tables[steps - 1][(reached + 1) % entries] = 1;
        }
        let predicate = BranchingProgram::new(class, reads, tables).unwrap();
        let message = (0..dimension).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
        let (ciphertext, secret) =
            encrypt_with_secret(&public_key, &attribute, &message, &mut execution, &mut store)
                .unwrap();
        let key = keygen(&public_key, &master, &predicate, &mut execution, &mut store).unwrap();
        let evaluated =
            evaluate(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();

        // The evaluation error of the output row, opened with the secret.
        let ring = public_key.ring().unwrap();
        let columns = parameters.digits();
        let mut runtime = BTreeMap::new();
        secret.bind_runtime(&mut runtime);
        for port in [4, 5] {
            runtime.insert(
                format!("{OUTPUT_ROW}.{port}"),
                evaluated.outputs[&format!("output-0.{port}")].clone(),
            );
        }
        let s_bar = secret.row().0.clone();
        let output = key.output().unwrap();
        let evaluation = key.evaluation().production().clone();
        let output_key = |port: usize| {
            ring.artifact_input(
                evaluation.clone(),
                format!("public-key/{output:?}.{port}"),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
        let output_row = Mat::concat(
            ConcatAxis::Columns,
            [4, 5].map(|port| ring.input(format!("{OUTPUT_ROW}.{port}"), (1, columns))).to_vec(),
        );
        let output_first = Mat::concat(ConcatAxis::Columns, vec![output_key(0), output_key(1)]);
        let output_second = Mat::concat(ConcatAxis::Columns, vec![output_key(2), output_key(3)]);
        let error_row = output_row - output_second - s_bar * output_first;
        let graph =
            DslContext::new("mxx-pe/test-evaluation-error").output("e_F", error_row).unwrap();
        let manifests = BTreeMap::from([(
            key.evaluation().production().clone(),
            key.evaluation().manifest().clone(),
        )]);
        let outputs = api::execute_graph(
            StageGraph::new(graph.build().unwrap(), runtime, manifests),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let (e_max, e_rms) = statistics(&q, &coefficients(&download(&execution, &outputs, "e_F")));

        // The phase error.
        let (phase, runtime, manifests) =
            phase_graph(&public_key, &key, &ciphertext, &evaluated).unwrap();
        let graph = DslContext::new("mxx-pe/test-phase").output("phase", phase).unwrap();
        let outputs = api::execute_graph(
            StageGraph::new(graph.build().unwrap(), runtime, manifests),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let half = &q / 2u8;
        let phase = download(&execution, &outputs, "phase")
            .entry(0, 0)
            .coeffs_biguints()
            .into_iter()
            .zip(&message)
            .map(|(value, bit)| (value + &q - if *bit { half.clone() } else { 0u8.into() }) % &q)
            .collect::<Vec<_>>();
        let (p_max, p_rms) = statistics(&q, &phase);
        eprintln!(
            "trial {trial}: e_F max {e_max:.1} rms {e_rms:.1}; phase max {p_max:.1} rms {p_rms:.1}"
        );
    }
}
