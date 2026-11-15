//! The core lowering of the lattice DSL (`mxx_ir_core::lower`) computes what the CPU executor
//! computes, both by host execution of the core program and by running its RV64IM proof image on
//! the machine emulator.

use super::*;
use crate::{
    artifact::MemoryArtifactStore,
    backend::poly::cpu_backend,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_core::{
    expr::Bindings,
    flatten::{FlattenOptions, flatten},
    host,
    rv64::machine::{BASE, Environment, Machine, Outcome},
    validate::validate,
};
use mxx_ir_core::{
    IntExpr,
    lower::{int_words, lower},
    node::{ConcatAxis, IndexRange},
};
use mxx_lattice_dsl::{
    DslContext, GraphValue, Int, IntType, Mat, Ring, Subgraph, iterate, parallel, select,
};
use num_bigint::BigUint;
use std::sync::Arc;

/// The core encoding of a matrix: limb-major residues, entries row-major, coefficients in order.
fn encode(matrix: &DCRTPolyMatrix, moduli: &[u64]) -> Vec<u8> {
    let (rows, columns) = matrix.size();
    let entries: Vec<Vec<BigUint>> = (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (r, c)))
        .map(|(r, c)| matrix.entry(r, c).coeffs_biguints())
        .collect();
    let mut bytes = Vec::new();
    for q in moduli {
        for entry in &entries {
            for coefficient in entry {
                let residue = u64::try_from(coefficient % q).expect("a residue fits a word");
                bytes.extend_from_slice(&residue.to_le_bytes());
            }
        }
    }
    bytes
}

/// The core encoding of a CPU executor value whose integers have `words` words.
fn encode_value(value: &RuntimeValue, words: u32) -> Vec<u8> {
    match value {
        RuntimeValue::Matrix(matrix) => {
            let matrix = matrix.as_cpu_full().expect("CPU matrix");
            encode(matrix, &matrix.params().to_crt().0)
        }
        RuntimeValue::Int(value) => {
            let mut bytes = value.to_signed_bytes_le();
            assert!(bytes.len() <= 8 * words as usize, "{value} fits {words} words");
            let fill = if value.sign() == num_bigint::Sign::Minus { 0xff } else { 0 };
            bytes.resize(8 * words as usize, fill);
            bytes
        }
        RuntimeValue::Bool(bit) => u64::from(*bit).to_le_bytes().to_vec(),
        RuntimeValue::IndexedFamily { values, .. } => {
            values.iter().flat_map(|member| encode_value(member, words)).collect()
        }
        other => panic!("no core encoding for {other:?}"),
    }
}

struct Inputs(Vec<Vec<u8>>);

impl Environment for Inputs {
    fn query(&mut self, request: &[u8]) -> Vec<u8> {
        let (0, index) = request.split_first().expect("an input request") else {
            panic!("not an input request")
        };
        self.0[u64::from_le_bytes(index.try_into().unwrap()) as usize].clone()
    }
}

struct Setup {
    parameters: DCRTPolyParams,
    moduli: Vec<u64>,
    ring: Ring,
    rows: usize,
    columns: usize,
}

fn setup() -> Setup {
    let (ring_dimension, crt_depth, rows, columns) = crate::env::test_core_lowering_shape();
    let parameters = DCRTPolyParams::new(ring_dimension as u32, crt_depth, 20, 4, None, None);
    let moduli = parameters.to_crt().0;
    let ring = Ring::from_crt_moduli(
        moduli.iter().copied().map(IntExpr::from).collect(),
        parameters.ring_dimension(),
    );
    Setup { parameters, moduli, ring, rows, columns }
}

impl Setup {
    fn random(&self, rows: usize, columns: usize) -> DCRTPolyMatrix {
        use rand::Rng;
        let modulus: BigUint = (*self.parameters.modulus()).clone();
        let mut rng = rand::rng();
        let polys = (0..rows)
            .map(|_| {
                (0..columns)
                    .map(|_| {
                        let coefficients: Vec<BigUint> = (0..self.parameters.ring_dimension())
                            .map(|_| {
                                BigUint::from(rng.random::<u64>()) *
                                    BigUint::from(rng.random::<u64>()) %
                                    &modulus
                            })
                            .collect();
                        DCRTPoly::from_biguints(&self.parameters, &coefficients)
                    })
                    .collect()
            })
            .collect();
        DCRTPolyMatrix::from_poly_vec(&self.parameters, polys)
    }

    /// Runs `graph` on the CPU executor, lowers it, and checks that host execution and the proof
    /// image give the CPU executor's outputs.
    fn check(
        &self,
        graph: &ValidatedGraph,
        matrices: &[(&str, &DCRTPolyMatrix)],
        integers: &[(&str, i64)],
    ) {
        self.check_with_rings(graph, matrices, integers, &[], &[]);
    }

    /// [`Self::check`] for a graph that also reads byte strings and whose values may also lie in
    /// the rings of `rings`.
    fn check_with_rings(
        &self,
        graph: &ValidatedGraph,
        matrices: &[(&str, &DCRTPolyMatrix)],
        integers: &[(&str, i64)],
        bytes: &[(&str, Vec<u8>)],
        rings: &[DCRTPolyParams],
    ) {
        let mut cpu_inputs = BTreeMap::new();
        let mut core_inputs = BTreeMap::new();
        for (name, matrix) in matrices {
            cpu_inputs.insert(name.to_string(), RuntimeValue::matrix((*matrix).clone()));
            core_inputs.insert(name.to_string(), encode(matrix, &matrix.params().to_crt().0));
        }
        for (name, value) in bytes {
            cpu_inputs.insert(name.to_string(), RuntimeValue::Bytes(Arc::from(value.clone())));
            core_inputs.insert(name.to_string(), value.clone());
        }
        for (name, value) in integers {
            cpu_inputs.insert(name.to_string(), RuntimeValue::Int(BigInt::from(*value)));
            core_inputs
                .insert(name.to_string(), host::write_int(i128::from(*value), int_words(graph)));
        }
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            graph,
            &mut cpu_backend(std::iter::once(self.parameters.clone()).chain(rings.iter().cloned())),
            cpu_inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let program = validate(Arc::new(lower(graph).unwrap()), Bindings::new()).unwrap();
        let expected: Vec<Vec<u8>> = program
            .program
            .outputs
            .iter()
            .map(|(name, _)| encode_value(&result.outputs[name], int_words(graph)))
            .collect();
        let inputs: Vec<Vec<u8>> =
            program.program.inputs.iter().map(|(name, _)| core_inputs[name].clone()).collect();
        assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), expected);

        let image = flatten(&program, &FlattenOptions::default()).unwrap();
        let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
        let (outcome, steps) =
            machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
        let Outcome::Halted(payload) = outcome else {
            panic!("{outcome:?} at {:#x}", machine.pc() - BASE)
        };
        let mut rest = &payload[..];
        let mut outputs = Vec::new();
        while !rest.is_empty() {
            let len = u64::from_le_bytes(rest[..8].try_into().unwrap()) as usize;
            outputs.push(rest[8..8 + len].to_vec());
            rest = &rest[8 + len..];
        }
        assert_eq!(outputs, expected);
        assert!(
            u128::from(steps) <= image.step_bound,
            "{steps} steps over the bound {}",
            image.step_bound
        );
    }
}

#[test]
fn test_core_lowering_of_matrix_arithmetic_matches_cpu_executor() {
    let s = setup();
    let (a, b) = (s.ring.input("a", (s.rows, s.columns)), s.ring.input("b", (s.rows, s.columns)));
    let c = s.ring.input("c", (s.columns, 2));
    let v = s.ring.input("v", (1, s.columns));
    let graph = DslContext::new("core-lowering")
        .output("difference", &a - &b)
        .unwrap()
        .output("negated_sum", -&(&a + &b))
        .unwrap()
        .output("product", &a * &c)
        .unwrap()
        .output("scaled", s.ring.polynomial([IntExpr::from(7)]) * v.clone())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mb, mc, mv) = (
        s.random(s.rows, s.columns),
        s.random(s.rows, s.columns),
        s.random(s.columns, 2),
        s.random(1, s.columns),
    );
    s.check(&graph, &[("a", &ma), ("b", &mb), ("c", &mc), ("v", &mv)], &[]);
}

#[test]
fn test_core_lowering_of_loops_families_and_selection_matches_cpu_executor() {
    let s = setup();
    let members = 3usize;
    let context = DslContext::new("core-lowering-structure");
    let (a, b) = (s.ring.input("a", (s.rows, s.columns)), s.ring.input("b", (s.rows, s.columns)));
    let member: Int = context.input("member", IntType).unwrap();
    let selector: Int = context.input("selector", IntType).unwrap();
    let sums = parallel(members, |_| Ok(&a + &b)).unwrap();
    let differences = parallel(members, |m| Ok(&sums.at(m) - &a)).unwrap();
    let accumulated = iterate(2, a.clone(), |_, state| Ok(&state + &b)).unwrap();
    let chosen = select(selector, vec![a.clone(), b.clone()]).unwrap();
    let graph = context
        .output("static", differences.at(1))
        .unwrap()
        .output("dynamic", differences.at(member))
        .unwrap()
        .output("accumulated", accumulated)
        .unwrap()
        .output("chosen", chosen)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mb) = (s.random(s.rows, s.columns), s.random(s.rows, s.columns));
    s.check(
        &graph,
        &[("a", &ma), ("b", &mb)],
        &[
            ("member", rand::random_range(0..members as i64)),
            ("selector", rand::random_range(0..2)),
        ],
    );
}

#[test]
fn test_core_lowering_of_shapes_ring_maps_and_integers_matches_cpu_executor() {
    let s = setup();
    assert!(s.columns >= 2, "the slice needs two columns");
    let context = DslContext::new("core-lowering-shapes");
    let (a, b) = (s.ring.input("a", (s.rows, s.columns)), s.ring.input("b", (s.rows, s.columns)));
    let shift: Int = context.input("shift", IntType).unwrap();
    let range = |start: usize, end: usize| IndexRange { start: start.into(), end: end.into() };
    let first_row_tail = a.clone().slice(Some(range(0, 1)), Some(range(1, s.columns)));
    let chosen = select(shift.clone().less(1).to_int(), vec![a.clone(), b.clone()]).unwrap();
    let double = Subgraph::<Mat, Mat>::define("double", a.schema(), |x| Ok(&x + &x)).unwrap();
    let graph = context
        .output("transposed", a.clone().transpose())
        .unwrap()
        .output("slice", first_row_tail)
        .unwrap()
        .output("rows", Mat::concat(ConcatAxis::Rows, vec![a.clone(), b.clone()]))
        .unwrap()
        .output("columns", Mat::concat(ConcatAxis::Columns, vec![a.clone(), b.clone()]))
        .unwrap()
        .output("automorphism", a.clone().ring_automorphism(3))
        .unwrap()
        .output("monomial", b.clone().multiply_monomial(shift.clone() + 2))
        .unwrap()
        .output(
            "quotient_monomial",
            b.clone().multiply_monomial(shift.clone().div(3) + shift.rem(5)),
        )
        .unwrap()
        .output("doubled", double.call(a.clone()).unwrap())
        .unwrap()
        .output("chosen", chosen)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mb) = (s.random(s.rows, s.columns), s.random(s.rows, s.columns));
    s.check(&graph, &[("a", &ma), ("b", &mb)], &[("shift", rand::random_range(-20..20))]);
}

#[test]
fn test_core_lowering_of_constant_matrices_matches_cpu_executor() {
    let s = setup();
    let a = s.ring.input("a", (2, 2));
    let graph = DslContext::new("core-lowering-constants")
        .output("zero_plus", &a + &s.ring.zero((2, 2)))
        .unwrap()
        .output("identity_product", &a * &s.ring.identity(2))
        .unwrap()
        .output(
            "gadget",
            s.ring.gadget(2, 1u64 << s.parameters.base_bits(), s.parameters.modulus_digits()),
        )
        .unwrap()
        .output(
            "polynomial",
            s.ring.polynomial([IntExpr::from(3), IntExpr::from(-5), IntExpr::from(7)]),
        )
        .unwrap()
        .output("all_coefficients", s.ring.all_coefficients((1, 2), -9))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let ma = s.random(2, 2);
    s.check(&graph, &[("a", &ma)], &[]);
}

#[test]
fn test_core_lowering_of_uniform_samples_is_in_range_and_crt_consistent() {
    let s = setup();
    let (minimum, maximum) = (-1i64, 1i64);
    let graph = DslContext::new("core-lowering-samples")
        .output("residues", s.ring.uniform_residue((s.rows, s.columns)))
        .unwrap()
        .output("interval", s.ring.uniform_interval((s.rows, s.columns), minimum, maximum))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    let words = s.rows * s.columns * s.parameters.ring_dimension() as usize;
    let check = |outputs: &[Vec<u8>]| {
        let limbs = |bytes: &[u8]| -> Vec<Vec<u64>> {
            bytes
                .chunks(8 * words)
                .map(|limb| {
                    limb.chunks(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect()
                })
                .collect()
        };
        // Outputs are in name order: `interval`, then `residues`.
        for (residues, q) in limbs(&outputs[1]).iter().zip(&s.moduli) {
            assert!(residues.iter().all(|r| r < q));
        }
        // Each interval coefficient is one integer in [minimum, maximum] in every limb.
        let interval = limbs(&outputs[0]);
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..words {
            let r0 = interval[0][i] as i128;
            let q0 = s.moduli[0] as i128;
            let value = if r0 <= maximum as i128 { r0 } else { r0 - q0 };
            assert!((minimum as i128..=maximum as i128).contains(&value), "{value}");
            for (limb, q) in interval.iter().zip(&s.moduli) {
                assert_eq!(limb[i] as i128, value.rem_euclid(*q as i128));
            }
            seen.insert(value);
        }
        assert!(seen.len() > 1, "the samples vary");
    };
    let outputs = host::run(&program, Vec::new(), None).unwrap();
    check(&outputs);
    let image = flatten(&program, &FlattenOptions::default()).unwrap();
    let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
    let (outcome, steps) =
        machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(Vec::new()));
    let Outcome::Halted(payload) = outcome else {
        panic!("{outcome:?} at {:#x}", machine.pc() - BASE)
    };
    let mut rest = &payload[..];
    let mut image_outputs = Vec::new();
    while !rest.is_empty() {
        let len = u64::from_le_bytes(rest[..8].try_into().unwrap()) as usize;
        image_outputs.push(rest[8..8 + len].to_vec());
        rest = &rest[8 + len..];
    }
    check(&image_outputs);
    assert!(u128::from(steps) <= image.step_bound);
}

/// Ring-LWE encryption of one bit with a uniform `a` and ternary secret and error, decrypted by
/// threshold decoding: the CPU executor, host execution, and the proof image all decrypt the
/// message.
#[test]
fn test_core_lowering_of_rlwe_encryption_decrypts_its_message() {
    let s = setup();
    let context = DslContext::new("core-lowering-rlwe");
    let message: Int = context.input("message", IntType).unwrap();
    let a = s.ring.uniform_residue((1, 1));
    let secret = s.ring.uniform_interval((1, 1), -1, 1);
    let error = s.ring.uniform_interval((1, 1), -1, 1);
    let delta = s.ring.polynomial([s.ring.modulus().floor_div(2)]);
    let plaintext = message.lift_to_constant_polynomial(s.ring.matrix_type((1, 1)));
    let b = &a * &secret + error + &delta * &plaintext;
    let decrypted = (b - &a * &secret).threshold_decode_bools(2, 1).remove(0);
    let graph = context
        .output("decrypted", decrypted)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    for bit in [0i64, 1] {
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &graph,
            &mut cpu_backend([s.parameters.clone()]),
            BTreeMap::from([("message".into(), RuntimeValue::Int(BigInt::from(bit)))]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        assert!(matches!(result.outputs["decrypted"], RuntimeValue::Bool(b) if b == (bit == 1)));
        let expected = vec![(bit as u64).to_le_bytes().to_vec()];
        let inputs = vec![host::write_int(i128::from(bit), int_words(&graph))];
        assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), expected);
        let image = flatten(&program, &FlattenOptions::default()).unwrap();
        let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
        let (outcome, steps) =
            machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
        let Outcome::Halted(payload) = outcome else {
            panic!("{outcome:?} at {:#x}", machine.pc() - BASE)
        };
        assert_eq!(payload[8..].to_vec(), expected[0]);
        assert!(u128::from(steps) <= image.step_bound);
    }
}

/// Gaussian samples are integers within the cutoff, the same in every limb, with the mean and
/// variance of the discrete Gaussian.
#[test]
fn test_core_lowering_of_gaussian_samples_has_the_gaussian_moments() {
    let s = setup();
    let (sigma, bound) = (4i64, 26i64);
    let (rows, columns) = (8, 32);
    let graph = DslContext::new("core-lowering-gaussian")
        .output("e", s.ring.gaussian((rows, columns), sigma, bound))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    let words = rows * columns * s.parameters.ring_dimension() as usize;
    let check = |bytes: &[u8]| {
        let limbs: Vec<Vec<u64>> = bytes
            .chunks(8 * words)
            .map(|limb| limb.chunks(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect())
            .collect();
        let values: Vec<i128> = (0..words)
            .map(|i| {
                let (r0, q0) = (limbs[0][i] as i128, s.moduli[0] as i128);
                let value = if r0 <= bound as i128 { r0 } else { r0 - q0 };
                assert!(value.abs() <= bound as i128, "{value}");
                for (limb, q) in limbs.iter().zip(&s.moduli) {
                    assert_eq!(limb[i] as i128, value.rem_euclid(*q as i128));
                }
                value
            })
            .collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<i128>() as f64 / n;
        let variance = values.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let expected = (sigma * sigma) as f64;
        assert!(mean.abs() < 0.6, "mean {mean}");
        assert!((0.75 * expected..1.25 * expected).contains(&variance), "variance {variance}");
    };
    check(&host::run(&program, Vec::new(), None).unwrap()[0]);
    let image = flatten(&program, &FlattenOptions::default()).unwrap();
    let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
    let (outcome, steps) =
        machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(Vec::new()));
    let Outcome::Halted(payload) = outcome else {
        panic!("{outcome:?} at {:#x}", machine.pc() - BASE)
    };
    check(&payload[8..]);
    assert!(u128::from(steps) <= image.step_bound);
}

/// The README's Ring-LWE example: `a` derived from a seed by the hash sampler (bit for bit the
/// CPU executor's), a Gaussian secret and error, and decryption by threshold decoding.
#[test]
fn test_core_lowering_of_the_readme_rlwe_example() {
    use mxx_lattice_dsl::HashTag;
    let s = setup();
    let context = DslContext::new("core-lowering-readme-rlwe");
    let seed = s.ring.bytes_input("seed", 32);
    let message: Int = context.input("message", IntType).unwrap();
    let a = s.ring.hash_matrix(seed, HashTag::from(b"rlwe-example/a".as_slice()), (1, 1));
    let secret = s.ring.gaussian((1, 1), 4, 26);
    let error = s.ring.gaussian((1, 1), 4, 26);
    let delta = s.ring.polynomial([s.ring.modulus().floor_div(2)]);
    let plaintext = message.lift_to_constant_polynomial(s.ring.matrix_type((1, 1)));
    let b = &a * &secret + error + &delta * &plaintext;
    let decrypted = (b - &a * &secret).threshold_decode_bools(2, 1).remove(0);
    let graph = context
        .output("a", a)
        .unwrap()
        .output("decrypted", decrypted)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    for bit in [0i64, 1] {
        let seed: Vec<u8> = (0..32).map(|_| rand::random()).collect();
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &graph,
            &mut cpu_backend([s.parameters.clone()]),
            BTreeMap::from([
                ("seed".into(), RuntimeValue::Bytes(Arc::from(seed.clone()))),
                ("message".into(), RuntimeValue::Int(BigInt::from(bit))),
            ]),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::Matrix(cpu_a) = &result.outputs["a"] else { panic!("matrix output") };
        let expected = vec![
            encode(cpu_a.as_cpu_full().unwrap(), &s.moduli),
            (bit as u64).to_le_bytes().to_vec(),
        ];
        let inputs: Vec<Vec<u8>> = program
            .program
            .inputs
            .iter()
            .map(|(name, _)| {
                if name == "seed" {
                    seed.clone()
                } else {
                    host::write_int(i128::from(bit), int_words(&graph))
                }
            })
            .collect();
        assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), expected);
        let image = flatten(&program, &FlattenOptions::default()).unwrap();
        let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
        let (outcome, steps) =
            machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
        let Outcome::Halted(payload) = outcome else {
            panic!("{outcome:?} at {:#x}", machine.pc() - BASE)
        };
        let mut rest = &payload[..];
        let mut outputs = Vec::new();
        while !rest.is_empty() {
            let len = u64::from_le_bytes(rest[..8].try_into().unwrap()) as usize;
            outputs.push(rest[8..8 + len].to_vec());
            rest = &rest[8 + len..];
        }
        assert_eq!(outputs, expected);
        assert!(u128::from(steps) <= image.step_bound);
    }
}

/// The gadget decomposition and its product with the gadget, which reconstructs the input.
#[test]
fn test_core_lowering_of_gadget_decomposition_matches_cpu_executor() {
    let s = setup();
    let (base, digits) = (1u64 << s.parameters.base_bits(), s.parameters.modulus_digits());
    let a = s.ring.input("a", (2, s.columns));
    let decomposed = a.clone().decompose(base, digits);
    let graph = DslContext::new("core-lowering-gadget")
        .output("reconstructed", s.ring.gadget(2, base, digits).mul_small_rhs(decomposed))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let ma = s.random(2, s.columns);
    s.check(&graph, &[("a", &ma)], &[]);
}

/// The digits themselves of the small per-limb decomposition and of decompositions that drop
/// trailing towers, read out through a product with the identity.
#[test]
fn test_core_lowering_of_small_and_dropped_gadget_decompositions_matches_cpu_executor() {
    let s = setup();
    let base = 1u64 << s.parameters.base_bits();
    let per_tower = s.parameters.crt_bits().div_ceil(s.parameters.base_bits() as usize);
    let a = s.ring.input("a", (2, s.columns));
    let digits = |count: usize| {
        let identity = s.ring.identity(2 * count);
        move |preimage| identity.clone().mul_small_rhs(preimage)
    };
    let small_gadget = s.ring.constant(
        (2, 2 * per_tower),
        mxx_ir_core::node::ConstantMatrix::Gadget { base: base.into(), small: true },
    );
    let mut context = DslContext::new("core-lowering-gadget-variants")
        .output("small", digits(per_tower)(a.clone().small_decompose(base, per_tower)))
        .unwrap()
        .output(
            "small_reconstructed",
            small_gadget.mul_small_rhs(a.clone().small_decompose(base, per_tower)),
        )
        .unwrap()
        .output(
            "rotation",
            s.ring.constant(
                (1, 1),
                mxx_ir_core::node::ConstantMatrix::Rotation { exponent: 3.into() },
            ),
        )
        .unwrap()
        .output("true", mxx_lattice_dsl::Bool::constant(true))
        .unwrap();
    for retained in 1..s.moduli.len() {
        let count = retained * per_tower;
        context = context
            .output(format!("retained_{retained}"), digits(count)(a.clone().decompose(base, count)))
            .unwrap()
            .output(
                format!("reconstructed_{retained}"),
                s.ring.gadget(2, base, count).mul_small_rhs(a.clone().decompose(base, count)),
            )
            .unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let ma = s.random(2, s.columns);
    s.check(&graph, &[("a", &ma)], &[]);
}

/// Kronecker products, block diagonals, and the fused multiply-accumulate.
#[test]
fn test_core_lowering_of_tensor_block_diagonal_and_accumulate_matches_cpu_executor() {
    let s = setup();
    let a = s.ring.input("a", (2, 2));
    let b = s.ring.input("b", (1, 2));
    let c = s.ring.input("c", (2, 2));
    let bias = s.ring.input("bias", (2, 2));
    let accumulated = Mat::multi_row_gemm_accumulate(
        vec![(IntExpr::from(3), a.clone(), c.clone()), (IntExpr::from(-2), c.clone(), a.clone())],
        Some(bias),
    );
    let graph = DslContext::new("core-lowering-tensor")
        .output("tensor", a.clone().tensor(b.clone()))
        .unwrap()
        .output("block_diagonal", b.clone().block_diagonal(3))
        .unwrap()
        .output("accumulated", accumulated)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mb, mc, mbias) = (s.random(2, 2), s.random(1, 2), s.random(2, 2), s.random(2, 2));
    s.check(&graph, &[("a", &ma), ("b", &mb), ("c", &mc), ("bias", &mbias)], &[]);
}

/// Modulus switch, reduction, centered rebase, and centered round division: exact CRT maps.
#[test]
fn test_core_lowering_of_ring_conversions_matches_cpu_executor() {
    let s = setup();
    assert!(s.moduli.len() >= 2, "the conversions need at least two CRT primes");
    let n = s.parameters.ring_dimension();
    let crt_bits = s.parameters.crt_bits();
    let subset = |moduli: Vec<u64>| {
        let parameters =
            DCRTPolyParams::new(n, moduli.len(), crt_bits, 4, Some(moduli.clone()), None);
        let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), n);
        (parameters, ring)
    };
    let (prefix, prefix_ring) = subset(s.moduli[..s.moduli.len() - 1].to_vec());
    let (last, last_ring) = subset(vec![*s.moduli.last().unwrap()]);
    // A basis disjoint from the source, of wider primes.
    let other = DCRTPolyParams::new(n, s.moduli.len() + 1, crt_bits + 2, 4, None, None);
    let other_ring =
        Ring::from_crt_moduli(other.to_crt().0.into_iter().map(IntExpr::from).collect(), n);
    let large_divisor = i64::try_from(s.moduli[0] * 7 + 3).unwrap();
    let a = s.ring.input("a", (s.rows, s.columns));
    let graph = DslContext::new("core-lowering-ring-conversions")
        .output("switched", a.clone().modulus_switch(&prefix_ring))
        .unwrap()
        .output("switched_last", a.clone().modulus_switch(&last_ring))
        .unwrap()
        .output("reduced", a.clone().reduce_modulus(&last_ring))
        .unwrap()
        .output("rebased", a.clone().centered_rebase(&other_ring))
        .unwrap()
        .output("rebased_down", a.clone().centered_rebase(&prefix_ring))
        .unwrap()
        .output("divided", a.clone().centered_round_divide(17))
        .unwrap()
        .output("divided_large", a.clone().centered_round_divide(large_divisor))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let ma = s.random(s.rows, s.columns);
    s.check_with_rings(&graph, &[("a", &ma)], &[], &[], &[prefix, last, other]);
}

/// Exact block modulus switching, and OpenFHE's fast RNS ModUp and ModDown.
#[test]
fn test_core_lowering_of_block_and_rns_conversions_matches_cpu_executor() {
    let s = setup();
    assert!(s.moduli.len() >= 2, "the conversions need at least two CRT primes");
    let n = s.parameters.ring_dimension();
    let crt_bits = s.parameters.crt_bits();
    let basis = |moduli: Vec<u64>| {
        let parameters =
            DCRTPolyParams::new(n, moduli.len(), crt_bits, 4, Some(moduli.clone()), None);
        let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), n);
        (parameters, ring)
    };
    let (prefix, prefix_ring) = basis(s.moduli[..s.moduli.len() - 1].to_vec());
    // The source basis and two more primes of the same width.
    let extra: Vec<u64> = DCRTPolyParams::new(n, s.moduli.len() + 2, crt_bits, 4, None, None)
        .to_crt()
        .0
        .into_iter()
        .filter(|q| !s.moduli.contains(q))
        .take(2)
        .collect();
    let (extended, extended_ring) = basis(s.moduli.iter().copied().chain(extra).collect());
    let a = s.ring.input("a", (s.rows, s.columns));
    let graph = DslContext::new("core-lowering-rns-conversions")
        .output("block_switched", a.clone().block_mod_switch(&prefix_ring, 17))
        .unwrap()
        .output("up_single", a.clone().rns_mod_up(&extended_ring, 1, false))
        .unwrap()
        .output("up_pairs_normalized", a.clone().rns_mod_up(&extended_ring, 2, true))
        .unwrap()
        .output("down", a.clone().rns_mod_down(&prefix_ring, 65537))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let ma = s.random(s.rows, s.columns);
    s.check_with_rings(&graph, &[("a", &ma)], &[], &[], &[prefix, extended]);
}

/// Coefficients and OpenFHE's evaluation slots as integers and back, one coefficient, integer
/// matrix-vector products, and integer arithmetic beyond one word.
#[test]
fn test_core_lowering_of_polynomial_values_and_wide_integers_matches_cpu_executor() {
    let s = setup();
    let n = s.parameters.ring_dimension() as usize;
    let a = s.ring.input("a", (1, 1));
    let b = s.ring.input("b", (1, 1));
    let context = DslContext::new("core-lowering-polynomial-values");
    let x: Int = context.input("x", IntType).unwrap();
    let (coefficients, evaluations) = (a.coefficients(), a.evaluations());
    let other = b.coefficients();
    let c = a.clone().extract_coefficient(n - 1);
    // Products of two coefficients exceed one word; signs exercise the Euclidean operations.
    let wide = c.clone().mul(other.at(0)).sub(x.clone().mul(Int::constant(1_000_003)));
    let graph = context
        .output("coefficients", coefficients.clone())
        .unwrap()
        .output("evaluations", evaluations.clone())
        .unwrap()
        .output("from_coefficients", s.ring.from_coefficients(&coefficients))
        .unwrap()
        .output("from_evaluations", s.ring.from_evaluations(&evaluations))
        .unwrap()
        .output("coefficient", c.clone())
        .unwrap()
        .output("product", coefficients.matrix_vector_product(&other))
        .unwrap()
        .output("transposed_product", coefficients.vector_matrix_product(&other))
        .unwrap()
        .output("wide", wide.clone())
        .unwrap()
        .output("quotient", wide.clone().div(x.clone()))
        .unwrap()
        .output("remainder", wide.clone().rem(x.clone()))
        .unwrap()
        .output("less", wide.clone().less(x.clone()))
        .unwrap()
        .output("high_bit", wide.clone().bit(70).unwrap())
        .unwrap()
        .output("sign_bit", x.clone().bit(200).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mb) = (s.random(1, 1), s.random(1, 1));
    for x in [-7, 13] {
        s.check(&graph, &[("a", &ma), ("b", &mb)], &[("x", x)]);
    }
}

/// CRT recomposition of two decoded levels over different rings into a third ring, and canonical
/// coefficient bits packed back into a polynomial.
#[test]
fn test_core_lowering_of_crt_recomposition_and_bit_packing_matches_cpu_executor() {
    let s = setup();
    assert!(s.moduli.len() >= 2, "the levels need at least two CRT primes");
    let n = s.parameters.ring_dimension();
    let crt_bits = s.parameters.crt_bits();
    let prefix = DCRTPolyParams::new(
        n,
        s.moduli.len() - 1,
        crt_bits,
        4,
        Some(s.moduli[..s.moduli.len() - 1].to_vec()),
        None,
    );
    let prefix_ring =
        Ring::from_crt_moduli(prefix.to_crt().0.into_iter().map(IntExpr::from).collect(), n);
    let other = DCRTPolyParams::new(n, s.moduli.len() + 1, crt_bits + 2, 4, None, None);
    let other_ring =
        Ring::from_crt_moduli(other.to_crt().0.into_iter().map(IntExpr::from).collect(), n);
    let a = s.ring.input("a", (1, s.columns));
    let b = prefix_ring.input("b", (1, s.columns));
    let recomposed = Mat::crt_recompose(
        vec![a.clone(), b],
        vec![IntExpr::from(17), IntExpr::from(257)],
        vec![IntExpr::from(257 * 2), IntExpr::from(17 * 8)],
        &other_ring,
    );
    // The coefficient bits of a scalar polynomial, little-endian, and the polynomial again.
    let bits = (*s.parameters.modulus()).bits() as usize;
    let c = s.ring.input("c", (1, 1));
    let coefficients = c.coefficients();
    let flat = parallel(n as usize * bits, |index| {
        coefficients.at(index.clone().div(bits as i64)).bit(index.rem(bits as i64))
    })
    .unwrap();
    let graph = DslContext::new("core-lowering-crt-recompose")
        .output("recomposed", recomposed)
        .unwrap()
        .output("packed", s.ring.pack_polynomial_coefficients(flat, bits))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let (ma, mc) = (s.random(1, s.columns), s.random(1, 1));
    let mb = {
        let values = s.random(1, s.columns);
        let polys = (0..s.columns)
            .map(|j| {
                DCRTPoly::from_biguints(
                    &prefix,
                    &values
                        .entry(0, j)
                        .coeffs_biguints()
                        .iter()
                        .map(|v| v % (*prefix.modulus()).clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        DCRTPolyMatrix::from_poly_vec(&prefix, vec![polys])
    };
    s.check_with_rings(&graph, &[("a", &ma), ("b", &mb), ("c", &mc)], &[], &[], &[prefix, other]);
}

/// Hash integer families, one and two words wide, bit for bit the CPU executor's, and SHA-256 bit
/// columns.
#[test]
fn test_core_lowering_of_hash_integer_families_matches_cpu_executor() {
    use mxx_lattice_dsl::HashTag;
    let s = setup();
    let context = DslContext::new("core-lowering-hash-integers");
    let key = s.ring.bytes_input("key", 32);
    let a = s.ring.input("a", (1, 1));
    let (packed, counts) = context.sha256_bit_columns([7; 32], 5, 4, 3);
    let narrow =
        context.hash_int_family(key.clone(), HashTag::from(b"narrow".as_slice()), 9, 1 << 5);
    let wide =
        context.hash_int_family(key, HashTag::from(b"wide".as_slice()), 5, BigInt::from(1) << 70);
    let graph = context
        .output("narrow", narrow)
        .unwrap()
        .output("wide", wide)
        .unwrap()
        .output("a", a)
        .unwrap()
        .output("packed", packed)
        .unwrap()
        .output("counts", counts)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let key: Vec<u8> = (0..32).map(|_| rand::random()).collect();
    let ma = s.random(1, 1);
    s.check_with_rings(&graph, &[("a", &ma)], &[], &[("key", key)], &[]);
}

/// Hash tags that depend on loop indices and on runtime integers, built at run time bit for bit
/// as the CPU executor frames them.
#[test]
fn test_core_lowering_of_runtime_hash_tags_matches_cpu_executor() {
    use mxx_lattice_dsl::HashTag;
    let s = setup();
    let context = DslContext::new("core-lowering-runtime-hash-tags");
    let key = s.ring.bytes_input("key", 32);
    let x: Int = context.input("x", IntType).unwrap();
    let matrices = parallel(3, |index| {
        let mut tag = HashTag::from(b"matrix".as_slice());
        tag.push_integer(index.clone().sub(1));
        tag.push_decimal(index.clone().mul(-7))?;
        tag.push_u64_le(index.clone());
        tag.push_integer(x.clone());
        tag.push_u64_le(x.clone().mul(x.clone()));
        Ok(s.ring.hash_matrix(key.clone(), tag, (1, 2)))
    })
    .unwrap();
    let mut tag = HashTag::from(b"integers".as_slice());
    tag.push_integer(x.clone().mul(Int::constant(BigInt::from(1) << 80)));
    tag.push_integer(Int::constant(0));
    let integers = context.hash_int_family(key, tag, 4, 1 << 12);
    let graph = context
        .output("matrix_0", matrices.at(0))
        .unwrap()
        .output("matrix_2", matrices.at(2))
        .unwrap()
        .output("integers", integers)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let key: Vec<u8> = (0..32).map(|_| rand::random()).collect();
    for x in [-3, 0, 5] {
        s.check_with_rings(&graph, &[], &[("x", x)], &[("key", key.clone())], &[]);
    }
}

/// Hash samples decomposed into gadget digits, over all towers and per limb, read out through a
/// product with the identity.
#[test]
fn test_core_lowering_of_decomposed_hash_samples_matches_cpu_executor() {
    use mxx_lattice_dsl::HashTag;
    let s = setup();
    let base = 1u64 << s.parameters.base_bits();
    let per_tower = s.parameters.crt_bits().div_ceil(s.parameters.base_bits() as usize);
    let digits = per_tower * s.moduli.len();
    let context = DslContext::new("core-lowering-decomposed-hash");
    let key = s.ring.bytes_input("key", 32);
    let decomposed = s.ring.hash_decomposed(
        key.clone(),
        HashTag::from(b"decomposed".as_slice()),
        (2 * digits, 2),
        base,
        digits,
    );
    let small = s.ring.hash_small_decomposed(
        key,
        HashTag::from(b"small".as_slice()),
        (2 * per_tower, 2),
        base,
        per_tower,
    );
    let graph = context
        .output("decomposed", s.ring.identity(2 * digits).mul_small_rhs(decomposed))
        .unwrap()
        .output("small", s.ring.identity(2 * per_tower).mul_small_rhs(small))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let key: Vec<u8> = (0..32).map(|_| rand::random()).collect();
    s.check_with_rings(&graph, &[], &[], &[("key", key)], &[]);
}

/// Bindings of a parallel loop (`k = 2 i + 1`, depending on its index) and of a subgraph call
/// (`k = 7`), both shadowing the graph parameter `k = 100`. The DSL emits no bindings, so they are
/// set in the serialized graph.
#[test]
fn test_core_lowering_of_loop_and_call_bindings_matches_cpu_executor() {
    let s = setup();
    let k = || s.ring.polynomial([IntExpr::Var("k".into())]);
    let a = s.ring.input("a", (1, s.columns));
    let scaled = parallel(3, |_| Ok(k() * a.clone())).unwrap();
    let scale = Subgraph::<Mat, Mat>::define("scale", a.schema(), |x| Ok(k() * x)).unwrap();
    let graph = DslContext::new("core-lowering-bindings")
        .int_parameter("k")
        .output("member_0", scaled.at(0))
        .unwrap()
        .output("member_2", scaled.at(2))
        .unwrap()
        .output("called", scale.call(a.clone()).unwrap())
        .unwrap()
        .output("root", k() * a.clone())
        .unwrap()
        .build()
        .unwrap()
        .graph;
    let mut json = serde_json::to_value(&graph).unwrap();
    let index_binding =
        serde_json::to_value(IntExpr::LoopIndex(0) * IntExpr::from(2) + IntExpr::from(1)).unwrap();
    let constant_binding = serde_json::to_value(IntExpr::from(7)).unwrap();
    let mut bound = 0;
    fn visit(
        value: &mut serde_json::Value,
        loop_binding: &serde_json::Value,
        call_binding: &serde_json::Value,
        bound: &mut usize,
    ) {
        if let Some(object) = value.as_object_mut() {
            let tag = object.get("tag").and_then(|t| t.as_str()).map(str::to_owned);
            if let (Some(tag), Some(inner)) = (tag, object.get_mut("value")) &&
                let Some(bindings) = inner.get_mut("bindings")
            {
                let binding = match tag.as_str() {
                    "ParallelLoop" => Some(loop_binding),
                    "SubgraphCall" => Some(call_binding),
                    _ => None,
                };
                if let Some(binding) = binding {
                    *bindings = serde_json::json!([["k", binding]]);
                    *bound += 1;
                }
            }
            for child in object.values_mut() {
                visit(child, loop_binding, call_binding, bound);
            }
        } else if let Some(array) = value.as_array_mut() {
            for child in array {
                visit(child, loop_binding, call_binding, bound);
            }
        }
    }
    visit(&mut json, &index_binding, &constant_binding, &mut bound);
    assert_eq!(bound, 2, "one loop and one call are bound");
    let graph: mxx_ir_core::Graph = serde_json::from_value(json).unwrap();
    let mut env = ParamEnv::default();
    env.integers.insert("k".into(), BigInt::from(100));
    let graph = mxx_ir_core::validate(&graph, &env).unwrap();
    let ma = s.random(1, s.columns);
    s.check(&graph, &[("a", &ma)], &[]);
}

/// A computational (MP12) trapdoor: the lowered sample's public matrix `A` and secret `R`, `E`
/// (its limbs hold `A ‖ R ‖ E`) satisfy `A [R; E; I] = G` in every tower, `R` and `E` are small,
/// and `TrapdoorPublic` is `A`.
#[test]
fn test_core_lowering_of_trapdoor_samples_satisfies_the_gadget_relation() {
    let s = setup();
    let rows = s.rows.min(2);
    let base_bits = s.parameters.base_bits();
    let digits = s.parameters.modulus_digits();
    let sigma = 4i64;
    let context = DslContext::new("core-lowering-trapdoor");
    let trapdoor = s.ring.sample_trapdoor(rows, sigma, 1i64 << base_bits, digits, 1i64 << 20);
    let public = trapdoor.public_matrix();
    let graph = context
        .output("trapdoor", trapdoor)
        .unwrap()
        .output("public", public)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    flatten(&program, &FlattenOptions::default()).expect("a proof image");
    let outputs = host::run(&program, Vec::new(), None).unwrap();
    let n = s.parameters.ring_dimension() as usize;
    let k = rows * digits;
    let width = 2 * rows + k;
    let (a_words, r_words) = (rows * width * n, rows * k * n);
    let limb_words = a_words + 2 * r_words;
    let words = |bytes: &[u8]| -> Vec<u64> {
        bytes.chunks(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect()
    };
    let named = |name: &str| {
        let at = program.program.outputs.iter().position(|(n, _)| n == name).expect("an output");
        words(&outputs[at])
    };
    // The DSL outputs a trapdoor as its public matrix and the trapdoor value.
    let (trapdoor_words, public_words) = (named("trapdoor.1"), named("public"));
    let bound = 13 * sigma as u64;
    // The gadget matrix as the backends define it (digits per tower of the CRT basis).
    let gadget = words(&encode(
        &<DCRTPolyMatrix as crate::matrix::PolyMatrix>::gadget_matrix(&s.parameters, rows, None),
        &s.moduli,
    ));
    for (l, q) in s.moduli.iter().enumerate() {
        let limb = &trapdoor_words[l * limb_words..(l + 1) * limb_words];
        let (a, rest) = limb.split_at(a_words);
        let (r, e) = rest.split_at(r_words);
        assert_eq!(a, &public_words[l * a_words..(l + 1) * a_words], "TrapdoorPublic is A");
        for &c in r.iter().chain(e) {
            assert!(c <= bound || q - c <= bound, "a small secret coefficient");
        }
        // Entry (i, j) of a matrix of `columns` columns: its `n` coefficients.
        let entry = |m: &[u64], columns: usize, i: usize, j: usize| -> Vec<u64> {
            m[(i * columns + j) * n..(i * columns + j + 1) * n].to_vec()
        };
        let multiply = |x: &[u64], y: &[u64]| -> Vec<u64> {
            let mut out = vec![0u128; n];
            let qq = u128::from(*q);
            for (i, &xi) in x.iter().enumerate() {
                for (j, &yj) in y.iter().enumerate() {
                    let p = u128::from(xi) * u128::from(yj) % qq;
                    if i + j < n {
                        out[i + j] = (out[i + j] + p) % qq;
                    } else {
                        out[i + j - n] = (out[i + j - n] + qq - p) % qq;
                    }
                }
            }
            out.into_iter().map(|v| v as u64).collect()
        };
        // Row t of [R; E; I], column c.
        let t = |row: usize, c: usize| -> Vec<u64> {
            if row < rows {
                entry(r, k, row, c)
            } else if row < 2 * rows {
                entry(e, k, row - rows, c)
            } else {
                let mut unit = vec![0u64; n];
                unit[0] = u64::from(row - 2 * rows == c);
                unit
            }
        };
        for i in 0..rows {
            for c in 0..k {
                let mut sum = vec![0u64; n];
                for row in 0..width {
                    let product = multiply(&entry(a, width, i, row), &t(row, c));
                    for (s, p) in sum.iter_mut().zip(product) {
                        *s = ((u128::from(*s) + u128::from(p)) % u128::from(*q)) as u64;
                    }
                }
                let expected = entry(&gadget[l * r_words..(l + 1) * r_words], k, i, c);
                assert_eq!(sum, expected, "A [R; E; I] = G at ({i}, {c}) in tower {l}");
            }
        }
    }
}

/// The gadget trapdoor's public matrix is the gadget matrix of the backends.
#[test]
fn test_core_lowering_of_the_gadget_trapdoor_is_the_gadget_matrix() {
    let s = setup();
    let rows = s.rows.min(2);
    let (base_bits, digits) = (s.parameters.base_bits(), s.parameters.modulus_digits());
    let trapdoor = s.ring.gadget_trapdoor(rows, 1i64 << base_bits, digits);
    let graph = DslContext::new("core-lowering-gadget-trapdoor")
        .output("public", trapdoor.public_matrix())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    let outputs = host::run(&program, Vec::new(), None).unwrap();
    let gadget =
        <DCRTPolyMatrix as crate::matrix::PolyMatrix>::gadget_matrix(&s.parameters, rows, None);
    assert_eq!(outputs[0], encode(&gadget, &s.moduli));
}

/// A preimage under the gadget trapdoor is the target's gadget decomposition: its digits match
/// the CPU executor's, read out through a product with the identity, and the gadget matrix maps
/// them back to the target.
#[test]
fn test_core_lowering_of_gadget_trapdoor_preimages_matches_cpu_executor() {
    let s = setup();
    let rows = s.rows.min(2);
    let (base, digits) = (1u64 << s.parameters.base_bits(), s.parameters.modulus_digits());
    let trapdoor = s.ring.gadget_trapdoor(rows, base, digits);
    let a = s.ring.input("a", (rows, s.columns));
    let preimage = trapdoor.sample_preimage(a, (rows * digits, s.columns));
    let graph = DslContext::new("core-lowering-gadget-preimage")
        .output("digits", s.ring.identity(rows * digits).mul_small_rhs(preimage.clone()))
        .unwrap()
        .output("reconstructed", s.ring.gadget(rows, base, digits).mul_small_rhs(preimage))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let ma = s.random(rows, s.columns);
    s.check(&graph, &[("a", &ma)], &[]);
}

/// A preimage under a sampled (computational) trapdoor maps back to its target on the CPU
/// executor, the host, and the emulator, and the core's preimage is within the trapdoor's
/// coefficient bound.
#[test]
fn test_core_lowering_of_trapdoor_preimages_maps_back_to_the_target() {
    let s = setup();
    let rows = s.rows.min(2);
    let (base_bits, digits) = (s.parameters.base_bits(), s.parameters.modulus_digits());
    let bound = 1i64 << 22;
    let build = |name: &str, output: &str| {
        let trapdoor = s.ring.sample_trapdoor(rows, 4i64, 1i64 << base_bits, digits, bound);
        let public = trapdoor.public_matrix();
        let a = s.ring.input("a", (rows, s.columns));
        let preimage = trapdoor.sample_preimage(a, (rows * (digits + 2), s.columns));
        let value = if output == "reconstructed" {
            public.mul_small_rhs(preimage)
        } else {
            s.ring.identity(rows * (digits + 2)).mul_small_rhs(preimage)
        };
        DslContext::new(name)
            .output(output, value)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap()
    };
    let ma = s.random(rows, s.columns);
    s.check(&build("core-lowering-trapdoor-preimage", "reconstructed"), &[("a", &ma)], &[]);
    let graph = build("core-lowering-trapdoor-preimage-bound", "preimage");
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    assert_run_generates(&program);
    let outputs = host::run(&program, vec![encode(&ma, &s.moduli)], None).unwrap();
    let limb = outputs[0].len() / 8 / s.moduli.len();
    for (l, q) in s.moduli.iter().enumerate() {
        for word in outputs[0][8 * l * limb..8 * (l + 1) * limb].chunks(8) {
            let c = u64::from_le_bytes(word.try_into().unwrap());
            assert!(c.min(q - c) <= bound as u64, "coefficient {c} of tower {l} above {bound}");
        }
    }
}

/// A preimage under a statistical trapdoor maps back to its target on the CPU executor, the host,
/// and the emulator, and the core's preimage is within the trapdoor's coefficient bound.
#[test]
fn test_core_lowering_of_statistical_trapdoor_preimages_maps_back_to_the_target() {
    use mxx_ir_core::types::TrapdoorInstantiation;
    let s = setup();
    let digits = s.parameters.modulus_digits();
    let m_bar = 2 + s.rows;
    let bound = 1i64 << 30;
    let build = |name: &str, output: &str| {
        let trapdoor = s.ring.sample_trapdoor_with_width(
            1,
            TrapdoorInstantiation::Statistical { m_bar },
            4i64,
            1i64 << s.parameters.base_bits(),
            digits,
            None,
            bound,
        );
        let public = trapdoor.public_matrix();
        let a = s.ring.input("a", (1, s.columns));
        let preimage = trapdoor.sample_preimage(a, (m_bar + digits, s.columns));
        let value = if output == "reconstructed" {
            public.mul_small_rhs(preimage)
        } else {
            s.ring.identity(m_bar + digits).mul_small_rhs(preimage)
        };
        DslContext::new(name)
            .output(output, value)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap()
    };
    let ma = s.random(1, s.columns);
    s.check(&build("core-lowering-statistical-preimage", "reconstructed"), &[("a", &ma)], &[]);
    let graph = build("core-lowering-statistical-preimage-bound", "preimage");
    let program = validate(Arc::new(lower(&graph).unwrap()), Bindings::new()).unwrap();
    assert_run_generates(&program);
    let outputs = host::run(&program, vec![encode(&ma, &s.moduli)], None).unwrap();
    let limb = outputs[0].len() / 8 / s.moduli.len();
    for (l, q) in s.moduli.iter().enumerate() {
        for word in outputs[0][8 * l * limb..8 * (l + 1) * limb].chunks(8) {
            let c = u64::from_le_bytes(word.try_into().unwrap());
            assert!(c.min(q - c) <= bound as u64, "coefficient {c} of tower {l} above {bound}");
        }
    }
}

/// The run of `program`'s image (its retry loops with their coins included) is generated, with
/// placeholder Lean leaves.
fn assert_run_generates(program: &mxx_core::validate::ValidatedProgram) {
    let image = flatten(program, &FlattenOptions::default()).unwrap();
    fn leaf_names(events: &[mxx_core::flatten::GlueEvent], out: &mut Vec<String>) {
        for event in events {
            match event {
                mxx_core::flatten::GlueEvent::Leaf { name, .. } => out.push(name.clone()),
                mxx_core::flatten::GlueEvent::Loop { body, .. } => leaf_names(body, out),
                _ => {}
            }
        }
    }
    let mut names = Vec::new();
    leaf_names(&image.trace.as_ref().expect("a glue trace").0, &mut names);
    let leaves = names
        .into_iter()
        .map(|name| {
            let lean = mxx_core::compose::LeanLeaf {
                module: "M".into(),
                code_at: "At".into(),
                spec: "spec".into(),
            };
            (name, lean)
        })
        .collect();
    mxx_core::compose::run_module("G", "run", &image, &leaves).expect("a generated run");
}
