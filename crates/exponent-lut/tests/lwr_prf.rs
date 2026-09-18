#![cfg(feature = "gpu")]

//! The application-facing RNS Sparse-LWR PRF lifecycle.

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_exponent_lut::{
    ExpLayout, api,
    noise_refresh::prf::{PrfLabel, SparseLwrPrfProfile, SparseLwrPrfProgram},
    program::{AttributeKnowledge, FamilyRange, InputBinding, WireId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
        RnsBggNoiseValue, RnsBggParameters, RnsEncryptBuild, RnsNoiseBuild, RnsOnlineBuild,
        RnsPreprocessBuild, verify_output,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv};
use mxx_primitives::{
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use mxx_runtime::{
    ExecutionConfig,
    artifact::FileArtifactStore,
    backend::poly::gpu::{GpuDcrtBackend, gpu_backend_on},
};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

const HASH_KEY: &[u8] = b"lwr-prf-rns-hash-key";

struct Fixture {
    parameters: RnsBggParameters,
    profile: SparseLwrPrfProfile,
    program: SparseLwrPrfProgram,
    payload: mxx_exponent_lut::rns_bgg::RnsBggPayloadSecret,
    mask: mxx_exponent_lut::rns_bgg::RnsBggMaskSecret,
    rhs: BTreeMap<mxx_exponent_lut::program::RhsId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    lhs: BTreeMap<WireId, InputBinding<DCRTPoly, Vec<DCRTPoly>>>,
    noise: RnsBggNoiseBindings,
    d_exponents: Vec<usize>,
}

fn parameters() -> (RnsBggParameters, DCRTPolyParams) {
    let source = DCRTPolyParams::new(32, 3, 10, 5, None, None);
    let primes = source.to_crt().0;
    let middle = DCRTPolyParams::new(32, 2, 10, 5, Some(primes[..2].to_vec()), None);
    let terminal = DCRTPolyParams::new(32, 1, 10, 5, Some(primes[..1].to_vec()), None);
    let sigma = BigDecimal::from(1u8);
    let parameters = RnsBggParameters::new(vec![source.clone(), middle, terminal], sigma, 1)
        .expect("valid RNS modulus chain");
    (parameters, source)
}

fn fixture() -> Fixture {
    let (rns_parameters, poly_parameters) = parameters();
    let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
        .expect("valid sparse-LWR profile");
    let labels = [PrfLabel::from_components(&[b"lwr-rns-label"]).expect("canonical label")];
    let program = SparseLwrPrfProgram::new_structural(profile, 2, 2, HASH_KEY, &labels)
        .expect("structural sparse-LWR program");
    let payload = RnsEncryptBuild::<GpuDcrtBackend>::new(&rns_parameters, [7; 32])
        .sample_payload_secret()
        .expect("payload secret");
    let mask = RnsEncryptBuild::<GpuDcrtBackend>::new(&rns_parameters, [7; 32])
        .sample_mask_secret()
        .expect("mask secret");
    let zero = DCRTPoly::const_zero(&poly_parameters);
    let d_exponents = (0..profile.transform_column_count().expect("D width"))
        .map(|index| index % profile.q().expect("Q exponent"))
        .collect::<Vec<_>>();
    let d_values = d_exponents
        .iter()
        .map(|exponent| {
            let dimension = poly_parameters.ring_dimension() as usize;
            let rotation = (exponent * profile.layout.stride()) % (2 * dimension);
            let monomial = DCRTPoly::const_rotate_poly(&poly_parameters, rotation % dimension);
            if rotation >= dimension { -monomial } else { monomial }
        })
        .collect::<Vec<_>>();

    let mut lhs = BTreeMap::new();
    let input_projection = program.program().encryption_inputs();
    for (wire, declaration) in input_projection.inputs() {
        let value = match (declaration.knowledge(), declaration.shape()) {
            (AttributeKnowledge::PrivatePayload, InputBinding::Scalar(())) => {
                InputBinding::Scalar(zero.clone())
            }
            (AttributeKnowledge::PublicPayload, InputBinding::Scalar(())) => {
                InputBinding::Scalar(zero.clone())
            }
            (AttributeKnowledge::PublicPayload, InputBinding::Family(_)) => {
                InputBinding::Family(d_values.clone())
            }
            (knowledge, shape) => panic!("unexpected input declaration: {knowledge:?} {shape:?}"),
        };
        lhs.insert(*wire, value);
    }

    let selector = program.selection_families()[0].selector;
    let mut selector_values = vec![zero; program.block_count() * program.bucket_width()];
    for bucket in 0..program.block_count() {
        selector_values[bucket * program.bucket_width()] =
            DCRTPoly::const_rotate_poly(&poly_parameters, 0);
    }
    let rhs = BTreeMap::from([(selector, InputBinding::Family(selector_values))]);

    let error_bound = hard_cutoff_from_sigma_bound(&BigDecimal::from(1u8));
    let noise_inputs = program
        .program()
        .encryption_inputs()
        .inputs()
        .iter()
        .map(|(_, declaration)| {
            match (declaration.knowledge(), declaration.shape()) {
                (AttributeKnowledge::PrivatePayload, InputBinding::Scalar(())) |
                (AttributeKnowledge::PublicPayload, InputBinding::Scalar(())) |
                (AttributeKnowledge::PublicPayload, InputBinding::Family(_)) => {}
                (knowledge, shape) => {
                    panic!("unexpected noise declaration: {knowledge:?} {shape:?}")
                }
            }
            RnsBggNoiseValue::public(error_bound.clone(), &[BigInt::from(1u8)])
        })
        .collect::<Vec<_>>();
    let noise_rhs = program
        .selection_families()
        .first()
        .is_some()
        .then_some(RnsBggNoiseRhs {
            payload_l1: BigUint::from(1u8),
            payload_l2_squared: BigUint::from(1u8),
        })
        .into_iter()
        .collect::<Vec<_>>();
    let mut input_bindings = BTreeMap::new();
    for ((wire, declaration), value) in
        program.program().encryption_inputs().inputs().iter().zip(noise_inputs)
    {
        let binding = match declaration.shape() {
            InputBinding::Scalar(()) => InputBinding::Scalar(value),
            InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                element: value,
                range: FamilyRange::full(count.clone()).expect("valid input family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        };
        input_bindings.insert(*wire, binding);
    }
    let rhs_binding = program.selection_families().first().map(|family| {
        (
            family.selector,
            InputBinding::Family(RnsBggNoiseFamily {
                element: noise_rhs[0].clone(),
                range: FamilyRange::full(IntExpr::constant(family.capacity))
                    .expect("valid selector family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        )
    });
    let noise = RnsBggNoiseBindings::new(input_bindings, rhs_binding.into_iter().collect())
        .expect("declaration-order noise bindings");

    Fixture {
        parameters: rns_parameters,
        profile,
        program,
        payload,
        mask,
        rhs,
        lhs,
        noise,
        d_exponents,
    }
}

fn gpu_backend(parameters: &RnsBggParameters, devices: Vec<i32>) -> GpuDcrtBackend {
    let levels = parameters.levels().iter().map(|parameters| {
        mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.to_crt().0,
            parameters.base_bits(),
            None,
        )
    });
    gpu_backend_on(levels, devices)
}

fn sequential_dt_oracle(program: &SparseLwrPrfProgram, d_exponents: &[usize]) -> usize {
    let q = program.profile().q().expect("Q exponent");
    (0..program.block_count())
        .map(|bucket| {
            let transform = &program.label_transforms()[bucket * program.bucket_width()];
            transform
                .bits()
                .iter()
                .enumerate()
                .filter_map(|(column, bit)| (*bit == 1).then_some(d_exponents[column]))
                .sum::<usize>()
        })
        .sum::<usize>() %
        q
}

#[test]
#[sequential(dcrt_runtime)]
fn lwr_prf_end_to_end() {
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping Sparse-LWR round trip: no CUDA device detected");
        return;
    }
    let fixture = fixture();
    let Fixture { parameters, profile, program, payload, mask, rhs, lhs, noise, d_exponents } =
        fixture;
    let workload = RnsBggDerivedWorkload::from_program(program.program())
        .expect("Sparse-LWR workload derivation");
    let trace = api::noise_simulate(
        program.program(),
        RnsNoiseBuild::new(&parameters, program.program(), &workload, &noise)
            .expect("RNS noise lowering state"),
        |_, outputs| Ok(outputs),
    )
    .expect("RNS noise simulation");
    assert!(!trace.wires.is_empty(), "noise simulation must visit the PRF gates");

    let mut backend = gpu_backend(&parameters, devices.clone());
    let estimate_levels = parameters.levels().iter().map(|level| {
        GpuDCRTPolyParams::new(level.ring_dimension(), level.to_crt().0, level.base_bits(), None)
    });
    let estimate_backends = devices
        .iter()
        .copied()
        .map(|device| (gpu_backend_on(estimate_levels.clone(), [device]), device))
        .collect();
    let estimator = Rc::new(RefCell::new(GpuNodeMeasurementBackend::new(
        estimate_backends,
        MeasurementHarnessConfig {
            warm_up_iterations: 1,
            measured_iterations: 1,
            ..MeasurementHarnessConfig::default()
        },
    )));
    let observed = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let observe = |name: &'static str, graph: &mxx_ir_core::ValidatedGraph| {
        estimate(graph, &mut *estimator.borrow_mut()).expect("stage estimate");
        observed.borrow_mut().push(name);
    };
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = FileArtifactStore::new(directory.path()).expect("artifact store");
    let preprocess_encrypt = directory.path().join("preprocess-encrypt.json");
    let preprocess_online = directory.path().join("preprocess-online.json");
    let encryption = directory.path().join("encryption.json");
    let (preprocess, online) = api::preprocess(
        program.program(),
        &ParamEnv::default(),
        &rhs,
        &payload,
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [7; 32]),
        &mut backend,
        &mut store,
        [&preprocess_encrypt, &preprocess_online],
        ExecutionConfig::default(),
        |_, outputs| Ok(outputs),
        |graph| {
            observe("preprocess", graph);
            Ok(())
        },
    )
    .expect("RNS preprocessing");
    let encrypted = api::encrypt(
        &program.program().encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        &preprocess,
        &payload,
        &mask,
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [7; 32]),
        &mut backend,
        api::ArtifactDestination { store: &mut store, checkpoint: &encryption },
        ExecutionConfig::default(),
        |_| Ok(()),
        |graph| {
            observe("encrypt", graph);
            Ok(())
        },
    )
    .expect("RNS encryption");
    let mut result = api::online_eval(
        program.program(),
        &ParamEnv::default(),
        &online,
        &encrypted,
        HASH_KEY,
        RnsOnlineBuild::new(&parameters, [7; 32]),
        &mut backend,
        &mut store,
        ExecutionConfig::default(),
        |_, outputs| Ok(outputs),
        |graph| {
            observe("online_eval", graph);
            Ok(())
        },
    )
    .expect("RNS online evaluation");
    estimator.borrow_mut().measure_collected().expect("measure all three production graphs");
    assert_eq!(
        observed.borrow().as_slice(),
        ["preprocess", "encrypt", "online_eval"],
        "observers must see the exact three production graphs"
    );
    let expected_exponent = sequential_dt_oracle(&program, &d_exponents);
    let expected_readout =
        profile.canonical_readout().expect("canonical readout")[expected_exponent];
    let source_parameters = parameters.level(0).expect("LWR source parameters").clone();
    let expected_readout =
        u32::try_from(expected_readout).expect("canonical readout fits DCRT coefficient");
    let expected = DCRTPoly::from_u32s(
        &source_parameters,
        &vec![expected_readout; source_parameters.ring_dimension() as usize],
    );
    verify_output(
        program.program(),
        &mut result,
        0,
        &expected,
        &parameters,
        &payload,
        &mask,
        &mut backend,
        &mut store,
        ExecutionConfig::default(),
    )
    .expect("LWR application verification");
    assert!(!trace.outputs.is_empty(), "noise trace must finalize declared output");
    assert!(!result.outputs.is_empty(), "online evaluation must produce output");
}
