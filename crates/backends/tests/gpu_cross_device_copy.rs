#![cfg(feature = "gpu")]

//! Copies between devices, as a column-sharded product makes them: the
//! input is copied to every device and each device's output columns are
//! gathered back. On two physical GPUs without peer access (e.g. a PCIe
//! pair reporting `NS` in `nvidia-smi topo -p2p r`) the copies are staged
//! through host memory. On one GPU, run with `MXX_GPU_LOGICAL_DEVICES=0,0`
//! (and `MXX_GPU_HOST_STAGED_COPIES=1` to force the staged path).

use mxx_backends::{
    GpuRuntime, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_ir_core::{
    ParamEnv,
    node::ConcatAxis,
    ring::{RingExpr, RingRef},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use num_bigint::BigUint;
use std::{collections::BTreeMap, sync::Arc};

/// Plans and runs `a * (a * [a, a])` for a `size x size` input with its
/// output columns sharded over every device, and checks it against the CPU.
fn sharded_product(
    label: &str,
    ring_dimension: u32,
    crt_depth: usize,
    crt_bits: usize,
    size: usize,
) {
    let devices = detected_gpu_device_ids().len();
    if devices < 2 {
        eprintln!("{label}: skipped, needs two devices (or MXX_GPU_LOGICAL_DEVICES=0,0)");
        return;
    }
    let parameters = DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, 4, None, None);
    let entry = |seed: usize| {
        let coefficients = (0..ring_dimension as usize)
            .map(|index| BigUint::from((seed * 7919 + index * 104_729) as u64 % 1_000_003))
            .collect::<Vec<_>>();
        DCRTPoly::from_biguints(&parameters, &coefficients)
    };
    let input = DCRTPolyMatrix::from_poly_vec(
        &parameters,
        (0..size).map(|row| (0..size).map(|column| entry(row * size + column)).collect()).collect(),
    );
    let gpu_parameters = GpuDCRTPolyParams::new(
        parameters.ring_dimension(),
        parameters.moduli().to_vec(),
        parameters.base_bits(),
        None,
    );
    let moduli = parameters.moduli().iter().copied().map(Into::into).collect::<Vec<_>>();
    let ring = Ring::from_crt_moduli(moduli.clone(), parameters.ring_dimension());
    let source = ring.input("source", (size, size));
    let wide = Mat::concat(ConcatAxis::Columns, vec![source.clone(), source.clone()]);
    let graph = DslContext::new(label)
        .output("result", source.clone() * (source * wide))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let input_ring = RingRef::new(RingExpr::Explicit {
        crt_moduli: moduli,
        ring_dimension: parameters.ring_dimension(),
    })
    .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
    .unwrap();
    let value = RuntimeValue::gpu_matrix(
        ConcreteWireType::Matrix(ConcreteMatrixType {
            ring: input_ring,
            rows: size,
            columns: size,
        }),
        Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_parameters, &input)),
    )
    .unwrap();
    let bindings = BTreeMap::from([("source".to_owned(), value)]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_parameters])).unwrap();
    let mut plan = runtime.plan(graph, &bindings).unwrap();
    let active = plan
        .plan()
        .nodes
        .iter()
        .map(|node| node.columns_per_job.iter().filter(|width| **width > 0).count())
        .max()
        .unwrap_or(0);
    assert_eq!(active, devices, "{label}: the product shards over every device");
    let expected = input.clone() * (input.clone() * input.concat_columns(&[&input]));
    // A replay rebinds the staged copies' device addresses.
    for replay in 0..2u8 {
        let result = runtime
            .execute_with_artifacts(
                &mut plan,
                bindings.clone(),
                &mut MemoryArtifactStore::default(),
                [replay; 32],
            )
            .unwrap();
        let actual = runtime.download_matrix_output(&result.output("result").unwrap()).unwrap();
        assert_eq!(actual.size(), expected.size(), "{label}: shape");
        for row in 0..actual.size().0 {
            for column in 0..actual.size().1 {
                assert_eq!(
                    actual.entry(row, column).coeffs(),
                    expected.entry(row, column).coeffs(),
                    "{label}: replay {replay}, coefficients at ({row}, {column})",
                );
            }
        }
    }
}

#[test]
#[serial_test::serial]
fn small_sharded_product_copies_between_devices() {
    sharded_product("cross-device-copy-small", 8, 2, 20, 2);
}

/// Copies of a few MB, as in the failing NAND preprocessing.
#[test]
#[serial_test::serial]
fn large_sharded_product_copies_between_devices() {
    sharded_product("cross-device-copy-large", 4096, 4, 40, 4);
}
