//! GPU unit checks for RNS-BGG ring wiring and column actions.
//!
//! Device tests are opt-in (`--ignored`). Complete application round trips remain in the owning
//! application's integration tests.

use super::RnsBggParameters;
use mxx_backends::poly::{
    PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use num_bigint::BigUint;
use serial_test::serial as sequential;

fn gpu_parameters(level: &DCRTPolyParams, device: i32) -> GpuDCRTPolyParams {
    GpuDCRTPolyParams::new_with_gpu(
        level.ring_dimension(),
        level.to_crt().0,
        level.base_bits(),
        vec![device],
        None,
        None,
        None,
    )
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_rns_bgg_parameter_levels_preserve_ring_shapes() {
    let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
    let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
    let gpu = gpu_parameters(parameters.level(0).unwrap(), 0);
    assert_eq!(gpu.ring_dimension(), level.ring_dimension());
    assert_eq!(gpu.to_crt().0, level.to_crt().0);
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_rns_bgg_operations_per_drop_three_at_boundaries() {
    let pool = DCRTPolyParams::new(16, 4, 17, 4, None, None);
    let primes = pool.to_crt().0;
    let levels = [4usize, 3, 2, 1]
        .into_iter()
        .map(|depth| {
            let modulus = primes[..depth]
                .iter()
                .fold(BigUint::from(1u8), |product, prime| product * BigUint::from(*prime));
            pool.select_modulus(&modulus).expect("nested RNS modulus level")
        })
        .collect::<Vec<_>>();
    let parameters = RnsBggParameters::new(levels, 1, 3).expect("valid cadence");

    assert_eq!(parameters.operations_per_drop(), 3);
    // Logical levels 0..=2 share the first physical CRT depth.  The drop is
    // selected at the first operation in each subsequent cadence window.
    assert_eq!(super::planned_crt_depth(0, 3, 1, 4, 1).unwrap(), 4);
    assert_eq!(super::planned_crt_depth(2, 3, 1, 4, 1).unwrap(), 4);
    assert_eq!(super::planned_crt_depth(3, 3, 1, 4, 1).unwrap(), 3);
    assert_eq!(super::planned_crt_depth(5, 3, 1, 4, 1).unwrap(), 3);
    assert_eq!(super::planned_crt_depth(6, 3, 1, 4, 1).unwrap(), 2);
    assert_eq!(super::planned_crt_depth(9, 3, 1, 4, 1).unwrap(), 1);
    assert!(super::planned_crt_depth(0, 0, 1, 4, 1).is_err());
}

/// Column-local decomposition must preserve dense CRT arithmetic across a
/// modulus rebase, including every output column and multiple acted rows.
#[test]
#[ignore = "requires a supported GPU"]
#[sequential(dcrt_runtime)]
fn test_gpu_rns_bgg_column_decomposition_matches_dense_action() {
    use super::{compact_decompose_columns, multiply_rows_by_columns};
    use mxx_backends::{GpuRuntime, MemoryArtifactStore, RuntimeValue};
    use mxx_dsl::{DslContext, Mat};
    use mxx_ir_core::{ParamEnv, node::ConcatAxis};
    use std::collections::BTreeMap;
    let read = |name, default| std::env::var(name).map_or(default, |v| v.parse().unwrap());
    let pool = DCRTPolyParams::new(read("MXX_COLUMN_ACTION_TEST_N", 16), 2, 17, 4, None, None);
    let reduced = pool.select_modulus(&BigUint::from(pool.to_crt().0[0])).unwrap();
    let parameters = RnsBggParameters::new(vec![pool.clone(), reduced], 0, 1).unwrap();
    let ring = parameters.ring(0).unwrap();
    let width = read("MXX_COLUMN_ACTION_TEST_COLUMNS", 3) as usize;
    let hash = ring.bytes_input("hash", 32);
    let rows = (0..2)
        .map(|row| {
            let entry = |side| {
                ring.hash_matrix(
                    hash.clone(),
                    format!("lhs-{row}-{side}").into_bytes(),
                    (1, pool.modulus_digits()),
                )
            };
            (entry(0), entry(1))
        })
        .collect::<Vec<_>>();
    let inputs = BTreeMap::from([(
        "hash".to_owned(),
        RuntimeValue::Bytes(rand::random::<[u8; 32]>().as_slice().into()),
    )]);
    let mut runtime = GpuRuntime::new(mxx_backends::backend::poly_gpu::gpu_backend_on(
        parameters.levels().iter().map(|p| gpu_parameters(p, 0)),
        mxx_backends::poly::dcrt::gpu::detected_gpu_device_ids(),
    ))
    .unwrap();
    for source_level in [0, 1] {
        let source_ring = parameters.ring(source_level).unwrap();
        let entry = |r, c| {
            source_ring.hash_matrix(hash.clone(), format!("rhs-{r}-{c}").into_bytes(), (1, width))
        };
        let matrix = ((entry(0, 0), entry(0, 1)), (entry(1, 0), entry(1, 1)));
        let source = parameters.level(source_level).unwrap();
        let column = |top: Mat, bottom: Mat| {
            Mat::concat(ConcatAxis::Rows, vec![top, bottom])
                .decompose(1u64 << source.base_bits(), source.modulus_digits())
                .centered_rebase(&ring)
        };
        let dense = (
            column(matrix.0.0.clone(), matrix.1.0.clone()),
            column(matrix.0.1.clone(), matrix.1.1.clone()),
        );
        // Rows in the evaluation ring use the source gadget's digit width.
        let rows = rows
            .iter()
            .map(|(left, right)| {
                (
                    left.clone().slice(
                        None,
                        Some(mxx_ir_core::node::IndexRange {
                            start: 0usize.into(),
                            end: source.modulus_digits().into(),
                        }),
                    ),
                    right.clone().slice(
                        None,
                        Some(mxx_ir_core::node::IndexRange {
                            start: 0usize.into(),
                            end: source.modulus_digits().into(),
                        }),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected = rows
            .iter()
            .map(|(left, right)| {
                let lhs = Mat::concat(ConcatAxis::Columns, vec![left.clone(), right.clone()]);
                (lhs.clone().mul_small_rhs(dense.0.clone()), lhs.mul_small_rhs(dense.1.clone()))
            })
            .collect::<Vec<_>>();
        let actual = multiply_rows_by_columns(
            rows,
            compact_decompose_columns(&parameters, 0, source_level, matrix).unwrap(),
        );
        let build = |rows: Vec<super::RnsBggRow>| {
            let mut graph = DslContext::new("column-action");
            for (i, (left, right)) in rows.into_iter().enumerate() {
                graph = graph
                    .output(format!("{i}-left"), left)
                    .unwrap()
                    .output(format!("{i}-right"), right)
                    .unwrap();
            }
            graph.build().unwrap().validate(&ParamEnv::default()).unwrap()
        };
        let expected = mxx_backends::executor::execute(
            &build(expected),
            &mut mxx_backends::backend::poly::cpu_backend(parameters.levels().to_vec()),
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            mxx_backends::transcript::SamplingMode::Fresh,
            Default::default(),
        )
        .unwrap();
        let mut plan = runtime.plan(build(actual), &inputs).unwrap();
        let result = runtime.execute(&mut plan, inputs.clone()).unwrap();
        for (name, value) in expected.outputs {
            let RuntimeValue::Matrix(value) = value else { panic!("matrix") };
            let actual = runtime.download_matrix_output(&result.output(&name).unwrap()).unwrap();
            assert_eq!(
                &actual,
                value.as_cpu_full().unwrap(),
                "source level {source_level}, {name}"
            );
        }
    }
}
