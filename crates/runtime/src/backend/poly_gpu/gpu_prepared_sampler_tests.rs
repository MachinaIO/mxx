use super::*;
use crate::{
    MemoryArtifactStore, RuntimeValue,
    backend::Backend,
    executor::{ExecutionConfig, execute_with_config},
    transcript::SamplingMode,
};
use mxx_dsl::{DslContext, Ring};
use mxx_ir_core::expr::ParamEnv;
use mxx_primitives::{
    poly::dcrt::params::DCRTPolyParams,
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};

#[test]
#[serial_test::serial(gpu_context)]
fn test_gpu_prepared_sampler_graph_preserves_relation_and_secret() {
    let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
        .map(|value| value.parse::<u32>().unwrap())
        .unwrap_or(32);
    let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(1);
    let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
    let cpu = DCRTPolyParams::new(n, 2, 17, 2, None, None);
    let params =
        GpuDCRTPolyParams::new_with_gpu(n, cpu.to_crt().0, 2, vec![device], Some(1), None, None);
    let digits = params.modulus_digits();
    let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
    let trapdoor = ring.sample_trapdoor(1, 5, 4, digits, 100_000_000);
    let target = ring.input("target", (1, columns));
    let preimage = trapdoor.sample_preimage(target, (digits + 2, columns));
    let check = preimage.mul_small_rhs(trapdoor.public_matrix());
    let graph = DslContext::new("prepared-sampler-graph")
        .output("check", check)
        .unwrap()
        .private_trapdoor_output("secret", trapdoor)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
    let source =
        DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
    let source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &source);
    let expected = source.to_compact_bytes();
    let inputs = BTreeMap::from([(
        "target".to_owned(),
        RuntimeValue::matrix(GpuFleetMatrix::from_matrix(source)),
    )]);
    let config = ExecutionConfig::default();
    backend.warm_up_prepared_graph(&graph, &inputs, &config).unwrap();
    let mut store = MemoryArtifactStore::default();
    for _ in 0..3 {
        #[cfg(feature = "gpu-instrumentation")]
        {
            mxx_primitives::poly::dcrt::gpu::gpu_test_reset_work_counters();
            mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        }
        let submitted = execute_with_config(
            &graph,
            &mut backend,
            inputs.clone(),
            &mut store,
            SamplingMode::Fresh,
            config.clone(),
        );
        #[cfg(feature = "gpu-instrumentation")]
        {
            mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(false);
            let (events, streams, validations, allocations, launches, measurements) =
                mxx_primitives::poly::dcrt::gpu::gpu_test_work_counters();
            assert_eq!((events, streams, validations, allocations, measurements), (0, 0, 0, 0, 0));
            assert!(launches > 0);
        }
        let mut result = submitted.unwrap();
        let RuntimeValue::Trapdoor { secret: Some(secret), public, .. } =
            result.materialize_output("secret", &mut backend, &mut store).unwrap()
        else {
            panic!("secret trapdoor result expected")
        };
        assert_eq!(secret.values.len(), 1);
        assert_eq!(public.size(), (1, digits + 2));
        let RuntimeValue::Matrix(actual) =
            result.materialize_output("check", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result expected")
        };
        assert_eq!(backend.matrix_to_bytes(actual).unwrap(), expected);
    }
}
