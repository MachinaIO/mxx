#[cfg(feature = "gpu")]
fn bench_gpu_matrix_mul() {
    use mxx_backends::{
        GpuRuntime, RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly_gpu::gpu_backend,
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::{
            PolyParams,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{
        ParamEnv,
        ring::{RingExpr, RingRef},
        types::{ConcreteMatrixType, ConcreteWireType},
    };
    use std::{collections::BTreeMap, hint::black_box, sync::Arc, time::Instant};
    use tracing::info;

    let _ = tracing_subscriber::fmt::try_init();
    // Keep parameters aligned with the CPU benchmark for a fair comparison.
    let params = DCRTPolyParams::new(16384, 15, 24, 12, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        params.ring_dimension(),
        params.moduli().to_vec(),
        params.base_bits(),
        None,
    );
    let crt_moduli = params.moduli().iter().copied().map(Into::into).collect::<Vec<_>>();
    let ring = Ring::from_crt_moduli(crt_moduli.clone(), params.ring_dimension());
    let graph = DslContext::new("bench-gpu-matrix-mul")
        .output("product", ring.input("left", (1, 30)) * ring.input("right", (30, 120)))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let resolved =
        RingRef::new(RingExpr::Explicit { crt_moduli, ring_dimension: params.ring_dimension() })
            .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
    let sampler = DCRTPolyUniformSampler::new();
    let upload = |rows: usize, columns: usize| {
        let matrix = sampler.sample_uniform(&params, rows, columns, DistType::FinRingDist);
        RuntimeValue::gpu_matrix(
            ConcreteWireType::Matrix(ConcreteMatrixType { ring: resolved.clone(), rows, columns }),
            Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &matrix)),
        )
        .unwrap()
    };
    let inputs =
        BTreeMap::from([("left".into(), upload(1, 30)), ("right".into(), upload(30, 120))]);

    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params.clone()])).unwrap();
    let start = Instant::now();
    let mut plan = runtime.plan(graph, &inputs).unwrap();
    info!("GPU matrix mul plan: {:?}", start.elapsed());
    let mut store = MemoryArtifactStore::default();
    // The first execute binds the plan; the timed one measures steady-state replay.
    drop(runtime.execute_with_artifacts(&mut plan, inputs.clone(), &mut store, [0; 32]).unwrap());
    let start = Instant::now();
    let result = runtime.execute_with_artifacts(&mut plan, inputs, &mut store, [1; 32]).unwrap();
    let elapsed = start.elapsed();
    black_box(result.output("product").unwrap());

    info!("GPU matrix mul execute: {:?}", elapsed);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_matrix_mul();
}
