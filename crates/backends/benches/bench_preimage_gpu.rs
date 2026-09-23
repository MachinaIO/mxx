#[cfg(feature = "gpu")]
fn bench_gpu_preimage() {
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
        ParamEnv, RealExpr,
        ring::{RingExpr, RingRef},
        types::{ConcreteMatrixType, ConcreteWireType},
    };
    use num_bigint::BigUint;
    use std::{collections::BTreeMap, hint::black_box, sync::Arc, time::Instant};
    use tracing::info;

    const SIGMA: f64 = 4.578;
    const TRAPDOOR_SIZE: usize = 1;
    const TARGET_COLS: usize = 50;
    const PREIMAGE_BOUND_BITS: usize = 48;

    let _ = tracing_subscriber::fmt::try_init();
    // Keep parameters aligned with the CPU benchmark for a fair comparison.
    let params = DCRTPolyParams::new(16384, 10, 24, 12, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        params.ring_dimension(),
        params.moduli().to_vec(),
        params.base_bits(),
        None,
    );
    let digits = params.modulus_digits();
    let crt_moduli = params.moduli().iter().copied().map(Into::into).collect::<Vec<_>>();
    let ring = Ring::from_crt_moduli(crt_moduli.clone(), params.ring_dimension());
    let trapdoor = ring.sample_trapdoor(
        TRAPDOOR_SIZE,
        RealExpr::from_f64_exact(SIGMA).unwrap(),
        1u64 << params.base_bits(),
        digits,
        BigUint::from(1u8) << PREIMAGE_BOUND_BITS,
    );
    let target = ring.input("target", (TRAPDOOR_SIZE, TARGET_COLS));
    let preimage = trapdoor.sample_preimage(target, (TRAPDOOR_SIZE * (digits + 2), TARGET_COLS));
    let graph = DslContext::new("bench-gpu-preimage")
        .output("preimage", preimage)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let resolved =
        RingRef::new(RingExpr::Explicit { crt_moduli, ring_dimension: params.ring_dimension() })
            .resolve(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
    let target = DCRTPolyUniformSampler::new().sample_uniform(
        &params,
        TRAPDOOR_SIZE,
        TARGET_COLS,
        DistType::FinRingDist,
    );
    let inputs = BTreeMap::from([(
        "target".into(),
        RuntimeValue::gpu_matrix(
            ConcreteWireType::Matrix(ConcreteMatrixType {
                ring: resolved,
                rows: TRAPDOOR_SIZE,
                columns: TARGET_COLS,
            }),
            Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&gpu_params, &target)),
        )
        .unwrap(),
    )]);

    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params.clone()])).unwrap();
    let start = Instant::now();
    let mut plan = runtime.plan(graph, &inputs).unwrap();
    info!("GPU trapdoor+preimage plan: {:?}", start.elapsed());
    let mut store = MemoryArtifactStore::default();
    // The first execute binds the plan; the timed one measures steady-state replay.
    drop(runtime.execute(&mut plan, inputs.clone(), &mut store, rand::random()).unwrap());
    let start = Instant::now();
    let result = runtime.execute(&mut plan, inputs, &mut store, rand::random()).unwrap();
    let elapsed = start.elapsed();
    black_box(result.output("preimage").unwrap());

    info!("GPU trapdoor+preimage execute: {:?}", elapsed);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmark skipped (enable with --features gpu).");
}

#[cfg(feature = "gpu")]
fn main() {
    bench_gpu_preimage();
}
