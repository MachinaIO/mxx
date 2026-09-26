//! GPU execution of Exponent-LUT stages through the direct GPU runtime.
//!
//! Each stage graph is planned once with the artifact store (so artifact
//! imports get fixed destinations), executed once, and its outputs are copied
//! into value-owned storage before the plan is dropped.

use crate::api::{ApiError, StageExecution, StageOutputs};
use mxx_backends::{
    ExecutionError, GpuExecutionPlan, GpuRuntime, RuntimeValue, SessionStore,
    matrix::dcrt_poly::DCRTPolyMatrix,
};
use mxx_ir_core::{ValidatedGraph, artifact::ProductionId};
use std::collections::BTreeMap;
use tracing::info;

fn gpu_error(error: impl std::fmt::Display) -> ApiError {
    ApiError::Runtime(ExecutionError::Backend(error.to_string()))
}

fn plan<S: SessionStore>(
    runtime: &mut GpuRuntime,
    validated: ValidatedGraph,
    inputs: &BTreeMap<String, RuntimeValue>,
    store: &mut S,
) -> Result<GpuExecutionPlan, ApiError> {
    let name = validated.source.name().to_owned();
    let plan = runtime.plan_with_store(validated, inputs, store).map_err(gpu_error)?;
    let report = plan.report();
    info!(
        graph = name,
        predicted_seconds = report.predicted_seconds,
        limiting_stage = ?report.limiting_stage,
        reason = report.reason,
        regions = plan.compiled_region_count(),
        launches = plan.compiled_launch_count(),
        "planned Exponent-LUT GPU stage"
    );
    Ok(plan)
}

impl<S: SessionStore + Send> StageExecution<S> for GpuRuntime {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        let name = validated.source.name().to_owned();
        let started = std::time::Instant::now();
        let mut plan = plan(self, validated, &inputs, store)?;
        let planned = std::time::Instant::now();
        let result = self
            .execute_with_artifacts(&mut plan, inputs, store, execution_nonce)
            .map_err(gpu_error)?;
        info!(
            graph = name,
            plan_ms = (planned - started).as_millis() as u64,
            execute_ms = planned.elapsed().as_millis() as u64,
            "Exponent-LUT GPU stage timing"
        );
        result.production_id.clone().ok_or_else(|| {
            ApiError::input("artifact execution did not produce a production identity")
        })
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        let name = validated.source.name().to_owned();
        let started = std::time::Instant::now();
        let mut plan = plan(self, validated, &inputs, store)?;
        let planned = std::time::Instant::now();
        let result = self
            .execute_with_artifacts(&mut plan, inputs, store, execution_nonce)
            .map_err(gpu_error)?;
        info!(
            graph = name,
            plan_ms = (planned - started).as_millis() as u64,
            execute_ms = planned.elapsed().as_millis() as u64,
            "Exponent-LUT GPU stage timing"
        );
        // The result owns its outputs, grouped per composite value; stage
        // outputs are addressed by their leaf names (`name.i`).
        Ok(StageOutputs {
            outputs: mxx_backends::backend::expand_composite_values(result.into_outputs()),
        })
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError> {
        GpuRuntime::download_matrix(self, value).map_err(gpu_error)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        api::{
            StageExecution, StageGraph, execute_artifacts, execute_graph, reload_stage_artifacts,
        },
        test_support::ring_of,
    };
    use mxx_backends::{
        GpuRuntime, MemoryArtifactStore,
        backend::poly_gpu::gpu_backend,
        poly::{
            Poly, PolyParams,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::{IntExpr, ParamEnv};
    use num_bigint::BigUint;
    use serial_test::serial as sequential;
    use std::collections::BTreeMap;

    fn runtime(parameters: &DCRTPolyParams) -> GpuRuntime {
        GpuRuntime::new(gpu_backend([GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.moduli().to_vec(),
            parameters.base_bits(),
            None,
        )]))
        .expect("GPU runtime")
    }

    #[test]
    #[sequential(dcrt_runtime)]
    fn test_gpu_stage_execution_publishes_artifacts_and_returns_owned_outputs() {
        let parameters = DCRTPolyParams::new(8, 2, 20, 4, None, None);
        let ring = ring_of(&parameters);
        let mut runtime = runtime(&parameters);
        let mut store = MemoryArtifactStore::default();

        let producer = DslContext::new("gpu-stage-artifact")
            .transferred_output("value", ring.polynomial([IntExpr::constant(3)]))
            .unwrap()
            .build()
            .unwrap();
        let stage = execute_artifacts(
            StageGraph::new(producer, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            "descriptor".to_owned(),
            rand::random(),
            &mut runtime,
            &mut store,
            |_| Ok(()),
        )
        .expect("GPU artifact stage executes");
        assert!(stage.manifest().artifacts.contains_key("value"));
        let reloaded =
            reload_stage_artifacts(&mut store, stage.session().clone(), "reloaded".to_owned())
                .expect("finalized GPU stage reloads");
        assert_eq!(reloaded.production(), stage.production());

        let first = DslContext::new("gpu-stage-evaluate")
            .output("value", ring.polynomial([IntExpr::constant(5)]))
            .unwrap()
            .build()
            .unwrap();
        let first = execute_graph(
            StageGraph::new(first, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut runtime,
            &mut store,
            |_| Ok(()),
        )
        .expect("GPU graph evaluates");
        // The owned output must survive its plan and bind into another plan.
        let input = ring.input("previous", (1, 1));
        let second = DslContext::new("gpu-stage-rebind")
            .output("doubled", input.clone() + input)
            .unwrap()
            .build()
            .unwrap();
        let second = execute_graph(
            StageGraph::new(
                second,
                BTreeMap::from([("previous".to_owned(), first.outputs["value"].clone())]),
                BTreeMap::new(),
            ),
            &ParamEnv::default(),
            &mut runtime,
            &mut store,
            |_| Ok(()),
        )
        .expect("owned GPU output binds into a second plan");
        let doubled = StageExecution::<MemoryArtifactStore>::download_matrix(
            &runtime,
            &second.outputs["doubled"],
        )
        .unwrap();
        let coefficients = doubled.entry(0, 0).coeffs_biguints();
        assert_eq!(coefficients[0], BigUint::from(10u8));
        assert!(coefficients[1..].iter().all(|value| *value == BigUint::from(0u8)));
    }
}
