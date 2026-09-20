//! Runtime-owned execution authorities.
//!
//! An authority owns the backend and is the only boundary at which a
//! validated graph becomes executable state. CPU execution keeps the normal
//! dynamic backend, while GPU execution must prepare an exact frozen plan
//! before its prepared value can run.

use crate::{
    Backend, RuntimeValue,
    executor::{ExecutionConfig, ExecutionResult, execute_prepared},
    session::SessionStore,
};
use mxx_ir_core::ValidatedGraph;
use std::collections::{BTreeMap, BTreeSet};

/// A backend owner that performs the prepare-then-run lifecycle for a graph.
pub trait ExecutionAuthority {
    type Backend: Backend;
    type Prepared;

    fn prepare(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue<Self::Backend>>,
    ) -> Result<Self::Prepared, String>;

    fn run<S: SessionStore>(
        &mut self,
        prepared: Self::Prepared,
        inputs: BTreeMap<String, RuntimeValue<Self::Backend>>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ExecutionResult<Self::Backend>, String>;
}

/// CPU authority. It deliberately exposes no GPU plan configuration.
pub struct CpuExecution<B: Backend> {
    backend: B,
}

pub struct PreparedCpuExecution {
    validated: ValidatedGraph,
    config: ExecutionConfig,
    input_names: BTreeSet<String>,
}

impl<B: Backend> CpuExecution<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: Backend> ExecutionAuthority for CpuExecution<B> {
    type Backend = B;
    type Prepared = PreparedCpuExecution;

    fn prepare(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue<Self::Backend>>,
    ) -> Result<Self::Prepared, String> {
        Ok(PreparedCpuExecution {
            validated,
            config: ExecutionConfig::default(),
            input_names: inputs.keys().cloned().collect(),
        })
    }

    fn run<S: SessionStore>(
        &mut self,
        prepared: Self::Prepared,
        inputs: BTreeMap<String, RuntimeValue<Self::Backend>>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ExecutionResult<Self::Backend>, String> {
        let input_names = inputs.keys().cloned().collect::<BTreeSet<_>>();
        if input_names != prepared.input_names {
            return Err("CPU execution inputs differ from the prepared input set".into());
        }
        execute_prepared(
            &prepared.validated,
            &mut self.backend,
            inputs,
            store,
            execution_nonce,
            prepared.config,
        )
        .map_err(|error| error.to_string())
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::{
        backend::poly_gpu::GpuDcrtBackend,
        gpu_measurement::{
            GpuPreparationRequest, GpuWarmupMeasurementConfig, PreparedGpuExecution,
            prepare as prepare_gpu,
        },
    };
    use mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams;

    /// GPU authority owning native parameters, backend state, and the pure
    /// preparation reports produced for each graph identity.
    pub struct GpuExecution {
        backend: GpuDcrtBackend,
        parameters: Vec<GpuDCRTPolyParams>,
        measurement_config: GpuWarmupMeasurementConfig,
        implementation_variant: String,
        preparations: Vec<PreparedGpuExecution>,
    }

    impl GpuExecution {
        pub fn new(
            backend: GpuDcrtBackend,
            parameters: impl IntoIterator<Item = GpuDCRTPolyParams>,
            measurement_config: GpuWarmupMeasurementConfig,
            implementation_variant: impl Into<String>,
        ) -> Self {
            Self {
                backend,
                parameters: parameters.into_iter().collect(),
                measurement_config,
                implementation_variant: implementation_variant.into(),
                preparations: Vec::new(),
            }
        }

        pub fn backend(&self) -> &GpuDcrtBackend {
            &self.backend
        }

        pub fn backend_mut(&mut self) -> &mut GpuDcrtBackend {
            &mut self.backend
        }

        pub fn preparations(&self) -> &[PreparedGpuExecution] {
            &self.preparations
        }
    }

    impl ExecutionAuthority for GpuExecution {
        type Backend = GpuDcrtBackend;
        type Prepared = PreparedGpuExecution;

        fn prepare(
            &mut self,
            validated: ValidatedGraph,
            inputs: &BTreeMap<String, RuntimeValue<Self::Backend>>,
        ) -> Result<Self::Prepared, String> {
            let execution_config = ExecutionConfig::default();
            let prepared = prepare_gpu(GpuPreparationRequest {
                validated,
                backend: &mut self.backend,
                inputs,
                parameters: &self.parameters,
                default_tile_widths: vec![1, 2, 4, 8],
                implementation_variant: self.implementation_variant.clone(),
                measurement_config: self.measurement_config.clone(),
                execution_config,
            })
            .map_err(|error| error.to_string())?;
            self.preparations.push(prepared.clone());
            Ok(prepared)
        }

        fn run<S: SessionStore>(
            &mut self,
            prepared: Self::Prepared,
            inputs: BTreeMap<String, RuntimeValue<Self::Backend>>,
            store: &mut S,
            execution_nonce: [u8; 32],
        ) -> Result<ExecutionResult<Self::Backend>, String> {
            prepared
                .run(&mut self.backend, inputs, store, execution_nonce)
                .map_err(|error| error.to_string())
        }
    }

    pub use GpuExecution as PublicGpuExecution;
}

#[cfg(feature = "gpu")]
pub use gpu::PublicGpuExecution as GpuExecution;
