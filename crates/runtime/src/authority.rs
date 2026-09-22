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
pub trait ExecutionAuthority<S: SessionStore> {
    type Backend: Backend;
    type Prepared;
    type Result;

    fn prepare(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue<Self::Backend>>,
    ) -> Result<Self::Prepared, String>;

    fn run(
        &mut self,
        prepared: &mut Self::Prepared,
        inputs: BTreeMap<String, RuntimeValue<Self::Backend>>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<Self::Result, String>;
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

impl<B, S> ExecutionAuthority<S> for CpuExecution<B>
where
    B: Backend,
    S: SessionStore,
{
    type Backend = B;
    type Prepared = PreparedCpuExecution;
    type Result = ExecutionResult<B>;

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

    fn run(
        &mut self,
        prepared: &mut Self::Prepared,
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
            prepared.config.clone(),
        )
        .map_err(|error| error.to_string())
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use crate::{backend::poly_gpu::GpuDcrtBackend, gpu_runtime::GpuExecutionPlan};

    impl<S> ExecutionAuthority<S> for crate::gpu_runtime::GpuRuntime
    where
        S: SessionStore + Send,
    {
        type Backend = GpuDcrtBackend;
        type Prepared = GpuExecutionPlan;
        type Result = crate::gpu_runtime::GpuExecutionResult;

        fn prepare(
            &mut self,
            validated: ValidatedGraph,
            inputs: &BTreeMap<String, RuntimeValue<Self::Backend>>,
        ) -> Result<Self::Prepared, String> {
            self.plan(validated, inputs).map_err(|error| error.to_string())
        }

        fn run(
            &mut self,
            prepared: &mut Self::Prepared,
            inputs: BTreeMap<String, RuntimeValue<Self::Backend>>,
            store: &mut S,
            execution_nonce: [u8; 32],
        ) -> Result<Self::Result, String> {
            self.execute(prepared, inputs, store, execution_nonce)
                .map_err(|error| error.to_string())
        }
    }
}
