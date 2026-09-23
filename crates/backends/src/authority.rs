//! Runtime-owned execution authorities.
//!
//! An authority owns execution state and is the boundary at which a validated
//! graph becomes executable. GPU execution prepares an exact frozen plan
//! before its prepared value can run.

use crate::{
    RuntimeValue,
    backend::poly::CpuDcrtBackend,
    executor::{ExecutionConfig, ExecutionResult, execute_prepared},
    session::SessionStore,
};
use mxx_ir_core::ValidatedGraph;
use std::collections::{BTreeMap, BTreeSet};

/// An execution owner that performs the prepare-then-run lifecycle for a graph.
pub trait ExecutionAuthority<S: SessionStore> {
    type Prepared;
    type Result<'a>
    where
        Self: 'a,
        Self::Prepared: 'a;

    fn prepare(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<Self::Prepared, String>;

    fn run<'a>(
        &mut self,
        prepared: &'a mut Self::Prepared,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<Self::Result<'a>, String>;
}

/// CPU authority. It deliberately exposes no GPU plan configuration.
pub struct CpuExecution {
    backend: CpuDcrtBackend,
}

pub struct PreparedCpuExecution {
    validated: ValidatedGraph,
    config: ExecutionConfig,
    input_names: BTreeSet<String>,
}

impl CpuExecution {
    pub fn new(backend: CpuDcrtBackend) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &CpuDcrtBackend {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut CpuDcrtBackend {
        &mut self.backend
    }
}

impl<S: SessionStore> ExecutionAuthority<S> for CpuExecution {
    type Prepared = PreparedCpuExecution;
    type Result<'a> = ExecutionResult;

    fn prepare(
        &mut self,
        validated: ValidatedGraph,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<Self::Prepared, String> {
        Ok(PreparedCpuExecution {
            validated,
            config: ExecutionConfig::default(),
            input_names: inputs.keys().cloned().collect(),
        })
    }

    fn run<'a>(
        &mut self,
        prepared: &'a mut Self::Prepared,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut S,
        execution_nonce: [u8; 32],
    ) -> Result<ExecutionResult, String> {
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
    use crate::gpu_runtime::GpuExecutionPlan;

    impl<S> ExecutionAuthority<S> for crate::gpu_runtime::GpuRuntime
    where
        S: SessionStore + Send,
    {
        type Prepared = GpuExecutionPlan;
        type Result<'a> = crate::gpu_runtime::GpuExecutionResult<'a>;

        fn prepare(
            &mut self,
            validated: ValidatedGraph,
            inputs: &BTreeMap<String, RuntimeValue>,
        ) -> Result<Self::Prepared, String> {
            self.plan(validated, inputs).map_err(|error| error.to_string())
        }

        fn run<'a>(
            &mut self,
            prepared: &'a mut Self::Prepared,
            inputs: BTreeMap<String, RuntimeValue>,
            store: &mut S,
            execution_nonce: [u8; 32],
        ) -> Result<Self::Result<'a>, String> {
            self.execute(prepared, inputs, store, execution_nonce)
                .map_err(|error| error.to_string())
        }
    }
}
