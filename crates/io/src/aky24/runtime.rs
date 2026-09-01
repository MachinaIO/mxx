//! Runtime integration for the manifest-linked AKY24 iO cascade.

use super::{
    artifacts::Aky24ArtifactNames,
    cascade::{Aky24CascadeCompiler, Aky24CascadeGraphError},
    config::Aky24GoldreichPrf,
};
use crate::Obfuscation;
use mxx_ir_core::{
    ParamEnv,
    artifact::{ProductionId, production_id},
    encoding::spec_hash,
};
use mxx_primitives::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue, SessionStore, backend::poly::PolyBackend, execute,
    execute_in_session_with_config, transcript::SamplingMode,
};
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

/// Compact public iO object. All matrix artifacts live in the session store
/// under `production`; no trapdoor or private preprocessing value is retained.
#[derive(Clone, Debug)]
pub struct Aky24IoObfuscation {
    pub function: Aky24GoldreichPrf,
    pub production: ProductionId,
}

pub struct Aky24IoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
{
    pub compiler: Aky24CascadeCompiler,
    pub parameters: <M::P as Poly>::Params,
    pub backend: PolyBackend<M, U, H, T>,
    pub store: S,
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, Error)]
pub enum Aky24IoRuntimeError {
    #[error(transparent)]
    Compile(#[from] Aky24CascadeGraphError),
    #[error("AKY24 iO runtime parameters do not match its compiler configuration")]
    ParameterMismatch,
    #[error("the requested function differs from the function fixed by the AKY24 compiler")]
    FunctionMismatch,
    #[error("AKY24 iO graph validation failed: {0}")]
    Validation(String),
    #[error("AKY24 iO artifact store failed: {0}")]
    Store(String),
    #[error("AKY24 iO graph execution failed: {0}")]
    Execution(String),
    #[error("AKY24 iO input has the wrong number of bits")]
    InputLength,
    #[error("AKY24 iO preprocessing graph does not match the public handle")]
    ProductionGraphMismatch,
    #[error("AKY24 iO evaluation returned an invalid Boolean output family")]
    OutputLayout,
}

impl<M, U, H, T, S> Aky24IoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    M::P: Send + Sync + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    pub fn new(
        compiler: Aky24CascadeCompiler,
        parameters: <M::P as Poly>::Params,
        store: S,
    ) -> Result<Self, Aky24IoRuntimeError> {
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        let config = compiler.config();
        if config.modulus != num_bigint::BigInt::from(modulus.as_ref().clone()) ||
            config.ring_dimension != parameters.ring_dimension() as usize ||
            config.digit_count != parameters.modulus_digits() ||
            config.gadget_base != num_bigint::BigInt::from(1u64 << parameters.base_bits())
        {
            return Err(Aky24IoRuntimeError::ParameterMismatch);
        }
        Ok(Self {
            compiler,
            backend: PolyBackend::new_for_execution([parameters.clone()]),
            parameters,
            store,
            execution_config: ExecutionConfig::default(),
        })
    }

    pub fn with_execution_config(mut self, execution_config: ExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    pub fn obfuscate_with_nonce(
        &mut self,
        function: &Aky24GoldreichPrf,
        execution_nonce: [u8; 32],
    ) -> Result<Aky24IoObfuscation, Aky24IoRuntimeError> {
        if function != &self.compiler.config().function {
            return Err(Aky24IoRuntimeError::FunctionMismatch);
        }
        let preprocessing = self.compiler.build_preprocessing()?.graph;
        let bindings = ParamEnv::default();
        let validated = preprocessing
            .validate(&bindings)
            .map_err(|error| Aky24IoRuntimeError::Validation(error.to_string()))?;
        let production = production_id(
            spec_hash(&validated.source, &validated.bindings)
                .map_err(|error| Aky24IoRuntimeError::Validation(error.to_string()))?,
            execution_nonce,
        );
        execute_in_session_with_config(
            &validated,
            &mut self.backend,
            BTreeMap::new(),
            &mut self.store,
            execution_nonce,
            self.execution_config,
        )
        .map_err(|error| Aky24IoRuntimeError::Execution(error.to_string()))?;
        Ok(Aky24IoObfuscation { function: function.clone(), production })
    }

    pub fn evaluate_bits(
        &mut self,
        obfuscation: &Aky24IoObfuscation,
        input: &[bool],
    ) -> Result<Vec<bool>, Aky24IoRuntimeError> {
        if input.len() != self.compiler.config().input_size {
            return Err(Aky24IoRuntimeError::InputLength);
        }
        if obfuscation.function != self.compiler.config().function {
            return Err(Aky24IoRuntimeError::FunctionMismatch);
        }

        let preprocessing = self.compiler.build_preprocessing()?.graph;
        let validated_preprocessing = preprocessing
            .validate(&ParamEnv::default())
            .map_err(|error| Aky24IoRuntimeError::Validation(error.to_string()))?;
        let preprocessing_hash =
            spec_hash(&validated_preprocessing.source, &validated_preprocessing.bindings)
                .map_err(|error| Aky24IoRuntimeError::Validation(error.to_string()))?;
        if preprocessing_hash != obfuscation.production.spec_hash {
            return Err(Aky24IoRuntimeError::ProductionGraphMismatch);
        }

        let manifest = self
            .store
            .load_manifest(&obfuscation.production)
            .map_err(|error| Aky24IoRuntimeError::Store(error.to_string()))?;
        let manifests = BTreeMap::from([(obfuscation.production.clone(), manifest)]);
        let evaluation =
            self.compiler.build_evaluation(input, obfuscation.production.clone())?.graph;
        let validated = evaluation
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .map_err(|error| Aky24IoRuntimeError::Validation(error.to_string()))?;
        let result = execute(
            &validated,
            &mut self.backend,
            BTreeMap::new(),
            &mut self.store,
            SamplingMode::Fresh,
        )
        .map_err(|error| Aky24IoRuntimeError::Execution(error.to_string()))?;
        let Some(RuntimeValue::Family(outputs)) =
            result.outputs.get(Aky24ArtifactNames::OUTPUT)
        else {
            return Err(Aky24IoRuntimeError::OutputLayout);
        };
        if outputs.len() != self.compiler.config().function.output_bits {
            return Err(Aky24IoRuntimeError::OutputLayout);
        }
        outputs
            .iter()
            .map(|output| match output {
                RuntimeValue::Bool(value) => Ok(*value),
                _ => Err(Aky24IoRuntimeError::OutputLayout),
            })
            .collect()
    }
}

impl<M, U, H, T, S> Obfuscation for Aky24IoRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    M::P: Send + Sync + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    type Function = Aky24GoldreichPrf;
    type Obfuscation = Aky24IoObfuscation;
    type Input = Vec<bool>;
    type Output = Vec<bool>;
    type Error = Aky24IoRuntimeError;

    fn obfuscate(&mut self, function: &Self::Function) -> Result<Self::Obfuscation, Self::Error> {
        self.obfuscate_with_nonce(function, rand::random())
    }

    fn evaluate(
        &mut self,
        obfuscation: &Self::Obfuscation,
        input: &Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        self.evaluate_bits(obfuscation, input)
    }
}
