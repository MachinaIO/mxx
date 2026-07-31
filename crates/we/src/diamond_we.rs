use std::{marker::PhantomData, path::PathBuf, sync::Arc};

use crate::{
    circuit::{NoCircuitEvaluator, PolyCircuit},
    input_injector::DiamondInjector,
    matrix::PolyMatrix,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};

use super::WitnessEnc;

pub mod bench_estimator;
pub mod graph;
mod graph_runtime;
pub mod simulation;

pub use bench_estimator::{DiamondWEBenchEstimate, DiamondWEBenchEstimator};
pub use graph_runtime::{DiamondWECiphertext, DiamondWEGraphRuntimeError};
pub use simulation::{
    DiamondWECrtDepthSearchResult, DiamondWEErrorSimulation, diamond_we_find_crt_depth,
};

#[derive(Clone)]
pub struct DiamondWE<
    M,
    US,
    HS,
    TS,
    PKPE = NoCircuitEvaluator,
    PKST = NoCircuitEvaluator,
    ENCPE = NoCircuitEvaluator,
    ENCST = NoCircuitEvaluator,
> where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub injector: DiamondInjector<M, US, HS, TS>,
    pub witness_size: usize,
    pub artifact_dir: PathBuf,
    pub bgg_tag: Vec<u8>,
    pub evaluator_checkpoint_id: Vec<u8>,
    pub pk_lookup_evaluator: Option<PKPE>,
    pub pk_slot_transfer_evaluator: Option<PKST>,
    pub target_witness: Option<Vec<bool>>,
    pub enc_lookup_base_matrix: Option<M>,
    pub enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
    pub enc_lookup_evaluator: Option<ENCPE>,
    pub enc_slot_transfer_evaluator: Option<ENCST>,
    _m: PhantomData<M>,
}

impl<M, US, HS, TS>
    DiamondWE<
        M,
        US,
        HS,
        TS,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    >
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub fn new(
        injector: DiamondInjector<M, US, HS, TS>,
        witness_size: usize,
        artifact_dir: impl Into<PathBuf>,
        bgg_tag: Vec<u8>,
        target_witness: Option<Vec<bool>>,
    ) -> Self {
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            evaluator_checkpoint_id: Vec::new(),
            pk_lookup_evaluator: None,
            pk_slot_transfer_evaluator: None,
            enc_lookup_base_matrix: None,
            enc_lookup_evaluator_factory: None,
            target_witness,
            enc_lookup_evaluator: None,
            enc_slot_transfer_evaluator: None,
            _m: PhantomData,
        }
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    #[allow(clippy::too_many_arguments)]
    pub fn with_evaluators(
        injector: DiamondInjector<M, US, HS, TS>,
        witness_size: usize,
        artifact_dir: impl Into<PathBuf>,
        bgg_tag: Vec<u8>,
        evaluator_checkpoint_id: Vec<u8>,
        target_witness: Option<Vec<bool>>,
        pk_lookup_evaluator: Option<PKPE>,
        pk_slot_transfer_evaluator: Option<PKST>,
        enc_lookup_base_matrix: Option<M>,
        enc_lookup_evaluator_factory: Option<Arc<dyn Fn(M) -> ENCPE + Send + Sync>>,
        enc_lookup_evaluator: Option<ENCPE>,
        enc_slot_transfer_evaluator: Option<ENCST>,
    ) -> Self {
        let has_evaluator_state = pk_lookup_evaluator.is_some() ||
            pk_slot_transfer_evaluator.is_some() ||
            enc_lookup_evaluator_factory.is_some() ||
            enc_lookup_evaluator.is_some() ||
            enc_slot_transfer_evaluator.is_some();
        assert!(
            !has_evaluator_state || !evaluator_checkpoint_id.is_empty(),
            "DiamondWE evaluator_checkpoint_id must identify evaluator configuration and state"
        );
        Self {
            injector,
            witness_size,
            artifact_dir: artifact_dir.into(),
            bgg_tag,
            evaluator_checkpoint_id,
            pk_lookup_evaluator,
            pk_slot_transfer_evaluator,
            enc_lookup_base_matrix,
            target_witness,
            enc_lookup_evaluator_factory,
            enc_lookup_evaluator,
            enc_slot_transfer_evaluator,
            _m: PhantomData,
        }
    }
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> WitnessEnc<M::P>
    for DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    TS::Trapdoor: Clone + std::fmt::Debug,
{
    type Msg = bool;
    type Inst = Vec<bool>;
    type Wtns = Vec<bool>;
    type Ciphertext = DiamondWECiphertext<M::P>;

    fn enc(
        &self,
        message: &Self::Msg,
        circuit: PolyCircuit<M::P>,
        instance: &Self::Inst,
    ) -> Self::Ciphertext {
        self.try_encrypt(*message, circuit, instance)
            .unwrap_or_else(|error| panic!("DiamondWE Graph IR encryption failed: {error}"))
    }

    fn dec(&self, ciphertext: &Self::Ciphertext, witness: &Self::Wtns) -> Self::Msg {
        self.try_decrypt(ciphertext, witness)
            .unwrap_or_else(|error| panic!("DiamondWE Graph IR decryption failed: {error}"))
    }
}
