use std::hint::black_box;

use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::debug;

use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, PublicKeyAuxBenchEstimator,
        benchmark_gate_operation, estimate_public_key_circuit_bench_with_aux,
        scale_independent_estimate, scale_independent_summary,
    },
    bgg::{encoding::BggEncoding, public_key::BggPublicKey, sampler::BGGPublicKeySampler},
    circuit::PolyCircuit,
    input_injector::DIAMOND_SECRET_SIZE,
    io::diamond_io::DiamondIONativeBenchEstimator,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    slot_transfer::bgg_pubkey::{column_chunk_bounds, column_chunk_count},
};

use super::DiamondWE;

#[derive(Debug, Clone, PartialEq)]
pub struct DiamondWEBenchEstimate {
    pub enc: CircuitBenchSummary,
    pub dec: CircuitBenchSummary,
    pub enc_input_injection: CircuitBenchSummary,
    pub dec_input_injection: CircuitBenchSummary,
    pub ciphertext_bytes: BigUint,
    pub input_injection_bytes: BigUint,
    pub we_preimage_bytes: BigUint,
}

#[derive(Debug, Clone)]
pub struct DiamondWEBenchEstimator<'a, PKBE, EncBE, NBE> {
    pub public_key_estimator: &'a PKBE,
    pub encoding_estimator: &'a EncBE,
    pub native_estimator: &'a NBE,
    pub trapdoor_checkpoint: CircuitBenchEstimate,
    pub trapdoor_preimage: CircuitBenchEstimate,
    pub final_preimage_one_column: CircuitBenchEstimate,
    pub bgg_public_key_sample: CircuitBenchEstimate,
    pub scalar_matrix_hash_sample: CircuitBenchEstimate,
}

impl<'a, PKBE, EncBE, NBE> DiamondWEBenchEstimator<'a, PKBE, EncBE, NBE>
where
    PKBE: Sync,
    EncBE: Sync,
    NBE: DiamondIONativeBenchEstimator + Sync,
{
    pub fn new<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        public_key_estimator: &'a PKBE,
        encoding_estimator: &'a EncBE,
        native_estimator: &'a NBE,
        diamond: &DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        iterations: usize,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        <M::P as Poly>::Params: Sync,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let units = DiamondWEBenchUnitEstimates::benchmark(diamond, iterations);
        Self {
            public_key_estimator,
            encoding_estimator,
            native_estimator,
            trapdoor_checkpoint: units.trapdoor_checkpoint,
            trapdoor_preimage: units.trapdoor_preimage,
            final_preimage_one_column: units.final_preimage_one_column,
            bgg_public_key_sample: units.bgg_public_key_sample,
            scalar_matrix_hash_sample: units.scalar_matrix_hash_sample,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_unit_costs(
        public_key_estimator: &'a PKBE,
        encoding_estimator: &'a EncBE,
        native_estimator: &'a NBE,
        trapdoor_checkpoint: CircuitBenchEstimate,
        trapdoor_preimage: CircuitBenchEstimate,
        final_preimage_one_column: CircuitBenchEstimate,
        bgg_public_key_sample: CircuitBenchEstimate,
        scalar_matrix_hash_sample: CircuitBenchEstimate,
    ) -> Self {
        Self {
            public_key_estimator,
            encoding_estimator,
            native_estimator,
            trapdoor_checkpoint,
            trapdoor_preimage,
            final_preimage_one_column,
            bgg_public_key_sample,
            scalar_matrix_hash_sample,
        }
    }

    pub fn estimate<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        &self,
        diamond: &DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        circuit: &PolyCircuit<M::P>,
    ) -> DiamondWEBenchEstimate
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        <M::P as Poly>::Params: Sync,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
        PKBE: BenchEstimator<BggPublicKey<M>> + PublicKeyAuxBenchEstimator<M::P> + Sync,
        EncBE: BenchEstimator<BggEncoding<M>> + Sync,
    {
        let shape = DiamondWEBenchShape::from_diamond_and_circuit(diamond, circuit);
        let input_injection = self.estimate_input_injection(shape.clone());
        let params = &diamond.injector.params;
        let enc = self.estimate_enc::<M>(params, circuit, shape.clone(), &input_injection);
        let dec = self.estimate_dec(circuit, shape.clone(), &input_injection);
        let storage = self.estimate_storage(shape);
        let DiamondWEStorageEstimate {
            input_injection_bytes, we_preimage_bytes, total_bytes, ..
        } = storage;
        DiamondWEBenchEstimate {
            enc,
            dec,
            enc_input_injection: input_injection.enc,
            dec_input_injection: input_injection.dec,
            ciphertext_bytes: total_bytes,
            input_injection_bytes,
            we_preimage_bytes,
        }
    }

    fn estimate_enc<M>(
        &self,
        params: &<M::P as Poly>::Params,
        circuit: &PolyCircuit<M::P>,
        shape: DiamondWEBenchShape,
        input_injection: &DiamondWEInputInjectionBenchEstimateParts,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        <M::P as Poly>::Params: Sync,
        PKBE: BenchEstimator<BggPublicKey<M>> + PublicKeyAuxBenchEstimator<M::P> + Sync,
    {
        let (public_sampling, enc_branches) = rayon::join(
            || {
                parallel_summaries(&[
                    scale_estimate(self.bgg_public_key_sample.clone(), shape.witness_size + 1),
                    scale_estimate(self.scalar_matrix_hash_sample.clone(), 2),
                ])
            },
            || {
                let instance_pubkeys = scale_estimate(
                    self.public_key_estimator.estimate_small_scalar_mul(&[1]),
                    shape.instance_size,
                );
                let public_circuit_eval = estimate_public_key_circuit_bench_with_aux::<
                    BggPublicKey<M>,
                    PKBE,
                >(
                    self.public_key_estimator, params, circuit
                );
                let public_circuit_branch =
                    sequential_summaries(&[instance_pubkeys.clone(), public_circuit_eval.clone()]);
                let early_projection = self.estimate_enc_early_projection(shape.clone());
                let input_and_early_projection =
                    sequential_summaries(&[input_injection.enc.clone(), early_projection.clone()]);
                (
                    instance_pubkeys,
                    public_circuit_eval,
                    public_circuit_branch,
                    early_projection,
                    input_and_early_projection,
                )
            },
        );
        let (
            instance_pubkeys,
            public_circuit_eval,
            public_circuit_branch,
            early_projection,
            input_and_early_projection,
        ) = enc_branches;
        let public_and_input = parallel_summaries(&[
            input_and_early_projection.clone(),
            public_circuit_branch.clone(),
        ]);
        let decoder_projection = self.estimate_enc_decoder_projection::<M>(shape.clone());
        let total = sequential_summaries(&[
            public_sampling.clone(),
            public_and_input.clone(),
            decoder_projection.clone(),
        ]);

        debug!(
            ?public_sampling,
            ?instance_pubkeys,
            ?public_circuit_eval,
            ?early_projection,
            ?decoder_projection,
            ?total,
            "estimated DiamondWE enc benchmark"
        );
        total
    }

    fn estimate_dec<M>(
        &self,
        circuit: &PolyCircuit<M::P>,
        shape: DiamondWEBenchShape,
        input_injection: &DiamondWEInputInjectionBenchEstimateParts,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<BggEncoding<M>> + Sync,
    {
        let (public_resampling, circuit_input_path) = rayon::join(
            || {
                parallel_summaries(&[
                    scale_estimate(self.bgg_public_key_sample.clone(), shape.witness_size + 1),
                    scale_estimate(self.scalar_matrix_hash_sample.clone(), 2),
                ])
            },
            || self.estimate_dec_circuit_input_path::<M>(circuit, shape.clone()),
        );
        let initial = parallel_summaries(&[public_resampling.clone(), input_injection.dec.clone()]);

        let decoder_vector_projection = scale_summary(
            self.native_estimator.estimate_vector_matrix_product(shape.state_col_size, 1),
            1,
        );
        let k_projection = decoder_vector_projection.clone();
        let decoder_projection = decoder_vector_projection;
        let parallel_after_states = parallel_summaries(&[
            circuit_input_path.clone(),
            k_projection.clone(),
            decoder_projection.clone(),
        ]);
        let cancellation = self.estimate_dec_vector_cancellation::<M>(shape.clone());
        let total = sequential_summaries(&[
            initial.clone(),
            parallel_after_states.clone(),
            cancellation.clone(),
        ]);

        debug!(
            ?public_resampling,
            input_injection = ?input_injection.dec,
            ?circuit_input_path,
            ?k_projection,
            ?decoder_projection,
            ?cancellation,
            ?total,
            "estimated DiamondWE dec benchmark"
        );
        total
    }

    fn estimate_enc_early_projection(&self, shape: DiamondWEBenchShape) -> CircuitBenchSummary {
        let ordinary_target_building = scale_summary(
            sequential_summaries(&[
                self.native_estimator.estimate_row_parallel_matrix_product(
                    DIAMOND_SECRET_SIZE,
                    DIAMOND_SECRET_SIZE,
                    shape.modulus_digits,
                ),
                estimate_summary(
                    self.native_estimator.estimate_vector_sub(
                        DIAMOND_SECRET_SIZE
                            .checked_mul(shape.modulus_digits)
                            .expect("DiamondWE ordinary target vector size overflow"),
                    ),
                ),
            ]),
            shape.ordinary_preimage_count(),
        );
        let ordinary_preimages = scale_summary(
            self.final_preimage_summary_for_width(shape.modulus_digits),
            shape.ordinary_preimage_count(),
        );
        let k_preimage = self.final_preimage_summary_for_width(1);

        let mut parts = vec![ordinary_target_building, ordinary_preimages, k_preimage];
        if let Some(lookup_cols) = shape.lookup_bridge_cols {
            parts.push(self.final_preimage_summary_for_width(lookup_cols));
        }
        parallel_summaries(parts.as_slice())
    }

    fn estimate_enc_decoder_projection<M>(&self, shape: DiamondWEBenchShape) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        PKBE: BenchEstimator<BggPublicKey<M>>,
    {
        let target = sequential_summaries(&[
            estimate_summary(self.public_key_estimator.estimate_sub()),
            self.native_estimator
                .estimate_vector_matrix_product(shape.modulus_digits, DIAMOND_SECRET_SIZE),
            estimate_summary(self.native_estimator.estimate_vector_add(DIAMOND_SECRET_SIZE)),
        ]);
        sequential_summaries(&[target, self.final_preimage_summary_for_width(1)])
    }

    fn estimate_dec_circuit_input_path<M>(
        &self,
        circuit: &PolyCircuit<M::P>,
        shape: DiamondWEBenchShape,
    ) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<BggEncoding<M>> + Sync,
    {
        let one_and_witness = scale_summary(
            self.native_estimator
                .estimate_vector_matrix_product(shape.state_col_size, shape.modulus_digits),
            shape.ordinary_preimage_count(),
        );
        let lookup = shape
            .dec_lookup_bridge_cols
            .map(|cols| {
                self.native_estimator.estimate_vector_matrix_product(shape.state_col_size, cols)
            })
            .unwrap_or_default();
        let input_projection = parallel_summaries(&[one_and_witness, lookup]);
        let instance_encodings = scale_estimate(
            self.encoding_estimator.estimate_small_scalar_mul(&[1]),
            shape.instance_size,
        );
        let circuit_eval = self.encoding_estimator.estimate_circuit_bench(circuit);
        sequential_summaries(&[input_projection, instance_encodings, circuit_eval])
    }

    fn estimate_dec_vector_cancellation<M>(&self, shape: DiamondWEBenchShape) -> CircuitBenchSummary
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        EncBE: BenchEstimator<BggEncoding<M>>,
    {
        sequential_summaries(&[
            estimate_summary(self.encoding_estimator.estimate_sub()),
            self.native_estimator
                .estimate_vector_matrix_product(shape.modulus_digits, DIAMOND_SECRET_SIZE),
            estimate_summary(self.native_estimator.estimate_vector_add(DIAMOND_SECRET_SIZE)),
            estimate_summary(self.native_estimator.estimate_vector_sub(DIAMOND_SECRET_SIZE)),
        ])
    }

    fn estimate_input_injection(
        &self,
        shape: DiamondWEBenchShape,
    ) -> DiamondWEInputInjectionBenchEstimateParts {
        let checkpoint_sampling =
            scale_estimate(self.trapdoor_checkpoint.clone(), shape.checkpoint_count);
        let initial_state = sequential_summaries(&[
            self.native_estimator
                .estimate_vector_matrix_product(shape.state_row_size, shape.state_col_size),
            estimate_summary(self.native_estimator.estimate_vector_add(shape.state_col_size)),
        ]);
        let transition_target_building =
            shape.estimate_transition_target_building(self.native_estimator);
        let transition_preimages =
            scale_estimate(self.trapdoor_preimage.clone(), shape.transition_preimage_chunk_count);
        let transition_stage =
            sequential_summaries(&[transition_target_building, transition_preimages]);
        let enc = sequential_summaries(&[
            checkpoint_sampling.clone(),
            parallel_summaries(&[initial_state.clone(), transition_stage.clone()]),
        ]);
        let dec = shape.estimate_online_input_injection(self.native_estimator);
        DiamondWEInputInjectionBenchEstimateParts { enc, dec }
    }

    fn final_preimage_summary_for_width(&self, width: usize) -> CircuitBenchSummary {
        scale_estimate(self.final_preimage_one_column.clone(), width)
    }

    fn estimate_storage(&self, shape: DiamondWEBenchShape) -> DiamondWEStorageEstimate {
        let input_injection_metadata_and_seed_bytes =
            shape.input_injection_metadata_and_seed_bytes();
        let input_injection_bytes = shape.input_injection_bytes();
        let we_preimage_bytes = shape.we_preimage_bytes();
        let total_bytes = input_injection_bytes.clone() + &we_preimage_bytes;

        debug!(
            ?input_injection_metadata_and_seed_bytes,
            ?input_injection_bytes,
            ?we_preimage_bytes,
            ?total_bytes,
            "estimated DiamondWE ciphertext storage"
        );

        DiamondWEStorageEstimate {
            input_injection_metadata_and_seed_bytes,
            input_injection_bytes,
            we_preimage_bytes,
            total_bytes,
        }
    }
}

#[derive(Debug, Clone)]
struct DiamondWEBenchShape {
    ring_dim: usize,
    witness_size: usize,
    instance_size: usize,
    modulus_digits: usize,
    modulus_bits: u16,
    state_row_size: usize,
    state_col_size: usize,
    b_public_col_size: usize,
    checkpoint_count: usize,
    final_checkpoint_count: usize,
    transition_matrix_count: usize,
    transition_preimage_chunk_count: usize,
    input_injection_transition_preimage_bytes: usize,
    online_level_state_counts: Vec<usize>,
    transition_chunk_col_lens: Vec<usize>,
    lookup_bridge_cols: Option<usize>,
    dec_lookup_bridge_cols: Option<usize>,
}

impl DiamondWEBenchShape {
    fn from_diamond_and_circuit<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        diamond: &DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        circuit: &PolyCircuit<M::P>,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        assert_eq!(circuit.num_output(), 1, "DiamondWE bench estimation requires one output");
        assert!(
            circuit.num_input() >= diamond.witness_size,
            "DiamondWE bench-estimation circuit input count must be at least witness_size"
        );
        assert_eq!(
            diamond.witness_size,
            diamond.injector.input_count * diamond.injector.batch_bits(),
            "DiamondWE witness_size must match the DiamondInjector bit input count"
        );
        let params = &diamond.injector.params;
        let ring_dim = params.ring_dimension() as usize;
        let modulus_digits = params.modulus_digits();
        let modulus_bits = params
            .modulus_bits()
            .try_into()
            .expect("DiamondWE modulus bits must fit in u16 for compact-byte estimates");
        let state_row_size =
            2usize.checked_mul(DIAMOND_SECRET_SIZE).expect("DiamondWE state row size overflow");
        let b_public_col_size = state_row_size
            .checked_mul(modulus_digits + 2)
            .expect("DiamondWE B public column count overflow");
        let state_col_size = b_public_col_size;
        let chunk_count = column_chunk_count(state_col_size);
        let transition_chunk_col_lens = (0..chunk_count)
            .map(|chunk_idx| column_chunk_bounds(state_col_size, chunk_idx).1)
            .collect::<Vec<_>>();
        let transition_chunk_bytes = transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                matrix_compact_bytes::<M>(params, state_col_size, col_len, modulus_bits)
            })
            .collect::<Vec<_>>();
        let transition_preimage_bytes_per_matrix = transition_chunk_bytes.iter().sum::<usize>();
        let mut transition_matrix_count = 0usize;
        let mut transition_preimage_chunk_count = 0usize;
        let mut online_level_state_counts = Vec::with_capacity(diamond.injector.input_count);
        for level in 1..=diamond.injector.input_count {
            let state_count = state_count_at_level(diamond, level);
            transition_matrix_count = transition_matrix_count
                .checked_add(
                    diamond
                        .injector
                        .base
                        .checked_mul(state_count)
                        .expect("DiamondWE transition matrix count overflow"),
                )
                .expect("DiamondWE transition matrix count overflow");
            transition_preimage_chunk_count = transition_preimage_chunk_count
                .checked_add(
                    diamond
                        .injector
                        .base
                        .checked_mul(state_count)
                        .and_then(|count| count.checked_mul(chunk_count))
                        .expect("DiamondWE transition chunk count overflow"),
                )
                .expect("DiamondWE transition chunk count overflow");
            online_level_state_counts.push(state_count);
        }
        let checkpoint_count = (0..=diamond.injector.input_count)
            .map(|level| state_count_at_level(diamond, level))
            .try_fold(0usize, |acc, count| acc.checked_add(count))
            .expect("DiamondWE checkpoint count overflow");
        let final_checkpoint_count = state_count_at_level(diamond, diamond.injector.input_count);
        let input_injection_transition_preimage_bytes = transition_preimage_bytes_per_matrix
            .checked_mul(transition_matrix_count)
            .expect("DiamondWE transition preimage byte count overflow");
        let lookup_bridge_cols = diamond.enc_lookup_base_matrix.as_ref().map(PolyMatrix::col_size);
        assert!(
            diamond.enc_lookup_evaluator_factory.is_none() || lookup_bridge_cols.is_some(),
            "DiamondWE bench estimation requires enc_lookup_base_matrix when enc_lookup_evaluator_factory is set"
        );
        let dec_lookup_bridge_cols =
            diamond.enc_lookup_evaluator_factory.as_ref().and(lookup_bridge_cols);

        Self {
            ring_dim,
            witness_size: diamond.witness_size,
            instance_size: circuit.num_input() - diamond.witness_size,
            modulus_digits,
            modulus_bits,
            state_row_size,
            state_col_size,
            b_public_col_size,
            checkpoint_count,
            final_checkpoint_count,
            transition_matrix_count,
            transition_preimage_chunk_count,
            input_injection_transition_preimage_bytes,
            online_level_state_counts,
            transition_chunk_col_lens,
            lookup_bridge_cols,
            dec_lookup_bridge_cols,
        }
    }

    fn ordinary_preimage_count(&self) -> usize {
        1usize.checked_add(self.witness_size).expect("DiamondWE ordinary preimage count overflow")
    }

    fn input_injection_metadata_and_seed_bytes(&self) -> BigUint {
        let metadata_estimate = 128usize;
        let p_epsilon_bytes = matrix_compact_bytes_for_shape(
            self.state_row_size / 2,
            self.state_col_size,
            self.ring_dim,
            self.modulus_bits,
        );
        BigUint::from(metadata_estimate + p_epsilon_bytes)
    }

    fn input_injection_public_checkpoint_bytes(&self) -> usize {
        matrix_compact_bytes_for_shape(
            self.state_row_size,
            self.b_public_col_size,
            self.ring_dim,
            self.modulus_bits,
        )
        .checked_mul(self.final_checkpoint_count)
        .expect("DiamondWE B checkpoint byte count overflow")
    }

    fn input_injection_bytes(&self) -> BigUint {
        self.input_injection_metadata_and_seed_bytes() +
            BigUint::from(self.input_injection_public_checkpoint_bytes()) +
            BigUint::from(self.input_injection_transition_preimage_bytes)
    }

    fn we_preimage_bytes(&self) -> BigUint {
        let ordinary_preimage_bytes = matrix_compact_bytes_for_shape(
            self.state_col_size,
            self.modulus_digits,
            self.ring_dim,
            self.modulus_bits,
        );
        let scalar_preimage_bytes = matrix_compact_bytes_for_shape(
            self.state_col_size,
            1,
            self.ring_dim,
            self.modulus_bits,
        );
        let lookup_preimage_bytes = self
            .lookup_bridge_cols
            .map(|lookup_cols| {
                matrix_compact_bytes_for_shape(
                    self.state_col_size,
                    lookup_cols,
                    self.ring_dim,
                    self.modulus_bits,
                )
            })
            .unwrap_or(0usize);

        BigUint::from(
            ordinary_preimage_bytes
                .checked_mul(self.ordinary_preimage_count())
                .expect("DiamondWE ordinary preimage byte count overflow"),
        ) + BigUint::from(
            scalar_preimage_bytes
                .checked_mul(2)
                .expect("DiamondWE scalar preimage byte count overflow"),
        ) + BigUint::from(lookup_preimage_bytes)
    }

    fn estimate_transition_target_building<NBE>(
        &self,
        native_estimator: &NBE,
    ) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator + Sync,
    {
        let per_transition_matrix = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                sequential_summaries(&[
                    native_estimator.estimate_row_parallel_matrix_product(
                        self.state_row_size,
                        self.state_row_size,
                        col_len,
                    ),
                    estimate_summary(
                        native_estimator
                            .estimate_vector_add(self.state_row_size.saturating_mul(col_len)),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        scale_summary(
            parallel_summaries(per_transition_matrix.as_slice()),
            self.transition_matrix_count,
        )
    }

    fn estimate_online_input_injection<NBE>(&self, native_estimator: &NBE) -> CircuitBenchSummary
    where
        NBE: DiamondIONativeBenchEstimator + Sync,
    {
        let per_state = self
            .transition_chunk_col_lens
            .iter()
            .map(|&col_len| {
                native_estimator.estimate_vector_matrix_product(self.state_col_size, col_len)
            })
            .collect::<Vec<_>>();
        let per_state = parallel_summaries(per_state.as_slice());
        let levels = self
            .online_level_state_counts
            .iter()
            .map(|&state_count| scale_summary(per_state.clone(), state_count))
            .collect::<Vec<_>>();
        sequential_summaries(levels.as_slice())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiamondWEStorageEstimate {
    input_injection_metadata_and_seed_bytes: BigUint,
    input_injection_bytes: BigUint,
    we_preimage_bytes: BigUint,
    total_bytes: BigUint,
}

#[derive(Debug, Clone, PartialEq)]
struct DiamondWEInputInjectionBenchEstimateParts {
    enc: CircuitBenchSummary,
    dec: CircuitBenchSummary,
}

#[derive(Debug, Clone)]
struct DiamondWEBenchUnitEstimates {
    trapdoor_checkpoint: CircuitBenchEstimate,
    trapdoor_preimage: CircuitBenchEstimate,
    final_preimage_one_column: CircuitBenchEstimate,
    bgg_public_key_sample: CircuitBenchEstimate,
    scalar_matrix_hash_sample: CircuitBenchEstimate,
}

impl DiamondWEBenchUnitEstimates {
    fn benchmark<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
        diamond: &DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
        iterations: usize,
    ) -> Self
    where
        M: PolyMatrix + Send + Sync + 'static,
        M::P: 'static,
        US: PolyUniformSampler<M = M> + Send + Sync,
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    {
        let iterations = iterations.max(1);
        let params = &diamond.injector.params;
        let state_row_size =
            2usize.checked_mul(DIAMOND_SECRET_SIZE).expect("DiamondWE state row size overflow");
        let modulus_digits = params.modulus_digits();
        let trapdoor_checkpoint = bench_estimate(iterations, || {
            let trap_sampler = TS::new(params, diamond.injector.trapdoor_sigma);
            let (trapdoor, matrix) = trap_sampler.trapdoor(params, state_row_size);
            black_box((TS::trapdoor_to_bytes(&trapdoor), matrix.to_compact_bytes()))
        });

        let trap_sampler = TS::new(params, diamond.injector.trapdoor_sigma);
        let (trapdoor, public_matrix) = trap_sampler.trapdoor(params, state_row_size);
        let one_column_target = M::zero(params, state_row_size, 1);
        let preimage_one_column = bench_estimate(iterations, || {
            let preimage =
                trap_sampler.preimage(params, &trapdoor, &public_matrix, &one_column_target);
            black_box(preimage.to_compact_bytes())
        });
        let transition_preimage = scale_independent_estimate(
            preimage_one_column.clone(),
            column_chunk_bounds(
                state_row_size
                    .checked_mul(modulus_digits + 2)
                    .expect("DiamondWE benchmark state col overflow"),
                0,
            )
            .1,
        );
        let bgg_public_key_sample = bench_estimate(iterations, || {
            BGGPublicKeySampler::<[u8; 32], HS>::new([0x73u8; 32], DIAMOND_SECRET_SIZE).sample(
                params,
                b"diamond_we_bench_bgg_public_key",
                &[],
            )
        });
        let scalar_matrix_hash_sample = bench_estimate(iterations, || {
            let matrix = HS::new().sample_hash(
                params,
                [0x74u8; 32],
                b"diamond_we_bench_scalar_matrix",
                1,
                1,
                DistType::FinRingDist,
            );
            black_box(matrix.to_compact_bytes())
        });
        Self {
            trapdoor_checkpoint,
            trapdoor_preimage: transition_preimage,
            final_preimage_one_column: preimage_one_column,
            bgg_public_key_sample,
            scalar_matrix_hash_sample,
        }
    }
}

fn state_count_at_level<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>(
    diamond: &DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
    level: usize,
) -> usize
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    1usize
        .checked_add(
            level
                .checked_mul(diamond.injector.batch_bits())
                .expect("DiamondWE expanded state count overflow"),
        )
        .expect("DiamondWE expanded state count overflow")
}

fn matrix_compact_bytes<M>(
    params: &<M::P as Poly>::Params,
    nrow: usize,
    ncol: usize,
    modulus_bits: u16,
) -> usize
where
    M: PolyMatrix,
{
    matrix_compact_bytes_for_shape(nrow, ncol, params.ring_dimension() as usize, modulus_bits)
}

fn matrix_compact_bytes_for_shape(
    nrow: usize,
    ncol: usize,
    ring_dim: usize,
    modulus_bits: u16,
) -> usize {
    let coeff_count =
        nrow.checked_mul(ncol).and_then(|count| count.checked_mul(ring_dim)).unwrap_or(usize::MAX);
    let payload = coeff_count.checked_mul(modulus_bits as usize).unwrap_or(usize::MAX).div_ceil(8);
    payload + 128
}

fn bench_estimate<R, F>(iterations: usize, mut op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    let measurement = benchmark_gate_operation(iterations, || op());
    CircuitBenchEstimate::new(measurement.time, measurement.time)
        .with_peak_vram(measurement.peak_vram)
}

fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    let summary = CircuitBenchSummary::from_nanos(
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
    );
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn scale_estimate(estimate: CircuitBenchEstimate, count: usize) -> CircuitBenchSummary {
    scale_summary(estimate_summary(estimate), count)
}

fn scale_summary(summary: CircuitBenchSummary, count: usize) -> CircuitBenchSummary {
    scale_independent_summary(summary, count)
}

fn sequential_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts
        .par_iter()
        .map(|part| part.total_time.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let latency = parts.par_iter().map(|part| part.latency).sum::<f64>();
    let max_parallelism = parts
        .par_iter()
        .map(|part| part.max_parallelism.clone())
        .reduce(BigUint::zero, |left, right| left.max(right));
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.par_iter().map(|part| part.peak_vram).max().unwrap_or(0))
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

fn parallel_summaries(parts: &[CircuitBenchSummary]) -> CircuitBenchSummary {
    let total_time = parts
        .par_iter()
        .map(|part| part.total_time.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let latency = parts.par_iter().map(|part| part.latency).reduce(|| 0.0f64, f64::max);
    let max_parallelism = parts
        .par_iter()
        .map(|part| part.max_parallelism.clone())
        .reduce(BigUint::zero, |left, right| left + right);
    let summary = CircuitBenchSummary::from_nanos(total_time, latency, max_parallelism);
    #[cfg(feature = "gpu")]
    {
        summary.with_peak_vram(parts.par_iter().map(|part| part.peak_vram).sum::<usize>())
    }
    #[cfg(not(feature = "gpu"))]
    {
        summary
    }
}

#[cfg(test)]
mod tests {
    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        bench_estimator::SampleAuxBenchEstimate,
        func_enc::NoCircuitEvaluator,
        input_injector::DiamondInjector,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };

    #[derive(Debug)]
    struct DummyBggBenchEstimator;

    impl BenchEstimator<BggPublicKey<DCRTPolyMatrix>> for DummyBggBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(1.0, 1.0)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(2.0, 2.0)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(3.0, 3.0)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(4.0, 4.0)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(5.0, 5.0)
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(6.0, 6.0)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(7.0, 7.0)
        }

        fn estimate_slot_reduce(
            &self,
            input_count: usize,
            _num_slots: usize,
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(8.0 * input_count as f64, 8.0)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(9.0, 9.0)
        }
    }

    impl BenchEstimator<BggEncoding<DCRTPolyMatrix>> for DummyBggBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(1.5, 1.5)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(2.5, 2.5)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(3.5, 3.5)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(4.5, 4.5)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(5.5, 5.5)
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(6.5, 6.5)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(7.5, 7.5)
        }

        fn estimate_slot_reduce(
            &self,
            input_count: usize,
            _num_slots: usize,
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(8.5 * input_count as f64, 8.5)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(9.5, 9.5)
        }
    }

    impl PublicKeyAuxBenchEstimator<DCRTPoly> for DummyBggBenchEstimator {
        fn estimate_public_lut_sample_aux_matrices_for_circuit(
            &self,
            _params: &DCRTPolyParams,
            circuit: &PolyCircuit<DCRTPoly>,
        ) -> SampleAuxBenchEstimate {
            let lut_gates = circuit
                .count_gates_by_type_vec()
                .get(&crate::circuit::PolyGateKind::PubLut)
                .copied()
                .unwrap_or(0);
            SampleAuxBenchEstimate {
                total_time: lut_gates as f64 * 11.0,
                latency: if lut_gates == 0 { 0.0 } else { 11.0 },
                compact_bytes: BigUint::from(lut_gates),
            }
        }
    }

    #[derive(Debug)]
    struct DummyNativeBenchEstimator;

    impl DiamondIONativeBenchEstimator for DummyNativeBenchEstimator {
        fn estimate_poly_vector_mul(&self, vector_len: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(vector_len as f64, 1.0)
                .with_max_parallelism(vector_len as u128)
        }

        fn estimate_vector_inner_product(&self, vector_len: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(vector_len as f64 * 2.0, 2.0)
                .with_max_parallelism(vector_len as u128)
        }

        fn estimate_vector_add(&self, vector_len: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(vector_len as f64, 1.0)
                .with_max_parallelism(vector_len as u128)
        }

        fn estimate_vector_sub(&self, vector_len: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(vector_len as f64, 1.0)
                .with_max_parallelism(vector_len as u128)
        }
    }

    fn diamond_we(
        params: DCRTPolyParams,
        witness_size: usize,
    ) -> DiamondWE<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    > {
        assert_eq!(witness_size % 2, 0, "test witness_size must use whole input digits");
        let injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params, witness_size / 2, 4, 2, 4.578, 0.0);
        let dir = tempdir().expect("temporary DiamondWE benchmark directory should be created");
        DiamondWE::new(injector, witness_size, dir.path(), b"diamond_we_bench_test".to_vec())
    }

    fn estimator<'a>(
        bgg: &'a DummyBggBenchEstimator,
        native: &'a DummyNativeBenchEstimator,
    ) -> DiamondWEBenchEstimator<
        'a,
        DummyBggBenchEstimator,
        DummyBggBenchEstimator,
        DummyNativeBenchEstimator,
    > {
        let unit = |seconds| CircuitBenchEstimate::new(seconds, seconds);
        DiamondWEBenchEstimator::with_unit_costs(
            bgg,
            bgg,
            native,
            unit(1.0),
            unit(2.0),
            unit(3.0),
            unit(4.0),
            unit(5.0),
        )
    }

    fn circuit(input_count: usize) -> PolyCircuit<DCRTPoly> {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(input_count).to_vec();
        let output = circuit.or_gate(inputs[1], inputs[input_count - 1]);
        circuit.output(vec![output]);
        circuit
    }

    #[test]
    fn test_diamond_we_bench_estimate_reports_enc_and_dec_work() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let we = diamond_we(params, witness_size);
        let bgg = DummyBggBenchEstimator;
        let native = DummyNativeBenchEstimator;
        let estimator = estimator(&bgg, &native);
        let estimate = estimator.estimate(&we, &circuit(witness_size + 1));

        assert!(!estimate.enc.total_time.is_zero());
        assert!(!estimate.dec.total_time.is_zero());
        assert!(!estimate.enc_input_injection.total_time.is_zero());
        assert!(!estimate.dec_input_injection.total_time.is_zero());
        assert!(!estimate.ciphertext_bytes.is_zero());
        assert!(!estimate.input_injection_bytes.is_zero());
        assert!(!estimate.we_preimage_bytes.is_zero());
        assert_eq!(
            estimate.ciphertext_bytes,
            estimate.input_injection_bytes.clone() + estimate.we_preimage_bytes.clone()
        );
        assert!(!estimate.enc.max_parallelism.is_zero());
        assert!(!estimate.dec.max_parallelism.is_zero());
    }

    #[test]
    fn test_diamond_we_bench_estimate_scales_with_instance_count() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let we = diamond_we(params, witness_size);
        let bgg = DummyBggBenchEstimator;
        let native = DummyNativeBenchEstimator;
        let estimator = estimator(&bgg, &native);
        let one_instance = estimator.estimate(&we, &circuit(witness_size + 1));
        let two_instances = estimator.estimate(&we, &circuit(witness_size + 2));

        assert!(two_instances.enc.total_time > one_instance.enc.total_time);
        assert!(two_instances.dec.total_time > one_instance.dec.total_time);
        assert_eq!(two_instances.ciphertext_bytes, one_instance.ciphertext_bytes);
    }

    #[test]
    fn test_diamond_we_bench_estimate_counts_lookup_bridge_width() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let mut narrow = diamond_we(params.clone(), witness_size);
        let mut wide = diamond_we(params.clone(), witness_size);
        narrow.enc_lookup_base_matrix = Some(DCRTPolyMatrix::zero(&params, 1, 1));
        wide.enc_lookup_base_matrix = Some(DCRTPolyMatrix::zero(&params, 1, 3));
        let bgg = DummyBggBenchEstimator;
        let native = DummyNativeBenchEstimator;
        let estimator = estimator(&bgg, &native);
        let narrow_estimate = estimator.estimate(&narrow, &circuit(witness_size + 1));
        let wide_estimate = estimator.estimate(&wide, &circuit(witness_size + 1));

        assert!(wide_estimate.enc.total_time > narrow_estimate.enc.total_time);
        assert_eq!(wide_estimate.dec.total_time, narrow_estimate.dec.total_time);
        assert!(wide_estimate.we_preimage_bytes > narrow_estimate.we_preimage_bytes);
        assert!(wide_estimate.ciphertext_bytes > narrow_estimate.ciphertext_bytes);
    }

    #[test]
    fn test_diamond_we_ciphertext_bytes_scale_with_witness_size() {
        let params = DCRTPolyParams::default();
        let bgg = DummyBggBenchEstimator;
        let native = DummyNativeBenchEstimator;
        let estimator = estimator(&bgg, &native);
        let small_witness_size = 2;
        let large_witness_size = 4;
        let small = diamond_we(params.clone(), small_witness_size);
        let large = diamond_we(params.clone(), large_witness_size);
        let small_estimate = estimator.estimate(&small, &circuit(small_witness_size + 1));
        let large_estimate = estimator.estimate(&large, &circuit(large_witness_size + 1));

        assert!(large_estimate.we_preimage_bytes > small_estimate.we_preimage_bytes);
        assert!(large_estimate.input_injection_bytes > small_estimate.input_injection_bytes);
        assert!(large_estimate.ciphertext_bytes > small_estimate.ciphertext_bytes);
    }
}
