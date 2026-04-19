pub mod carry_montgomery;
pub mod nested_rns;

use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    matrix::PolyMatrix,
    poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{fmt::Debug, hash::Hash, sync::Arc};

pub use carry_montgomery::*;
pub use nested_rns::*;

pub trait ModularArithmeticContext<P: Poly>: Clone + Debug + Send + Sync + 'static {
    fn register_local_in(&self, circuit: &mut PolyCircuit<P>) -> Self;

    fn register_shared_in(
        &self,
        source_circuit: &PolyCircuit<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn register_shared_subcircuits_in(
        &self,
        source_circuit: &PolyCircuit<P>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.register_shared_in(source_circuit, circuit)
    }

    fn q_moduli_depth(&self) -> usize;

    fn decomposition_len(&self) -> usize;

    fn q_level_row_width(&self) -> usize;

    fn randomizer_decomposition_bound(&self) -> u64;

    fn decomposition_term_bound(&self, term_idx: usize) -> BigUint;

    fn plaintext_capacity_bound(&self) -> BigUint;

    fn active_levels(&self, enable_levels: Option<usize>, level_offset: Option<usize>) -> usize {
        let level_offset = level_offset.unwrap_or(0);
        let levels = enable_levels.unwrap_or(self.q_moduli_depth().saturating_sub(level_offset));
        assert!(
            level_offset + levels <= self.q_moduli_depth(),
            "active range exceeds q_moduli_depth: level_offset={level_offset}, enable_levels={levels}, q_moduli_depth={}",
            self.q_moduli_depth()
        );
        levels
    }

    fn gadget_len(&self, enable_levels: Option<usize>, level_offset: Option<usize>) -> usize {
        self.active_levels(enable_levels, level_offset) * self.decomposition_len()
    }
}

pub trait ModularArithmeticGadget<P: Poly>: Clone + Debug + Send + Sync + 'static {
    type Context: ModularArithmeticContext<P>;

    fn context(&self) -> &Arc<Self::Context>;

    fn level_offset(&self) -> usize;

    fn enable_levels(&self) -> Option<usize>;

    fn max_plaintexts(&self) -> &[BigUint];

    fn p_max_traces(&self) -> &[BigUint];

    fn input(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn input_like_with_ctx(
        template: &Self,
        ctx: Arc<Self::Context>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            template.enable_levels(),
            Some(template.level_offset()),
            template.max_plaintexts().to_vec(),
            template.p_max_traces().to_vec(),
            circuit,
        )
    }

    fn active_q_moduli(&self) -> Vec<u64>;

    fn flatten(&self) -> Vec<BatchedWire>;

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self;

    fn q_level_row_batch(&self, q_idx: usize) -> BatchedWire;

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn flat_output_size(&self) -> usize {
        self.flatten().len()
    }

    fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self;

    fn mul_right_sparse(
        &self,
        other: &Self,
        _rhs_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.mul(other, circuit)
    }

    fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self;

    fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self;

    fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self;

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId;
}

#[derive(Debug, Clone)]
pub struct BinaryPlannerResult<K, M> {
    pub cache_key: K,
    pub output_metadata: M,
}

pub trait ModularArithmeticPlanner<P: Poly>: ModularArithmeticGadget<P> {
    type Metadata: Clone + Debug + Send + Sync + 'static;
    type AddPlanKey: Clone + Debug + Eq + Hash + Send + Sync + 'static;
    type SubPlanKey: Clone + Debug + Eq + Hash + Send + Sync + 'static;

    fn metadata(entry: &Self) -> Self::Metadata;

    fn normalized_metadata(
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> Self::Metadata;

    fn input_with_planner_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        metadata: &Self::Metadata,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn from_flat_outputs_with_planner_metadata(
        template: &Self,
        outputs: &[GateId],
        metadata: &Self::Metadata,
    ) -> Self;

    fn compute_add_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::AddPlanKey, Self::Metadata>;

    fn compute_sub_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::SubPlanKey, Self::Metadata>;

    fn normalize_mul_input(entry: &Self, circuit: &mut PolyCircuit<P>) -> Self;
}

pub trait DecomposeArithmeticGadget<P: Poly>: ModularArithmeticGadget<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M;

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M;

    fn gadget_decomposition_norm_bound(
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> BigUint;

    fn randomizer_decomposition_norm_bound(
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> BigUint {
        Self::gadget_decomposition_norm_bound(ctx, enable_levels, level_offset)
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self>;

    fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self>;

    fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId);

    fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self>;

    fn mul_rows_with_decomposed_rhs(
        params: &P::Params,
        lhs_row0: &[Self],
        lhs_row1: &[Self],
        rhs_top: &Self,
        rhs_bottom: &Self,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Self; 2] {
        assert_eq!(lhs_row0.len(), lhs_row1.len(), "Ring-GSW row lengths must match");
        assert!(
            lhs_row0.len().is_multiple_of(2),
            "Ring-GSW row width {} must be even",
            lhs_row0.len()
        );
        let gadget_len = lhs_row0.len() / 2;
        let lhs_row0_top = &lhs_row0[..gadget_len];
        let lhs_row0_bottom = &lhs_row0[gadget_len..];
        let lhs_row1_top = &lhs_row1[..gadget_len];
        let lhs_row1_bottom = &lhs_row1[gadget_len..];
        let top_products = rhs_top.conv_mul_right_decomposed_many(
            params,
            &[lhs_row0_top, lhs_row1_top],
            num_slots,
            circuit,
        );
        let bottom_products = rhs_bottom.conv_mul_right_decomposed_many(
            params,
            &[lhs_row0_bottom, lhs_row1_bottom],
            num_slots,
            circuit,
        );
        [
            top_products[0].add(&bottom_products[0], circuit),
            top_products[1].add(&bottom_products[1], circuit),
        ]
    }
}

pub fn flatten_gadget_entries<P, A>(entries: &[A]) -> Vec<BatchedWire>
where
    P: Poly,
    A: ModularArithmeticGadget<P>,
{
    entries
        .par_iter()
        .map(|entry| entry.flatten())
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}
