pub mod carry_montgomery;
pub mod nested_rns;

use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{fmt::Debug, hash::Hash, sync::Arc};

pub use carry_montgomery::*;
pub use nested_rns::*;

/// A nonempty contiguous window in the context's q-CRT basis.
///
/// Arithmetic gadgets use only the towers in `[offset, offset + depth)`. Packed
/// representations allocate lanes for exactly these towers; towers outside the window do not
/// occupy physical slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrtWindow {
    pub offset: usize,
    pub depth: usize,
}

impl CrtWindow {
    pub fn new(offset: usize, depth: usize, total_depth: usize) -> Self {
        assert!(depth > 0, "CRT window depth must be positive");
        let end = offset.checked_add(depth).expect("CRT window end overflow");
        assert!(
            end <= total_depth,
            "CRT window [{offset}, {end}) exceeds total depth {total_depth}"
        );
        Self { offset, depth }
    }

    pub fn full(total_depth: usize) -> Self {
        Self::new(0, total_depth, total_depth)
    }

    pub fn end(self) -> usize {
        self.offset + self.depth
    }

    pub fn physical_slots(self, coefficient_slots: usize) -> usize {
        coefficient_slots.checked_mul(self.depth).expect("packed CRT slot count overflow")
    }
}

pub trait ModularArithmeticContext<P: Poly>: Clone + Debug + Send + Sync + 'static {
    fn q_moduli_depth(&self) -> usize;

    fn decomposition_len(&self) -> usize;

    fn q_level_row_width(&self) -> usize;

    fn randomizer_decomposition_bound(&self) -> u64;

    fn decomposition_term_bound(&self, term_idx: usize) -> BigUint;

    fn plaintext_capacity_bound(&self) -> BigUint;

    fn validate_window(&self, window: CrtWindow) -> CrtWindow {
        CrtWindow::new(window.offset, window.depth, self.q_moduli_depth())
    }

    fn gadget_len(&self, window: CrtWindow) -> usize {
        self.validate_window(window).depth * self.decomposition_len()
    }
}

pub trait ModularArithmeticGadget<P: Poly>: Clone + Debug + Send + Sync + 'static {
    type Context: ModularArithmeticContext<P>;

    fn context(&self) -> &Arc<Self::Context>;

    fn crt_window(&self) -> CrtWindow;

    fn max_plaintexts(&self) -> &[BigUint];

    fn p_max_traces(&self) -> &[BigUint];

    fn num_coefficient_slots(&self) -> usize {
        1
    }

    fn input(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self;

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
        num_coefficient_slots: usize,
        window: CrtWindow,
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

    fn normalized_metadata(ctx: &Self::Context, window: CrtWindow) -> Self::Metadata;

    fn input_with_planner_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
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
        window: CrtWindow,
    ) -> M;

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        window: CrtWindow,
    ) -> M;

    fn gadget_constant_coeffs<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        window: CrtWindow,
    ) -> Vec<BigUint> {
        Self::gadget_matrix::<M>(params, ctx, window)
            .get_row(0)
            .into_par_iter()
            .map(|entry| entry.coeffs_biguints()[0].clone())
            .collect()
    }

    fn gadget_decomposed_constant_tower_coeffs<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        constant: BigUint,
        window: CrtWindow,
    ) -> Vec<Vec<u64>> {
        let window = ctx.validate_window(window);
        let active_levels = window.depth;
        let active_q_moduli = params
            .to_crt()
            .0
            .into_iter()
            .skip(window.offset)
            .take(active_levels)
            .collect::<Vec<_>>();
        let scaled_poly = P::from_biguint_to_constant(params, constant);
        let decomposed = Self::gadget_decomposed::<M>(
            params,
            ctx,
            &M::from_poly_vec_column(params, vec![scaled_poly]),
            window,
        );
        let (rows, cols) = decomposed.size();
        assert_eq!(cols, 1, "gadget decomposition of a constant must have one column");
        assert_eq!(
            rows,
            active_levels * ctx.decomposition_len(),
            "gadget decomposition row count mismatch"
        );
        decomposed
            .get_column(0)
            .into_iter()
            .map(|entry| {
                let coeff = entry.coeffs_biguints()[0].clone();
                active_q_moduli
                    .iter()
                    .copied()
                    .map(|q_i| {
                        (&coeff % BigUint::from(q_i))
                            .to_u64()
                            .expect("gadget decomposition residue must fit in u64")
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn gadget_decomposition_norm_bound(ctx: &Self::Context, window: CrtWindow) -> BigUint;

    fn randomizer_decomposition_norm_bound(ctx: &Self::Context, window: CrtWindow) -> BigUint {
        Self::gadget_decomposition_norm_bound(ctx, window)
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
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
