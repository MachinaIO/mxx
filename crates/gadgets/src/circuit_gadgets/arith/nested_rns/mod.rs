//! Lane-packed nested-RNS arithmetic.
//!
//! Every p-residue wire uses a compact coefficient-major active-window layout
//! `slot(coefficient, local_level) = coefficient * active_crt_depth + local_level`. Towers outside
//! the active q-window occupy no physical slots. Arithmetic and LUT helpers preserve local tower
//! lanes; only reconstruction and modulus-basis conversion aggregate or move lanes.
//!
//! Per-lane constants, masks, full reduction, and non-rotation slot transfers are represented as
//! identity slot-transfer gates. Under BGG lowering these require slot-transfer preprocessing
//! artifacts. Reconstruction adds one explicit transfer per active q-level, and convolution adds
//! compact `RepeatedLanes` transfers; coefficient rotations remain artifact-free rotations.

#[cfg(feature = "gpu")]
mod gpu;

mod context;
mod decomposed_mul;
mod encoding;
mod poly;

use super::CrtWindow;

use crate::{
    circuit::{
        BatchedWire, LutExpr, PolyCircuit, PublicLutProgram, SubCircuitInputMaxPlaintextNormRange,
        SubCircuitParamSpec, SubCircuitParamValue, gate::GateId,
    },
    circuit_gadgets::conv_mul::{
        negacyclic_conv_mul_right_decomposed_term_many_subcircuit, negacyclic_conv_mul_right_sparse,
    },
    poly::{Poly, PolyParams},
    utils::mod_inverse,
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{marker::PhantomData, sync::Arc};
use tracing::debug;

use encoding::sample_crt_primes;
pub use encoding::{
    encode_nested_rns_poly, encode_nested_rns_poly_compact_bytes, minimum_p_moduli_bits,
    nested_rns_gadget_decomposed, nested_rns_gadget_vector,
};

pub const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;

#[derive(Debug, Clone)]
/// Precomputed constants, LUT ids, and helper sub-circuit ids for one nested-RNS configuration.
///
/// `NestedRnsPoly<P>` operations never derive these values on the fly. Instead, `setup` computes
/// every modulus-dependent constant once and stores the result here so later arithmetic helpers can
/// remain purely structural and behavior-preserving.
pub struct NestedRnsPolyContext {
    pub p_moduli_bits: usize,
    pub max_unreduced_muls: usize,
    pub scale: u64,
    pub p_moduli: Vec<u64>,
    q_moduli: Vec<u64>,
    pub q_moduli_depth: usize,
    p_max: u64,
    lut_mod_p_max_map_size: BigUint,
    p_full: BigUint,
    p_over_pis: Vec<BigUint>,
    gadget_values: Vec<Vec<BigUint>>,
    pub full_reduce_max_plaintexts: Vec<BigUint>,
    add_without_reduce_id: usize,
    sub_with_trace_offsets_id: usize,
    lazy_reduce_id: usize,
    decomposition_terms_id: usize,
    gadget_decompose_id: usize,
}

#[derive(Debug, Clone, Copy)]
/// Registry of helper sub-circuit ids installed by `NestedRnsPolyContext::setup`.
///
/// The fields mirror the distinct arithmetic kernels used later by `NestedRnsPoly<P>` methods.
/// Keeping them grouped makes it clear which context-owned helper each operation dispatches to.
struct NestedRnsRegisteredSubcircuitIds {
    add_without_reduce_id: usize,
    sub_with_trace_offsets_id: usize,
    lazy_reduce_id: usize,
    decomposition_terms_id: usize,
    gadget_decompose_id: usize,
}

#[derive(Debug, Clone)]
/// Circuit-level nested-RNS polynomial representation.
///
/// `inner` stores the p-residue wires once. Each wire uses compact coefficient-major slots
/// `slot(c, local_level) = c * window.depth + local_level`. `max_plaintexts` and `p_max_traces`
/// are indexed by the same active-window-local level.
pub struct NestedRnsPoly<P: Poly> {
    pub ctx: Arc<NestedRnsPolyContext>,
    pub inner: BatchedWire,
    pub num_coefficient_slots: usize,
    pub window: CrtWindow,
    pub max_plaintexts: Vec<BigUint>,
    pub(crate) p_max_traces: Vec<BigUint>,
    _p: PhantomData<P>,
}

/// Reconstruction output whose authoritative values live only at q1 anchors.
///
/// For a coefficient-major packed value with `D` q-level lanes, coefficient `c`
/// is available exclusively at physical slot `c * D`.  The other lanes are an
/// implementation byproduct of the anchor reduction and must not be consumed as
/// reconstructed coefficients.  This deliberately differs from
/// [`NestedRnsPoly::reconstruct`], which produces a value valid in every lane.
#[derive(Debug, Clone, Copy)]
pub struct Q1AnchorReconstruction {
    anchor_wire: GateId,
    coefficient_slots: usize,
    q_moduli_depth: usize,
}

impl Q1AnchorReconstruction {
    /// Returns the circuit wire carrying the anchor-only reconstruction.
    pub fn anchor_wire(self) -> GateId {
        self.anchor_wire
    }

    /// Returns the sole authoritative physical slot for coefficient `coefficient`.
    pub fn anchor_slot(self, coefficient: usize) -> usize {
        assert!(
            coefficient < self.coefficient_slots,
            "q1 anchor coefficient index {coefficient} exceeds {} slots",
            self.coefficient_slots
        );
        coefficient * self.q_moduli_depth
    }
}
