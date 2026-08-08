//! Lane-packed nested-RNS arithmetic.
//!
//! Every p-residue wire uses coefficient-major physical slots
//! `slot(coefficient, q_level) = coefficient * q_moduli_depth + q_level`. Arithmetic and LUT
//! helpers preserve q-level lanes; only reconstruction and modulus-basis conversion aggregate or
//! move lanes. Inactive lanes are exact zero.
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

use crate::{
    circuit::{
        BatchedWire, LutExpr, PolyCircuit, PublicLutProgram, SubCircuitParamSpec,
        SubCircuitParamValue, gate::GateId,
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
    encode_nested_rns_poly, encode_nested_rns_poly_compact_bytes,
    encode_nested_rns_poly_compact_bytes_with_offset, encode_nested_rns_poly_with_offset,
    minimum_p_moduli_bits, nested_rns_gadget_decomposed, nested_rns_gadget_vector,
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
    mul_lazy_reduce_id: usize,
    mul_right_sparse_id: usize,
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
    mul_lazy_reduce_id: usize,
    mul_right_sparse_id: usize,
}

#[derive(Debug, Clone)]
/// Circuit-level nested-RNS polynomial representation.
///
/// `inner` stores the p-residue wires once. Each wire uses coefficient-major slots
/// `slot(c, level) = c * q_moduli_depth + level`; inactive q-level lanes are exact zero.
/// `max_plaintexts` and `p_max_traces` remain active-window-local metadata.
pub struct NestedRnsPoly<P: Poly> {
    pub ctx: Arc<NestedRnsPolyContext>,
    pub inner: BatchedWire,
    pub num_coefficient_slots: usize,
    pub level_offset: usize,
    pub enable_levels: Option<usize>,
    pub max_plaintexts: Vec<BigUint>,
    pub(crate) p_max_traces: Vec<BigUint>,
    _p: PhantomData<P>,
}
