#[cfg(feature = "gpu")]
mod gpu;

mod context;
mod encoding;
mod poly;

#[cfg(test)]
mod tests;

use crate::{
    circuit::{BatchedWire, PolyCircuit, SubCircuitParamKind, SubCircuitParamValue, gate::GateId},
    gadgets::conv_mul::{
        negacyclic_conv_mul_right_decomposed_term_many_subcircuit, negacyclic_conv_mul_right_sparse,
    },
    lookup::PublicLut,
    poly::{Poly, PolyParams},
    utils::{mod_inverse, round_div},
};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::{marker::PhantomData, sync::Arc};
use tracing::debug;

pub use encoding::{
    encode_nested_rns_poly, encode_nested_rns_poly_compact_bytes,
    encode_nested_rns_poly_compact_bytes_with_offset, encode_nested_rns_poly_with_offset,
    nested_rns_gadget_decomposed, nested_rns_gadget_vector,
};
use encoding::sample_crt_primes;

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
    lut_mod_p_ids: Vec<usize>,
    lut_x_to_y_ids: Vec<usize>,
    lut_x_to_real_ids: Vec<usize>,
    lut_real_to_v_id: usize,
    add_without_reduce_id: usize,
    sub_with_trace_offsets_id: usize,
    lazy_reduce_id: usize,
    decomposition_terms_id: usize,
    gadget_decompose_id: usize,
    full_reduce_id: usize,
    full_reduce_bindings: Vec<Vec<SubCircuitParamValue>>,
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
    full_reduce_id: usize,
    mul_lazy_reduce_id: usize,
    mul_right_sparse_id: usize,
}

#[derive(Debug, Clone)]
/// Circuit-level nested-RNS polynomial representation.
///
/// `inner` stores one batched p-residue vector per active q-level. `max_plaintexts` and
/// `p_max_traces` are conservative metadata carried alongside the wires so later helpers know when
/// a lazy reduce or full reduce is required without changing the arithmetic semantics.
pub struct NestedRnsPoly<P: Poly> {
    pub ctx: Arc<NestedRnsPolyContext>,
    pub inner: Vec<BatchedWire>,
    pub level_offset: usize,
    pub enable_levels: Option<usize>,
    pub max_plaintexts: Vec<BigUint>,
    pub(crate) p_max_traces: Vec<BigUint>,
    _p: PhantomData<P>,
}
