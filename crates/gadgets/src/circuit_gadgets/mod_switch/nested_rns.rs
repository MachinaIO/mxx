//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues correspond to the contiguous active q-level window `q_{level_offset}, ...,
//!   q_{level_offset + active_levels - 1}`. `self.inner` stores one p-residue batch whose physical
//!   lanes use `slot(c, level) = c * q_moduli_depth + level`; the active window is metadata and
//!   every lane outside it is literal zero.
//! - This module supports both CKKS special-prime insertion/removal at the prefix side of the
//!   active window and one-level suffix removal for rescaling.
//!
//! Output layout conventions:
//! - `mod_up_levels(k)` prepends the contiguous `k`-modulus block immediately before the active
//!   window.
//! - `mod_down_levels(k)` removes the contiguous prefix block of size `k` from the active window.
//! - `mod_down_one_level()` removes the final active suffix modulus to support rescaling.
//!
//! All per-modulus arithmetic is expressed by composing existing `NestedRnsPoly` operations
//! rather than directly manipulating q-level residues as raw integers.

use crate::{
    circuit::PolyCircuit, circuit_gadgets::arith::NestedRnsPoly, poly::Poly, utils::mod_inverse,
};
use num_bigint::BigUint;

fn reduce_nested_rns_terms_pairwise<P, F>(
    mut current_layer: Vec<NestedRnsPoly<P>>,
    circuit: &mut PolyCircuit<P>,
    mut combine: F,
) -> NestedRnsPoly<P>
where
    P: Poly,
    F: FnMut(&NestedRnsPoly<P>, &NestedRnsPoly<P>, &mut PolyCircuit<P>) -> NestedRnsPoly<P>,
{
    assert!(
        !current_layer.is_empty(),
        "pairwise reduction requires at least one NestedRnsPoly term"
    );
    while current_layer.len() > 1 {
        let mut next_layer = Vec::with_capacity((current_layer.len() + 1) / 2);
        let mut iter = current_layer.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next_layer.push(combine(&left, &right, circuit));
            } else {
                next_layer.push(left);
            }
        }
        current_layer = next_layer;
    }
    current_layer.pop().expect("pairwise reduction must leave one term")
}

fn product_modulus(moduli: &[u64]) -> BigUint {
    moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
}

fn modular_product_except(moduli: &[u64], skip_idx: usize, modulus: u64) -> u64 {
    assert!(modulus > 0, "modulus must be non-zero");
    let modulus_u128 = modulus as u128;
    moduli
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != skip_idx)
        .fold(1u128, |acc, (_, &value)| (acc * (value % modulus) as u128) % modulus_u128) as u64
}

fn modular_product(moduli: &[u64], modulus: u64) -> u64 {
    assert!(modulus > 0, "modulus must be non-zero");
    let modulus_u128 = modulus as u128;
    moduli.iter().fold(1u128, |acc, &value| (acc * (value % modulus) as u128) % modulus_u128) as u64
}

/// Upper-bound the total quotient contribution introduced by the explicit `full_reduce()` calls
/// inside [`NestedRnsPoly::conv_between_levels`].
pub fn full_reduce_error_quotient_by_conv(
    source_moduli: &[u64],
    full_reduce_max_plaintexts: &[BigUint],
) -> BigUint {
    if source_moduli.is_empty() {
        return BigUint::ZERO;
    }
    assert_eq!(
        source_moduli.len(),
        full_reduce_max_plaintexts.len(),
        "full_reduce_max_plaintexts must correspond to source_moduli for Conv quotient bounds"
    );
    full_reduce_max_plaintexts
        .iter()
        .zip(source_moduli.iter())
        .map(|(max_plaintext, q_i)| max_plaintext / BigUint::from(*q_i))
        .sum::<BigUint>()
}

/// Upper-bound the total quotient error contributed by [`NestedRnsPoly::conv_between_levels`].
///
/// This adds the explicit `full_reduce()` quotient term and the unsigned Conv carry term, one per
/// source modulus.
pub fn conv_error_quotient_upper_bound(
    source_moduli: &[u64],
    full_reduce_max_plaintexts: &[BigUint],
) -> BigUint {
    full_reduce_error_quotient_by_conv(source_moduli, full_reduce_max_plaintexts) +
        BigUint::from(source_moduli.len())
}

/// Upper-bound the reconstructed `mod_up_levels()` error for a source basis.
pub fn mod_up_reconstruct_error_upper_bound(
    source_moduli: &[u64],
    full_reduce_max_plaintexts: &[BigUint],
) -> BigUint {
    conv_error_quotient_upper_bound(source_moduli, full_reduce_max_plaintexts) *
        product_modulus(source_moduli)
}

/// Upper-bound the reconstructed `mod_down_levels()` error for a removed prefix basis.
pub fn mod_down_levels_reconstruct_error_upper_bound(
    removed_moduli: &[u64],
    full_reduce_max_plaintexts: &[BigUint],
) -> BigUint {
    BigUint::from(removed_moduli.len()) * product_modulus(removed_moduli) +
        conv_error_quotient_upper_bound(removed_moduli, full_reduce_max_plaintexts)
}

/// Upper-bound the reconstructed `mod_down_one_level()` error for one removed suffix modulus.
///
/// The current implementation satisfies an exact error formula `value - q_removed * output =
/// value mod q_removed`, so the error is always at most `q_removed - 1`.
pub fn mod_down_one_level_reconstruct_error_upper_bound(removed_modulus: u64) -> BigUint {
    assert!(removed_modulus > 0, "removed_modulus must be non-zero");
    BigUint::from(removed_modulus - 1)
}

impl<P: Poly> NestedRnsPoly<P> {
    fn mod_switch_active_levels(&self) -> usize {
        self.max_plaintexts.len()
    }

    fn mod_switch_level_offset(&self) -> usize {
        self.level_offset
    }

    fn retain_and_scale(
        &self,
        global_level: usize,
        scalar: u64,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let levels = self.mod_switch_active_levels();
        assert!(
            global_level >= self.level_offset && global_level < self.level_offset + levels,
            "retained level lies outside the active window"
        );
        let local = global_level - self.level_offset;
        let mut scalars = vec![0u64; levels];
        scalars[local] = scalar;
        let plan = (0..self.num_coefficient_slots)
            .map(|c| {
                (u32::try_from(c).expect("coefficient block must fit u32"), Some(scalars.clone()))
            })
            .collect::<Vec<_>>();
        let mut isolated = self.slot_transfer(&plan, circuit);
        for a in 0..levels {
            if a != local {
                isolated.max_plaintexts[a] = BigUint::ZERO;
                isolated.p_max_traces[a] = BigUint::ZERO;
            }
        }
        isolated
    }

    fn prefix_levels(&self, levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        assert!(
            levels <= self.mod_switch_active_levels(),
            "requested prefix {levels} exceeds available levels"
        );
        let scalars =
            (0..self.mod_switch_active_levels()).map(|a| u64::from(a < levels)).collect::<Vec<_>>();
        let plan = (0..self.num_coefficient_slots)
            .map(|c| (c as u32, Some(scalars.clone())))
            .collect::<Vec<_>>();
        let masked = self.slot_transfer(&plan, circuit);
        Self::new(
            self.ctx.clone(),
            masked.inner,
            self.num_coefficient_slots,
            Some(self.level_offset),
            Some(levels),
            self.max_plaintexts[..levels].to_vec(),
        )
        .with_p_max_traces(self.p_max_traces[..levels].to_vec())
    }

    fn suffix_levels(&self, skip_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(skip_levels <= active_levels, "requested suffix skip exceeds available levels");
        let levels = active_levels - skip_levels;
        let scalars = (0..active_levels).map(|a| u64::from(a >= skip_levels)).collect::<Vec<_>>();
        let plan = (0..self.num_coefficient_slots)
            .map(|c| (c as u32, Some(scalars.clone())))
            .collect::<Vec<_>>();
        let masked = self.slot_transfer(&plan, circuit);
        Self::new(
            self.ctx.clone(),
            masked.inner,
            self.num_coefficient_slots,
            Some(self.level_offset + skip_levels),
            Some(levels),
            self.max_plaintexts[skip_levels..].to_vec(),
        )
        .with_p_max_traces(self.p_max_traces[skip_levels..].to_vec())
    }

    fn move_lane(
        &self,
        source_global: usize,
        target_global: usize,
        output_level_offset: usize,
        total_levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let nonzero = self
            .max_plaintexts
            .iter()
            .enumerate()
            .filter(|(_, bound)| *bound != &BigUint::ZERO)
            .map(|(a, _)| self.level_offset + a)
            .collect::<Vec<_>>();
        assert!(
            nonzero.iter().all(|&g| g == source_global),
            "move_lane requires a single nonzero source lane"
        );
        assert!(
            target_global >= output_level_offset &&
                target_global < output_level_offset + total_levels,
            "target lane lies outside output window"
        );
        let slots = self.num_coefficient_slots * self.ctx.q_moduli_depth;
        let offset = (target_global + slots - source_global) % slots;
        let inner = self
            .inner
            .gate_ids()
            .map(|gate| circuit.slot_rotation_gate(gate, offset, slots).as_single_wire())
            .collect::<Vec<_>>();
        let source_local = source_global - self.level_offset;
        let target_local = target_global - output_level_offset;
        let mut bounds = vec![BigUint::ZERO; total_levels];
        let mut traces = vec![BigUint::ZERO; total_levels];
        bounds[target_local] = self.max_plaintexts[source_local].clone();
        traces[target_local] = self.p_max_traces[source_local].clone();
        Self::new(
            self.ctx.clone(),
            crate::circuit::BatchedWire::from_batches(inner),
            self.num_coefficient_slots,
            Some(output_level_offset),
            Some(total_levels),
            bounds,
        )
        .with_p_max_traces(traces)
    }

    fn broadcast_lane(
        &self,
        source_global: usize,
        output_level_offset: usize,
        levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let isolated = self.retain_and_scale(source_global, 1, circuit);
        let terms = (output_level_offset..output_level_offset + levels)
            .map(|target| {
                isolated.move_lane(source_global, target, output_level_offset, levels, circuit)
            })
            .collect::<Vec<_>>();
        reduce_nested_rns_terms_pairwise(terms, circuit, |left, right, circuit| {
            left.add(right, circuit)
        })
    }

    fn merge_disjoint(prefix: Self, original: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(prefix.num_coefficient_slots, original.num_coefficient_slots);
        assert_eq!(
            prefix.level_offset + prefix.mod_switch_active_levels(),
            original.level_offset,
            "merged nested-RNS windows must be adjacent"
        );
        assert!(
            prefix.max_plaintexts.iter().all(|b| b == &BigUint::ZERO) ||
                original.max_plaintexts.iter().all(|b| b == &BigUint::ZERO) ||
                prefix.level_offset + prefix.mod_switch_active_levels() <= original.level_offset,
            "merged nested-RNS nonzero windows must be disjoint"
        );
        let inner = prefix
            .inner
            .gate_ids()
            .zip(original.inner.gate_ids())
            .map(|(left, right)| circuit.add_gate(left, right).as_single_wire())
            .collect::<Vec<_>>();
        let mut bounds = prefix.max_plaintexts;
        bounds.extend(original.max_plaintexts.iter().cloned());
        let mut traces = prefix.p_max_traces;
        traces.extend(original.p_max_traces.iter().cloned());
        Self::new(
            original.ctx.clone(),
            crate::circuit::BatchedWire::from_batches(inner),
            original.num_coefficient_slots,
            Some(prefix.level_offset),
            Some(bounds.len()),
            bounds,
        )
        .with_p_max_traces(traces)
    }

    fn conv_between_levels(
        &self,
        source_local_indices: &[usize],
        target_global_indices: &[usize],
        output_level_offset: usize,
        output_levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(!source_local_indices.is_empty(), "Conv requires at least one source level");
        assert!(!target_global_indices.is_empty(), "Conv requires at least one target level");
        let active_levels = self.mod_switch_active_levels();
        for &source_idx in source_local_indices {
            assert!(
                source_idx < active_levels,
                "source_idx {source_idx} out of range for {active_levels} active levels"
            );
        }
        for &target_idx in target_global_indices {
            assert!(
                output_level_offset <= target_idx &&
                    target_idx < output_level_offset + output_levels,
                "target_idx {target_idx} out of range for output window [{output_level_offset}, {})",
                output_level_offset + output_levels
            );
        }

        let q_moduli = self.ctx.q_moduli();
        let source_moduli = source_local_indices
            .iter()
            .map(|&idx| q_moduli[self.level_offset + idx])
            .collect::<Vec<_>>();
        let mut target_terms =
            Vec::with_capacity(source_local_indices.len() * target_global_indices.len());
        for (source_pos, &source_idx) in source_local_indices.iter().enumerate() {
            let source_modulus = q_moduli[self.level_offset + source_idx];
            let q_hat_mod_q_i = modular_product_except(&source_moduli, source_pos, source_modulus);
            let q_hat_inv_mod_q_i =
                mod_inverse(q_hat_mod_q_i, source_modulus).unwrap_or_else(|| {
                    panic!(
                        "q_hat inverse must exist for source_idx {} modulo {}",
                        source_idx, source_modulus
                    )
                });
            let source_term = self
                .retain_and_scale(self.level_offset + source_idx, q_hat_inv_mod_q_i, circuit)
                .full_reduce(circuit);

            for &target_idx in target_global_indices {
                let target_modulus = q_moduli[target_idx];
                let q_hat_mod_target =
                    modular_product_except(&source_moduli, source_pos, target_modulus);
                let target_term = source_term
                    .move_lane(
                        self.level_offset + source_idx,
                        target_idx,
                        output_level_offset,
                        output_levels,
                        circuit,
                    )
                    .uniform_const_mul(q_hat_mod_target, circuit);
                target_terms.push(target_term);
            }
        }

        reduce_nested_rns_terms_pairwise(target_terms, circuit, |left, right, circuit| {
            left.add(right, circuit)
        })
    }

    /// Evaluate the paper's Algorithm 1 `ModUp` when a contiguous block of `extra_levels` moduli is
    /// prepended immediately before the active window.
    pub fn mod_up_levels(&self, extra_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        let source_offset = self.mod_switch_level_offset();
        assert!(extra_levels > 0, "ModUp requires at least one prepended level");
        assert!(
            extra_levels <= source_offset,
            "ModUp requires {extra_levels} available prefix levels before source_offset {source_offset}"
        );
        let output_level_offset = source_offset - extra_levels;
        let target_indices = (output_level_offset..source_offset).collect::<Vec<_>>();
        let converted = self.conv_between_levels(
            &(0..source_levels).collect::<Vec<_>>(),
            &target_indices,
            output_level_offset,
            extra_levels,
            circuit,
        );
        Self::merge_disjoint(converted, self, circuit)
    }

    pub fn mod_up_one_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.mod_up_levels(1, circuit)
    }

    /// Evaluate the paper's Algorithm 2 `ModDown` when the removable basis is the initial prefix
    /// block of `remove_levels` active q-levels.
    pub fn mod_down_levels(&self, remove_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(remove_levels > 0, "ModDown requires at least one removable level");
        assert!(
            remove_levels < active_levels,
            "ModDown requires at least one kept level: active_levels={active_levels}, remove_levels={remove_levels}"
        );

        let level_offset = self.mod_switch_level_offset();
        let kept_levels = active_levels - remove_levels;
        let kept_offset = level_offset + remove_levels;
        let removed_indices = (0..remove_levels).collect::<Vec<_>>();
        let target_indices = (kept_offset..kept_offset + kept_levels).collect::<Vec<_>>();
        let q_moduli = self.ctx.q_moduli();
        let removed_moduli = &q_moduli[level_offset..kept_offset];
        let kept = self.suffix_levels(remove_levels, circuit);
        let converted_extra = self.conv_between_levels(
            &removed_indices,
            &target_indices,
            kept_offset,
            kept_levels,
            circuit,
        );
        let difference = kept.sub(&converted_extra, circuit);
        let inverse_constants = q_moduli[kept_offset..kept_offset + kept_levels]
            .iter()
            .map(|&q_i| {
                let removed_product_mod_q_i = modular_product(removed_moduli, q_i);
                mod_inverse(removed_product_mod_q_i, q_i).unwrap_or_else(|| {
                    panic!(
                        "removed basis product {:?} must be invertible modulo {}",
                        removed_moduli, q_i
                    )
                })
            })
            .collect::<Vec<_>>();
        difference.const_mul(&inverse_constants, circuit)
    }

    pub fn mod_down_one_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(active_levels > 1, "ModDown requires at least one kept level");
        let kept_levels = active_levels - 1;
        let level_offset = self.mod_switch_level_offset();
        let removed_local_idx = active_levels - 1;
        let removed_global_idx = level_offset + removed_local_idx;
        let q_moduli = self.ctx.q_moduli();
        let removed_modulus = q_moduli[removed_global_idx];
        let kept = self.prefix_levels(kept_levels, circuit);
        let converted_extra =
            self.broadcast_lane(removed_global_idx, level_offset, kept_levels, circuit);
        let difference = kept.sub(&converted_extra, circuit);
        let inverse_constants = q_moduli[level_offset..level_offset + kept_levels]
            .iter()
            .map(|&q_i| {
                mod_inverse(removed_modulus % q_i, q_i).unwrap_or_else(|| {
                    panic!(
                        "removed suffix modulus {} must be invertible modulo {}",
                        removed_modulus, q_i
                    )
                })
            })
            .collect::<Vec<_>>();
        difference.const_mul(&inverse_constants, circuit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        test_utils::{diagonal_matrix, execute_circuit_with_shape},
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_traits::{ToPrimitive, Zero};
    use std::sync::Arc;

    const P_MODULI_BITS: usize = 6;
    const MAX_UNREDUCED_MULS: usize = 2;
    const SCALE: u64 = 1 << 8;
    const BASE_BITS: u32 = 6;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        q_level: Option<usize>,
    ) -> (DCRTPolyParams, std::sync::Arc<crate::circuit_gadgets::arith::NestedRnsPolyContext>) {
        let params = DCRTPolyParams::new(2, 4, 12, BASE_BITS);
        let ctx = std::sync::Arc::new(crate::circuit_gadgets::arith::NestedRnsPolyContext::setup(
            circuit,
            &params,
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            SCALE,
            false,
            q_level,
        ));
        (params, ctx)
    }

    fn random_value_for_modulus(modulus: &BigUint) -> BigUint {
        let mut rng = rand::rng();
        crate::utils::gen_biguint_for_modulus(&mut rng, modulus)
    }

    fn max_value_for_modulus(modulus: &BigUint) -> BigUint {
        assert!(modulus != &BigUint::ZERO, "max input requires a non-zero modulus");
        modulus - BigUint::from(1u64)
    }

    fn residues_from_value(moduli: &[u64], value: &BigUint) -> Vec<u64> {
        moduli
            .iter()
            .map(|&q_i| (value % BigUint::from(q_i)).to_u64().expect("residue must fit in u64"))
            .collect()
    }

    fn encode_runtime_input(
        params: &DCRTPolyParams,
        ctx: &crate::circuit_gadgets::arith::NestedRnsPolyContext,
        value: &BigUint,
        level_offset: usize,
        enable_levels: Option<usize>,
    ) -> Vec<DCRTPolyMatrix> {
        crate::circuit_gadgets::arith::encode_nested_rns_poly_with_offset::<DCRTPoly>(
            ctx.p_moduli_bits,
            ctx.max_unreduced_muls,
            params,
            std::slice::from_ref(value),
            level_offset,
            enable_levels,
        )
        .into_iter()
        .map(|lanes| {
            diagonal_matrix(
                params,
                lanes.into_iter().map(|lane| DCRTPoly::from_biguint_to_constant(params, lane)),
            )
        })
        .collect()
    }

    fn execute_outputs(
        name: &str,
        params: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: &[DCRTPolyMatrix],
    ) -> Vec<BigUint> {
        let wire_size = inputs[0].row_size();
        execute_circuit_with_shape(name, params, circuit, inputs, (wire_size, wire_size))
            .into_iter()
            .map(|matrix| matrix.entry(0, 0).coeffs_biguints()[0].clone())
            .collect()
    }

    #[test]
    fn sparse_reduction_and_subtraction_preserve_literal_zero_lanes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(2, 3, 12, BASE_BITS);
        let ctx = Arc::new(crate::circuit_gadgets::arith::NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            6,
            2,
            SCALE,
            false,
            None,
        ));
        let level_offset = 0;
        let active_levels = 3;
        let source_global = 1;
        let target_global = 2;
        let source_local = source_global - level_offset;
        let mut left_bounds = vec![BigUint::ZERO; active_levels];
        left_bounds[source_local] = BigUint::from(ctx.q_moduli()[source_global] - 1);
        let mut left_traces = vec![BigUint::ZERO; active_levels];
        left_traces[source_local] = ctx.reduced_p_max_trace();
        let left = NestedRnsPoly::input_with_metadata(
            ctx.clone(),
            1,
            Some(active_levels),
            Some(level_offset),
            left_bounds,
            left_traces,
            &mut circuit,
        )
        .retain_and_scale(source_global, 1, &mut circuit);
        let right = NestedRnsPoly::input_with_metadata(
            ctx.clone(),
            1,
            Some(active_levels),
            Some(level_offset),
            vec![BigUint::ZERO; active_levels],
            vec![BigUint::ZERO; active_levels],
            &mut circuit,
        )
        .retain_and_scale(source_global, 1, &mut circuit);
        let difference = left.sub(&right, &mut circuit);
        let reduced = difference.full_reduce(&mut circuit);
        let moved = reduced.move_lane(
            source_global,
            target_global,
            level_offset,
            active_levels,
            &mut circuit,
        );
        assert_eq!(
            moved
                .max_plaintexts
                .iter()
                .enumerate()
                .filter(|(_, bound)| *bound != &BigUint::ZERO)
                .map(|(local, _)| level_offset + local)
                .collect::<Vec<_>>(),
            vec![target_global]
        );
        circuit.output([difference.inner, reduced.inner, moved.inner]);

        let value = BigUint::from(1u8);
        let mut inputs =
            encode_runtime_input(&params, &ctx, &value, level_offset, Some(active_levels));
        inputs.extend(encode_runtime_input(
            &params,
            &ctx,
            &BigUint::ZERO,
            level_offset,
            Some(active_levels),
        ));
        let wire_size = ctx.q_moduli_depth;
        let outputs = execute_circuit_with_shape(
            "nested-rns-sparse-sub-reduce-move",
            &params,
            &circuit,
            &inputs,
            (wire_size, wire_size),
        );
        assert_eq!(outputs.len(), 3 * ctx.p_moduli.len());
        let reduced_outputs = &outputs[ctx.p_moduli.len()..2 * ctx.p_moduli.len()];
        let moved_outputs = &outputs[2 * ctx.p_moduli.len()..];
        for (reduced_output, output) in reduced_outputs.iter().zip(moved_outputs) {
            let expected_target =
                reduced_output.entry(source_global, source_global).coeffs_biguints();
            for slot in 0..wire_size {
                let coefficients = output.entry(slot, slot).coeffs_biguints();
                if slot == target_global {
                    assert_eq!(coefficients, expected_target);
                } else {
                    assert!(
                        coefficients.iter().all(|coefficient| coefficient == &BigUint::ZERO),
                        "physical slot {slot} must remain literal zero"
                    );
                }
            }
        }
    }

    #[test]
    fn small_packed_mod_up_preserves_the_source_basis() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(2, 4, 12, BASE_BITS);
        let ctx = Arc::new(crate::circuit_gadgets::arith::NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            6,
            2,
            SCALE,
            false,
            None,
        ));
        let input = NestedRnsPoly::input(ctx.clone(), 1, Some(2), Some(2), &mut circuit);
        let raised = input.mod_up_levels(1, &mut circuit);
        assert_eq!(raised.level_offset, 1);
        assert_eq!(raised.enable_levels, Some(3));
        let output = raised.reconstruct(&mut circuit);
        circuit.output(vec![raised.inner, output.into()]);

        let value = BigUint::from(1u8);
        let inputs = encode_runtime_input(&params, &ctx, &value, 2, Some(2));
        let outputs = execute_circuit_with_shape(
            "small-packed-mod-up",
            &params,
            &circuit,
            &inputs,
            (ctx.q_moduli_depth, ctx.q_moduli_depth),
        );
        let (raw, reconstructed) = outputs.split_at(ctx.p_moduli.len());
        assert_eq!(reconstructed.len(), 1);
        for residue in raw {
            assert!(
                residue
                    .entry(0, 0)
                    .coeffs_biguints()
                    .iter()
                    .all(|coefficient| coefficient.is_zero()),
                "physical lane 0 must remain literal zero after ModUp"
            );
        }
        let reconstructed = reconstructed[0].entry(0, 0).coeffs_biguints()[0].clone();
        for &source_modulus in &ctx.q_moduli()[2..] {
            assert_eq!(reconstructed.clone() % BigUint::from(source_modulus), value);
        }
    }

    #[test]
    fn small_packed_mod_down_one_level_matches_exact_rescale() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::new(2, 3, 12, BASE_BITS);
        let ctx = Arc::new(crate::circuit_gadgets::arith::NestedRnsPolyContext::setup(
            &mut circuit,
            &params,
            6,
            2,
            SCALE,
            false,
            None,
        ));
        let input = NestedRnsPoly::input(ctx.clone(), 1, Some(3), None, &mut circuit);
        let lowered = input.mod_down_one_level(&mut circuit);
        let output = lowered.reconstruct(&mut circuit);
        circuit.output(vec![lowered.inner, output.into()]);

        let removed_modulus = ctx.q_moduli()[2];
        let value = BigUint::from(removed_modulus + 1);
        let inputs = encode_runtime_input(&params, &ctx, &value, 0, Some(3));
        let outputs = execute_circuit_with_shape(
            "small-packed-mod-down-one",
            &params,
            &circuit,
            &inputs,
            (ctx.q_moduli_depth, ctx.q_moduli_depth),
        );
        let (raw, reconstructed) = outputs.split_at(ctx.p_moduli.len());
        assert_eq!(reconstructed.len(), 1);
        for residue in raw {
            assert!(
                residue
                    .entry(2, 2)
                    .coeffs_biguints()
                    .iter()
                    .all(|coefficient| coefficient.is_zero()),
                "dropped physical lane must be literal zero after ModDown"
            );
        }
        let output = reconstructed[0].entry(0, 0).coeffs_biguints()[0].clone();
        let kept_modulus = product_modulus(&ctx.q_moduli()[..2]);
        assert_eq!(output % kept_modulus, BigUint::from(1u8));
    }

    fn test_mod_switch_nested_rns_mod_up_levels_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<crate::circuit_gadgets::arith::NestedRnsPolyContext>,
        value: BigUint,
    ) {
        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[1..];
        let source_modulus = product_modulus(source_moduli);
        let raised_modulus = product_modulus(&q_moduli);
        assert!(value < source_modulus, "input must be reduced modulo the active source basis");

        let input = NestedRnsPoly::input(ctx.clone(), 1, Some(3), Some(1), &mut circuit);
        let raised = input.mod_up_levels(1, &mut circuit);
        assert_eq!(raised.enable_levels, Some(4));
        assert_eq!(raised.level_offset, 0);
        let input_reconstructed = input.reconstruct(&mut circuit);
        let raised_reconstructed = raised.reconstruct(&mut circuit);
        circuit.output(vec![input_reconstructed, raised_reconstructed]);

        let encoded_input = encode_runtime_input(&params, &ctx, &value, 1, Some(3));
        let eval_results = execute_outputs("nested-rns-mod-up", &params, &circuit, &encoded_input);
        assert_eq!(eval_results.len(), 2);
        let input_output = eval_results[0].clone();
        let output = eval_results[1].clone();
        let output_reduced = output.clone() % &raised_modulus;
        assert_eq!(input_output.clone() % &source_modulus, value);

        for &q_i in source_moduli {
            let expected_residue = &value % BigUint::from(q_i);
            assert_eq!(input_output.clone() % BigUint::from(q_i), expected_residue);
            assert_eq!(output.clone() % BigUint::from(q_i), expected_residue);
        }

        assert!(output_reduced >= value, "ModUp output must not underflow the original value");
        let diff = &output_reduced - &value;
        println!("ModUp reconstruct diff: {}", diff);
        let bound = mod_up_reconstruct_error_upper_bound(
            source_moduli,
            &ctx.full_reduce_max_plaintexts[1..],
        );
        println!("ModUp reconstruct error bound by conv: {}", &bound);

        assert!(
            diff <= bound,
            "ModUp reconstruct error {:?} exceeds the derived upper bound {}",
            diff,
            bound
        );
    }

    fn test_mod_switch_nested_rns_mod_down_one_level_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<crate::circuit_gadgets::arith::NestedRnsPolyContext>,
        value: BigUint,
    ) {
        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[1..4];
        let input_modulus = product_modulus(all_moduli);
        let extra_modulus = q_moduli[3];
        let kept_moduli = &q_moduli[1..3];
        let kept_modulus = product_modulus(kept_moduli);
        assert!(value < input_modulus, "input must be reduced modulo the active input basis");

        let input = NestedRnsPoly::input(ctx.clone(), 1, Some(3), Some(1), &mut circuit);
        let lowered = input.mod_down_one_level(&mut circuit);
        assert_eq!(lowered.level_offset, 1);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let encoded_input = encode_runtime_input(&params, &ctx, &value, 1, Some(3));
        let eval_results =
            execute_outputs("nested-rns-mod-down-one", &params, &circuit, &encoded_input);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].clone();
        let output_reduced = output.clone() % &kept_modulus;

        let input_residues = residues_from_value(all_moduli, &value);
        let extra_residue = input_residues[2];
        for (idx, &q_i) in kept_moduli.iter().enumerate() {
            let residue = input_residues[idx];
            let diff = (residue + q_i - (extra_residue % q_i)) % q_i;
            let inv = mod_inverse(extra_modulus % q_i, q_i).expect("inverse must exist");
            let expected = BigUint::from((diff as u128 * inv as u128 % q_i as u128) as u64);
            assert_eq!(output.clone() % BigUint::from(q_i), expected);
        }

        let scaled_output = BigUint::from(extra_modulus) * &output_reduced;
        assert!(scaled_output <= value);
        let mod_down_error = &value - scaled_output;
        assert_eq!(mod_down_error, BigUint::from(extra_residue));
        let bound = mod_down_one_level_reconstruct_error_upper_bound(extra_modulus);
        assert!(
            mod_down_error <= bound,
            "ModDown-one-level error {:?} exceeds the derived upper bound {}",
            mod_down_error,
            bound
        );
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[1..];
        let source_modulus = product_modulus(source_moduli);
        let value = random_value_for_modulus(&source_modulus);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, value);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_max() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        let source_modulus = product_modulus(&ctx.q_moduli()[1..]);
        let value = max_value_for_modulus(&source_modulus);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, value);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input_modulus = product_modulus(&ctx.q_moduli()[1..4]);
        let value = random_value_for_modulus(&input_modulus);
        test_mod_switch_nested_rns_mod_down_one_level_generic(circuit, params, ctx, value);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        test_mod_switch_nested_rns_mod_down_one_level_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level_max() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input_modulus = product_modulus(&ctx.q_moduli()[1..4]);
        let value = max_value_for_modulus(&input_modulus);
        test_mod_switch_nested_rns_mod_down_one_level_generic(circuit, params, ctx, value);
    }

    fn test_mod_switch_nested_rns_mod_down_levels_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<crate::circuit_gadgets::arith::NestedRnsPolyContext>,
        value: BigUint,
    ) {
        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[..4];
        let removed_moduli = &q_moduli[..2];
        let kept_moduli = &q_moduli[2..4];
        let kept_modulus = product_modulus(kept_moduli);
        let removed_modulus = product_modulus(removed_moduli);
        let all_modulus = product_modulus(all_moduli);
        assert!(value < all_modulus, "input must be reduced modulo the active input basis");

        let input = NestedRnsPoly::input(ctx.clone(), 1, Some(4), None, &mut circuit);
        let lowered = input.mod_down_levels(2, &mut circuit);
        assert_eq!(lowered.level_offset, 2);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let encoded_input = encode_runtime_input(&params, &ctx, &value, 0, Some(4));
        let eval_results =
            execute_outputs("nested-rns-mod-down-levels", &params, &circuit, &encoded_input);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].clone();
        let output_reduced = output.clone() % &kept_modulus;

        let real_output = &value * &removed_modulus / &all_modulus;
        let bound = mod_down_levels_reconstruct_error_upper_bound(
            removed_moduli,
            &ctx.full_reduce_max_plaintexts[..2],
        );
        println!("ModDown reconstruct error bound by moddown: {}", &bound);
        let diff = if output_reduced >= real_output {
            &output_reduced - &real_output
        } else {
            &output_reduced + &kept_modulus - &real_output
        };
        println!("ModDown reconstruct diff: {}", &diff);
        assert!(
            diff <= bound,
            "ModDown reconstruct error {:?} exceeds the derived upper bound {}",
            diff,
            bound
        );
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let all_modulus = product_modulus(&ctx.q_moduli()[..4]);
        let value = random_value_for_modulus(&all_modulus);
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, value);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[serial_test::serial]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_max() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let all_modulus = product_modulus(&ctx.q_moduli()[..4]);
        let value = max_value_for_modulus(&all_modulus);
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, value);
    }
}
