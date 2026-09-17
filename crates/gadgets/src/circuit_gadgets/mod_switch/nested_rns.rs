//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues correspond to the contiguous active q-level window. Physical lane `c *
//!   active_depth + k` stores semantic tower `q_{window.offset + k}`; no inactive towers are
//!   allocated.
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
    circuit::PolyCircuit,
    circuit_gadgets::arith::{CrtWindow, NestedRnsPoly},
    poly::Poly,
    utils::mod_inverse,
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
        self.window.offset
    }

    fn retain_and_scale(
        &self,
        global_level: usize,
        scalar: u64,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let levels = self.mod_switch_active_levels();
        assert!(
            global_level >= self.window.offset && global_level < self.window.end(),
            "retained level lies outside the active window"
        );
        let local = global_level - self.window.offset;
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
        self.repack_window(
            CrtWindow::new(self.window.offset, levels, self.ctx.q_moduli_depth),
            circuit,
        )
    }

    fn suffix_levels(&self, skip_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(skip_levels <= active_levels, "requested suffix skip exceeds available levels");
        let levels = active_levels - skip_levels;
        self.repack_window(
            CrtWindow::new(self.window.offset + skip_levels, levels, self.ctx.q_moduli_depth),
            circuit,
        )
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
            .map(|(a, _)| self.window.offset + a)
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
        let source_local = source_global - self.window.offset;
        let target_local = target_global - output_level_offset;
        let plan = (0..self.num_coefficient_slots)
            .flat_map(|coefficient| {
                (0..total_levels).map(move |local| {
                    if local == target_local {
                        let source = coefficient * self.window.depth + source_local;
                        (u32::try_from(source).expect("physical slot must fit u32"), None)
                    } else {
                        (0, Some(0))
                    }
                })
            })
            .collect::<Vec<_>>();
        let inner = self
            .inner
            .gate_ids()
            .map(|gate| circuit.slot_transfer_gate(gate, &plan).as_single_wire())
            .collect::<Vec<_>>();
        let mut bounds = vec![BigUint::ZERO; total_levels];
        let mut traces = vec![BigUint::ZERO; total_levels];
        bounds[target_local] = self.max_plaintexts[source_local].clone();
        traces[target_local] = self.p_max_traces[source_local].clone();
        Self::new(
            self.ctx.clone(),
            crate::circuit::BatchedWire::from_batches(inner),
            self.num_coefficient_slots,
            CrtWindow::new(output_level_offset, total_levels, self.ctx.q_moduli_depth),
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
            prefix.window.end(),
            original.window.offset,
            "merged nested-RNS windows must be adjacent"
        );
        assert!(
            prefix.max_plaintexts.iter().all(|b| b == &BigUint::ZERO) ||
                original.max_plaintexts.iter().all(|b| b == &BigUint::ZERO) ||
                prefix.window.end() <= original.window.offset,
            "merged nested-RNS nonzero windows must be disjoint"
        );
        let window = CrtWindow::new(
            prefix.window.offset,
            prefix.window.depth + original.window.depth,
            original.ctx.q_moduli_depth,
        );
        let prefix = prefix.repack_window(window, circuit);
        let original = original.repack_window(window, circuit);
        let inner = prefix
            .inner
            .gate_ids()
            .zip(original.inner.gate_ids())
            .map(|(left, right)| circuit.add_gate(left, right).as_single_wire())
            .collect::<Vec<_>>();
        let bounds = prefix
            .max_plaintexts
            .iter()
            .zip(&original.max_plaintexts)
            .map(|(left, right)| left + right)
            .collect();
        let traces = prefix
            .p_max_traces
            .iter()
            .zip(&original.p_max_traces)
            .map(|(left, right)| left + right)
            .collect();
        Self::new(
            original.ctx.clone(),
            crate::circuit::BatchedWire::from_batches(inner),
            original.num_coefficient_slots,
            window,
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
            .map(|&idx| q_moduli[self.window.offset + idx])
            .collect::<Vec<_>>();
        let mut target_terms =
            Vec::with_capacity(source_local_indices.len() * target_global_indices.len());
        for (source_pos, &source_idx) in source_local_indices.iter().enumerate() {
            let source_modulus = q_moduli[self.window.offset + source_idx];
            let q_hat_mod_q_i = modular_product_except(&source_moduli, source_pos, source_modulus);
            let q_hat_inv_mod_q_i =
                mod_inverse(q_hat_mod_q_i, source_modulus).unwrap_or_else(|| {
                    panic!(
                        "q_hat inverse must exist for source_idx {} modulo {}",
                        source_idx, source_modulus
                    )
                });
            let source_term = self
                .retain_and_scale(self.window.offset + source_idx, q_hat_inv_mod_q_i, circuit)
                .full_reduce(circuit);

            for &target_idx in target_global_indices {
                let target_modulus = q_moduli[target_idx];
                let q_hat_mod_target =
                    modular_product_except(&source_moduli, source_pos, target_modulus);
                let target_term = source_term
                    .move_lane(
                        self.window.offset + source_idx,
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
