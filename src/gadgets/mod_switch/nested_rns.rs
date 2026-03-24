//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues correspond to the contiguous active q-level window
//!   `q_{level_offset}, ..., q_{level_offset + active_levels - 1}` stored in `self.inner`.
//! - This module supports both CKKS special-prime insertion/removal at the prefix side of the
//!   active window and one-level suffix removal for rescaling.
//!
//! Output layout conventions:
//! - `conv_to_next_levels(k)` still converts into the contiguous suffix that follows the active
//!   window.
//! - `mod_up_levels(k)` prepends the contiguous `k`-modulus block immediately before the active
//!   window.
//! - `mod_down_levels(k)` removes the contiguous prefix block of size `k` from the active window.
//! - `mod_down_one_level()` removes the final active suffix modulus to support rescaling.
//!
//! All per-modulus arithmetic is expressed by composing existing `NestedRnsPoly` operations
//! rather than directly manipulating q-level residues as raw integers.

use crate::{circuit::PolyCircuit, gadgets::arith::NestedRnsPoly, poly::Poly, utils::mod_inverse};
use num_bigint::BigUint;

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

impl<P: Poly> NestedRnsPoly<P> {
    fn mod_switch_active_levels(&self) -> usize {
        match self.enable_levels {
            Some(levels) => {
                assert!(levels <= self.inner.len(), "enable_levels exceeds available levels");
                levels
            }
            None => self.inner.len(),
        }
    }

    fn mod_switch_level_offset(&self) -> usize {
        self.level_offset
    }

    fn zero_poly_with_offset(
        &self,
        level_offset: usize,
        levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let inner = (0..levels)
            .map(|_| {
                (0..self.ctx.p_moduli.len()).map(|_| circuit.const_zero_gate()).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let max_plaintexts = vec![BigUint::ZERO; levels];
        Self::new_with_offset(self.ctx.clone(), inner, level_offset, Some(levels), max_plaintexts)
    }

    fn zero_poly_with_levels(&self, levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        self.zero_poly_with_offset(self.level_offset, levels, circuit)
    }

    fn retain_only_level(&self, level_idx: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let levels = self.mod_switch_active_levels();
        assert!(level_idx < levels, "level_idx {level_idx} out of range for {levels} levels");

        let mut isolated = self.zero_poly_with_levels(levels, circuit);
        isolated.inner[level_idx] = self.inner[level_idx].clone();
        isolated.max_plaintexts[level_idx] = self.max_plaintexts[level_idx].clone();
        isolated
    }

    fn prefix_levels(&self, levels: usize) -> Self {
        assert!(levels <= self.inner.len(), "requested prefix {levels} exceeds available levels");
        Self::new_with_offset(
            self.ctx.clone(),
            self.inner[..levels].to_vec(),
            self.level_offset,
            Some(levels),
            self.max_plaintexts[..levels].to_vec(),
        )
    }

    fn suffix_levels(&self, skip_levels: usize) -> Self {
        assert!(skip_levels <= self.inner.len(), "requested suffix skip exceeds available levels");
        let levels = self.inner.len() - skip_levels;
        Self::new_with_offset(
            self.ctx.clone(),
            self.inner[skip_levels..].to_vec(),
            self.level_offset + skip_levels,
            Some(levels),
            self.max_plaintexts[skip_levels..].to_vec(),
        )
    }

    fn move_level_to_position(
        &self,
        source_idx: usize,
        output_level_offset: usize,
        total_levels: usize,
        target_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(source_idx < self.inner.len(), "source_idx {source_idx} out of range");
        assert!(target_idx < total_levels, "target_idx {target_idx} out of range");

        let mut moved = self.zero_poly_with_offset(output_level_offset, total_levels, circuit);
        moved.inner[target_idx] = self.inner[source_idx].clone();
        moved.max_plaintexts[target_idx] = self.max_plaintexts[source_idx].clone();
        moved
    }

    fn repeat_level_as_prefix(
        &self,
        source_idx: usize,
        output_level_offset: usize,
        levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(source_idx < self.inner.len(), "source_idx {source_idx} out of range");
        let replicated_bound = self.max_plaintexts[source_idx].clone();
        let mut repeated = self.zero_poly_with_offset(output_level_offset, levels, circuit);
        for level in 0..levels {
            repeated.inner[level] = self.inner[source_idx].clone();
            repeated.max_plaintexts[level] = replicated_bound.clone();
        }
        repeated
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
                output_level_offset <= target_idx && target_idx < output_level_offset + output_levels,
                "target_idx {target_idx} out of range for output window [{output_level_offset}, {})",
                output_level_offset + output_levels
            );
        }

        let q_moduli = self.ctx.q_moduli();
        let source_moduli = source_local_indices
            .iter()
            .map(|&idx| q_moduli[self.level_offset + idx])
            .collect::<Vec<_>>();
        let mut accumulator = self.zero_poly_with_offset(output_level_offset, output_levels, circuit);
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
            let mut source_scale = vec![0u64; active_levels];
            source_scale[source_idx] = q_hat_inv_mod_q_i;
            let source_term = self
                .retain_only_level(source_idx, circuit)
                .const_mul(&source_scale, circuit)
                .full_reduce(circuit);

            for &target_idx in target_global_indices {
                let target_modulus = q_moduli[target_idx];
                let q_hat_mod_target =
                    modular_product_except(&source_moduli, source_pos, target_modulus);
                let mut target_scale = vec![0u64; output_levels];
                target_scale[target_idx - output_level_offset] = q_hat_mod_target;
                let target_term = source_term
                    .move_level_to_position(
                        source_idx,
                        output_level_offset,
                        output_levels,
                        target_idx - output_level_offset,
                        circuit,
                    )
                    .const_mul(&target_scale, circuit);
                accumulator = accumulator.add(&target_term, circuit);
            }
        }

        accumulator
    }

    /// Evaluate the paper's `Conv` algorithm for a destination basis consisting of the next
    /// contiguous `extra_levels` q-moduli in this context.
    ///
    /// If the input uses the active prefix `q_0..q_{l-1}`, this returns a value with
    /// `enable_levels = Some(l + extra_levels)`. The first `l` levels are zero placeholders and
    /// the converted residues are written only into the appended levels.
    pub fn conv_to_next_levels(&self, extra_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        let source_offset = self.mod_switch_level_offset();
        assert!(extra_levels > 0, "Conv requires at least one target level");
        assert!(
            source_offset + source_levels + extra_levels <= self.ctx.q_moduli_depth,
            "Conv requires enough extra q-levels in the context: source_offset={source_offset}, source_levels={source_levels}, extra_levels={extra_levels}, q_moduli_depth={}",
            self.ctx.q_moduli_depth
        );

        let source_indices = (0..source_levels).collect::<Vec<_>>();
        let target_indices =
            (source_offset + source_levels..source_offset + source_levels + extra_levels)
                .collect::<Vec<_>>();
        self.conv_between_levels(
            &source_indices,
            &target_indices,
            source_offset,
            source_levels + extra_levels,
            circuit,
        )
    }

    pub fn conv_to_next_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.conv_to_next_levels(1, circuit)
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
        let mut inner = converted.inner;
        inner.extend(self.inner.iter().cloned());

        let mut max_plaintexts = converted.max_plaintexts;
        max_plaintexts.extend(self.max_plaintexts.iter().cloned());
        Self::new_with_offset(
            self.ctx.clone(),
            inner,
            output_level_offset,
            Some(source_levels + extra_levels),
            max_plaintexts,
        )
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
        let kept = self.suffix_levels(remove_levels);
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
        let kept = self.prefix_levels(kept_levels);
        let converted_extra = self.repeat_level_as_prefix(removed_local_idx, level_offset, kept_levels, circuit);
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
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::{One, ToPrimitive};
    use rand::Rng;

    const P_MODULI_BITS: usize = 6;
    const MAX_UNREDUCED_MULS: usize = crate::gadgets::arith::DEFAULT_MAX_UNREDUCED_MULS;
    const SCALE: u64 = 1 << 8;
    const BASE_BITS: u32 = 6;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        q_level: Option<usize>,
    ) -> (DCRTPolyParams, std::sync::Arc<crate::gadgets::arith::NestedRnsPolyContext>) {
        let params = DCRTPolyParams::new(4, 6, 18, BASE_BITS);
        let ctx = std::sync::Arc::new(crate::gadgets::arith::NestedRnsPolyContext::setup(
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

    fn product_modulus(moduli: &[u64]) -> BigUint {
        moduli.iter().fold(BigUint::one(), |acc, &q_i| acc * BigUint::from(q_i))
    }

    fn crt_value_from_residues(moduli: &[u64], residues: &[u64]) -> BigUint {
        assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
        let modulus = product_modulus(moduli);
        moduli.iter().zip(residues.iter()).fold(BigUint::ZERO, |acc, (&q_i, &residue)| {
            let q_i_big = BigUint::from(q_i);
            let q_hat = &modulus / &q_i_big;
            let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
            (acc + BigUint::from(residue) * q_hat * BigUint::from(q_hat_inv)) % &modulus
        })
    }

    fn random_residues_in_ranges(moduli: &[u64]) -> Vec<u64> {
        let mut rng = rand::rng();
        moduli.iter().map(|&q_i| rng.random_range(0..q_i)).collect()
    }

    fn small_deterministic_residues(moduli: &[u64]) -> Vec<u64> {
        moduli
            .iter()
            .enumerate()
            .map(|(idx, &q_i)| {
                let candidate = (idx as u64 % 4) + 1;
                if q_i <= 1 { 0 } else { candidate.min(q_i - 1) }
            })
            .collect()
    }

    fn div_ceil_biguint_by_u64(value: BigUint, divisor: u64) -> BigUint {
        let adjustment = BigUint::from(divisor.saturating_sub(1));
        (value + adjustment) / BigUint::from(divisor)
    }

    fn full_reduce_output_max_plaintext_bound(p_moduli: &[u64], q_modulus: u64) -> BigUint {
        let sum_p_moduli =
            p_moduli.iter().fold(BigUint::ZERO, |acc, &p_i| acc + BigUint::from(p_i));
        let modulus_count = u64::try_from(p_moduli.len())
            .expect("p_moduli length must fit in u64 for bound tracking");
        let numerator = (sum_p_moduli + BigUint::from(modulus_count)) * BigUint::from(q_modulus);
        div_ceil_biguint_by_u64(numerator, 4)
    }

    fn full_reduce_wrap_upper_bound(p_moduli: &[u64], q_modulus: u64, coeff: &BigUint) -> BigUint {
        let q_modulus_big = BigUint::from(q_modulus);
        assert!(coeff < &q_modulus_big, "full_reduce coefficient must be a canonical q-residue");
        let full_reduce_bound = full_reduce_output_max_plaintext_bound(p_moduli, q_modulus);
        (&full_reduce_bound - BigUint::one() - coeff) / q_modulus_big
    }

    fn fast_conversion_unsigned_lift(
        moduli: &[u64],
        residues: &[u64],
        target_moduli: &[u64],
        p_moduli: &[u64],
    ) -> (BigUint, BigUint, BigUint) {
        assert_eq!(moduli.len(), residues.len(), "CRT residues must match modulus count");
        let modulus = product_modulus(moduli);
        let value = crt_value_from_residues(moduli, residues);
        let mut full_reduce_wrap_slack = BigUint::ZERO;
        let lifted = moduli.iter().enumerate().fold(BigUint::ZERO, |acc, (idx, &q_i)| {
            let q_i_big = BigUint::from(q_i);
            let q_hat = &modulus / &q_i_big;
            let q_hat_mod_q_i = (&q_hat % &q_i_big).to_u64().expect("CRT residue must fit in u64");
            let q_hat_inv = mod_inverse(q_hat_mod_q_i, q_i).expect("CRT inverse must exist");
            let coeff = (BigUint::from(residues[idx]) * BigUint::from(q_hat_inv)) % &q_i_big;
            full_reduce_wrap_slack += full_reduce_wrap_upper_bound(p_moduli, q_i, &coeff);
            acc + coeff * q_hat
        });
        assert!(lifted >= value, "unsigned fast-conversion lift must not underflow");
        let delta = &lifted - &value;
        assert_eq!(delta.clone() % &modulus, BigUint::ZERO);
        let e_plus = delta / &modulus;
        assert!(e_plus < BigUint::from(moduli.len()));
        assert!(lifted < BigUint::from(moduli.len()) * &modulus);
        let accumulator_full_reduce_wrap_slack = target_moduli
            .iter()
            .map(|&q_i| full_reduce_wrap_upper_bound(p_moduli, q_i, &BigUint::ZERO))
            .max()
            .unwrap_or(BigUint::ZERO);
        let impl_error_upper =
            &e_plus + full_reduce_wrap_slack + accumulator_full_reduce_wrap_slack;
        (lifted, e_plus, impl_error_upper)
    }

    fn recover_unsigned_error_from_target_residue(
        value: &BigUint,
        source_modulus: &BigUint,
        target_modulus: u64,
        target_residue: &BigUint,
    ) -> BigUint {
        let target_modulus_big = BigUint::from(target_modulus);
        let source_mod_q_t = (source_modulus % &target_modulus_big)
            .to_u64()
            .expect("source modulus residue must fit in u64");
        let source_inv = mod_inverse(source_mod_q_t, target_modulus)
            .expect("source modulus must be invertible modulo target modulus");
        let value_mod_q_t =
            (value % &target_modulus_big).to_u64().expect("value residue must fit in u64");
        let target_residue_mod_q_t = (target_residue % &target_modulus_big)
            .to_u64()
            .expect("target residue must fit in u64");
        let diff = (target_residue_mod_q_t + target_modulus - value_mod_q_t) % target_modulus;
        BigUint::from((diff as u128 * source_inv as u128 % target_modulus as u128) as u64)
    }

    fn modular_difference(left: &BigUint, right: &BigUint, modulus: &BigUint) -> BigUint {
        if left >= right { left - right } else { left + modulus - right }
    }

    #[test]
    fn test_mod_switch_nested_rns_conv_to_next_levels() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let converted = input.conv_to_next_levels(2, &mut circuit);
        assert_eq!(converted.enable_levels, Some(4));
        assert_eq!(converted.inner.len(), 4);
        let reconstructed = converted.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[..2];
        let target_moduli = &q_moduli[2..4];
        let source_residues = small_deterministic_residues(source_moduli);
        let source_modulus = product_modulus(source_moduli);
        let value = crt_value_from_residues(source_moduli, &source_residues);
        let (_lifted, e_plus, impl_error_upper) =
            fast_conversion_unsigned_lift(source_moduli, &source_residues, target_moduli, &ctx.p_moduli);

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            Some(2),
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_results =
            circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].coeffs_biguints()[0].clone();
        for &q_i in source_moduli {
            assert_eq!(output.clone() % BigUint::from(q_i), BigUint::ZERO);
        }

        let recovered_errors = target_moduli
            .iter()
            .map(|&q_i| {
                recover_unsigned_error_from_target_residue(&value, &source_modulus, q_i, &output)
            })
            .collect::<Vec<_>>();
        assert!(
            recovered_errors.windows(2).all(|pair| pair[0] == pair[1]),
            "all appended target levels must encode the same fast-conversion error"
        );
        let recovered_error = &recovered_errors[0];
        assert!(recovered_error >= &e_plus);
        assert!(recovered_error <= &impl_error_upper);
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input = NestedRnsPoly::input_with_offset(ctx.clone(), 2, Some(2), &mut circuit);
        let raised = input.mod_up_levels(2, &mut circuit);
        assert_eq!(raised.enable_levels, Some(4));
        assert_eq!(raised.level_offset, 0);
        let input_reconstructed = input.reconstruct(&mut circuit);
        let raised_reconstructed = raised.reconstruct(&mut circuit);
        circuit.output(vec![input_reconstructed, raised_reconstructed]);

        let q_moduli = ctx.q_moduli();
        let target_moduli = &q_moduli[..2];
        let source_moduli = &q_moduli[2..4];
        let source_residues = random_residues_in_ranges(source_moduli);
        let source_modulus = product_modulus(source_moduli);
        let value = crt_value_from_residues(source_moduli, &source_residues);
        let (lifted, e_plus, impl_error_upper) =
            fast_conversion_unsigned_lift(source_moduli, &source_residues, target_moduli, &ctx.p_moduli);
        let raised_modulus = product_modulus(&q_moduli[..4]);
        let smallest_target_modulus =
            BigUint::from(*target_moduli.iter().min().expect("target basis must be non-empty"));

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly_with_offset(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            2,
            Some(2),
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_results =
            circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 2);
        let input_output = eval_results[0].coeffs_biguints()[0].clone();
        let output = eval_results[1].coeffs_biguints()[0].clone();
        let output_reduced = output.clone() % &raised_modulus;
        assert_eq!(input_output.clone() % &source_modulus, value);
        assert!(output_reduced >= value);
        assert!(lifted < raised_modulus, "lifted value must be less than raised modulus");
        assert_eq!(
            &lifted - &value,
            &source_modulus * &e_plus,
            "lifted value must equal value plus e_plus multiples of source modulus"
        );
        assert!(
            impl_error_upper < smallest_target_modulus,
            "implementation-level mod up error bound must fit below every appended modulus to avoid wraparound in per-target recovery"
        );
        for (idx, &q_i) in source_moduli.iter().enumerate() {
            assert_eq!(
                input_output.clone() % BigUint::from(q_i),
                BigUint::from(source_residues[idx])
            );
            assert_eq!(output.clone() % BigUint::from(q_i), BigUint::from(source_residues[idx]));
        }
        for &q_i in target_moduli {
            let recovered_error =
                recover_unsigned_error_from_target_residue(&value, &source_modulus, q_i, &output);
            assert!(recovered_error >= e_plus);
            assert!(recovered_error <= impl_error_upper);
        }

        let mod_up_error = (&output_reduced - &value) / &source_modulus;
        let recovered_errors = target_moduli
            .iter()
            .map(|&q_i| {
                recover_unsigned_error_from_target_residue(&value, &source_modulus, q_i, &output)
            })
            .collect::<Vec<_>>();
        assert!(
            recovered_errors.windows(2).all(|pair| pair[0] == pair[1]),
            "all appended target levels must encode the same ModUp error"
        );
        assert_eq!(mod_up_error, recovered_errors[0]);
        assert!(mod_up_error >= e_plus);
        assert!(mod_up_error <= impl_error_upper);
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input = NestedRnsPoly::input_with_offset(ctx.clone(), 1, Some(3), &mut circuit);
        let lowered = input.mod_down_one_level(&mut circuit);
        assert_eq!(lowered.level_offset, 1);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[1..4];
        let input_modulus = product_modulus(all_moduli);
        let extra_modulus = q_moduli[3];
        let kept_moduli = &q_moduli[1..3];
        let kept_modulus = product_modulus(kept_moduli);
        let input_residues = small_deterministic_residues(all_moduli);
        let value = crt_value_from_residues(all_moduli, &input_residues);
        assert!(value < input_modulus);

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly_with_offset(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            1,
            Some(3),
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_results =
            circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].coeffs_biguints()[0].clone();
        let output_reduced = output.clone() % &kept_modulus;

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
        assert!(mod_down_error < BigUint::from(extra_modulus));
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input = NestedRnsPoly::input(ctx.clone(), Some(4), &mut circuit);
        let lowered = input.mod_down_levels(2, &mut circuit);
        assert_eq!(lowered.level_offset, 2);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[..4];
        let removed_moduli = &q_moduli[..2];
        let kept_moduli = &q_moduli[2..4];
        let kept_modulus = product_modulus(kept_moduli);
        let removed_modulus = product_modulus(removed_moduli);
        let input_residues = random_residues_in_ranges(all_moduli);
        let removed_residues = &input_residues[..2];
        let value = crt_value_from_residues(all_moduli, &input_residues);
        let t = &value / &removed_modulus;
        let (_lifted, e_plus, _impl_error_upper) =
            fast_conversion_unsigned_lift(removed_moduli, removed_residues, kept_moduli, &ctx.p_moduli);

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            Some(4),
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_results =
            circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].coeffs_biguints()[0].clone();
        let output_reduced = output.clone() % &kept_modulus;
        let mod_down_error = modular_difference(&t, &output_reduced, &kept_modulus);

        for &q_i in kept_moduli {
            let q_big = BigUint::from(q_i);
            let expected = modular_difference(
                &(t.clone() % &q_big),
                &(mod_down_error.clone() % &q_big),
                &q_big,
            );
            assert_eq!(output.clone() % &q_big, expected);
        }
        assert!(mod_down_error >= e_plus);
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_one_prefix_level() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input = NestedRnsPoly::input(ctx.clone(), Some(3), &mut circuit);
        let lowered = input.mod_down_levels(1, &mut circuit);
        assert_eq!(lowered.level_offset, 1);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[..3];
        let removed_moduli = &q_moduli[..1];
        let kept_moduli = &q_moduli[1..3];
        let kept_modulus = product_modulus(kept_moduli);
        let removed_modulus = product_modulus(removed_moduli);
        let input_residues = random_residues_in_ranges(all_moduli);
        let removed_residues = &input_residues[..1];
        let value = crt_value_from_residues(all_moduli, &input_residues);
        let t = &value / &removed_modulus;
        let (_lifted, e_plus, _impl_error_upper) =
            fast_conversion_unsigned_lift(removed_moduli, removed_residues, kept_moduli, &ctx.p_moduli);

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            Some(3),
        );
        let plt_evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let eval_results =
            circuit.eval(&params, one, encoded_input, Some(&plt_evaluator), None, None);
        assert_eq!(eval_results.len(), 1);
        let output = eval_results[0].coeffs_biguints()[0].clone();
        let output_reduced = output.clone() % &kept_modulus;
        let mod_down_error = modular_difference(&t, &output_reduced, &kept_modulus);

        for &q_i in kept_moduli {
            let q_big = BigUint::from(q_i);
            let expected = modular_difference(
                &(t.clone() % &q_big),
                &(mod_down_error.clone() % &q_big),
                &q_big,
            );
            assert_eq!(output.clone() % &q_big, expected);
        }
        assert!(mod_down_error >= e_plus);
    }
}
