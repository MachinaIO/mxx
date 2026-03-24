//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues `a^(j) mod q_j` correspond to the active q-level prefix
//!   `self.inner[0..enable_levels_or_inner_len]`.
//! - This module supports the repository's contiguous-level special case: `Conv` converts from the
//!   active prefix basis `C = {q_0, ..., q_{l-1}}` into the next contiguous block `{q_l, ...,
//!   q_{l+k-1}}`, `ModUp` appends that block, and `ModDown` removes a contiguous suffix of the
//!   currently active levels.
//!
//! Output layout conventions:
//! - `conv_to_next_levels(k)` returns a value with `enable_levels = Some(l + k)`. Levels `0..l` are
//!   zero-filled placeholders and levels `l..l+k` store the converted residues.
//! - `mod_up_levels(k)` returns a value with `enable_levels = Some(l + k)` where levels `0..l`
//!   preserve the original active prefix and levels `l..l+k` store the appended residues.
//! - `mod_down_levels(k)` assumes the removable moduli are the final `k` active levels and returns
//!   only the kept prefix with `enable_levels = Some(l - k)`.
//! - The legacy one-level entry points are thin wrappers around the multi-level variants with `k =
//!   1`.
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

    fn zero_poly_with_levels(&self, levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let inner = (0..levels)
            .map(|_| {
                (0..self.ctx.p_moduli.len()).map(|_| circuit.const_zero_gate()).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let max_plaintexts = vec![BigUint::ZERO; levels];
        Self::new(self.ctx.clone(), inner, Some(levels), max_plaintexts)
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
        Self::new(
            self.ctx.clone(),
            self.inner[..levels].to_vec(),
            Some(levels),
            self.max_plaintexts[..levels].to_vec(),
        )
    }

    fn move_level_to_position(
        &self,
        source_idx: usize,
        total_levels: usize,
        target_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(source_idx < self.inner.len(), "source_idx {source_idx} out of range");
        assert!(target_idx < total_levels, "target_idx {target_idx} out of range");

        let mut moved = self.zero_poly_with_levels(total_levels, circuit);
        moved.inner[target_idx] = self.inner[source_idx].clone();
        moved.max_plaintexts[target_idx] = self.max_plaintexts[source_idx].clone();
        moved
    }

    fn repeat_level_as_prefix(
        &self,
        source_idx: usize,
        levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(source_idx < self.inner.len(), "source_idx {source_idx} out of range");
        let replicated_bound = self.max_plaintexts[source_idx].clone();
        let mut repeated = self.zero_poly_with_levels(levels, circuit);
        for level in 0..levels {
            repeated.inner[level] = self.inner[source_idx].clone();
            repeated.max_plaintexts[level] = replicated_bound.clone();
        }
        repeated
    }

    fn conv_between_levels(
        &self,
        source_indices: &[usize],
        target_indices: &[usize],
        total_levels: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(!source_indices.is_empty(), "Conv requires at least one source level");
        assert!(!target_indices.is_empty(), "Conv requires at least one target level");
        let active_levels = self.mod_switch_active_levels();
        for &source_idx in source_indices {
            assert!(
                source_idx < active_levels,
                "source_idx {source_idx} out of range for {active_levels} active levels"
            );
        }
        for &target_idx in target_indices {
            assert!(
                target_idx < total_levels,
                "target_idx {target_idx} out of range for {total_levels} levels"
            );
        }

        let q_moduli = self.ctx.q_moduli();
        let source_moduli = source_indices.iter().map(|&idx| q_moduli[idx]).collect::<Vec<_>>();
        let mut accumulator = self.zero_poly_with_levels(total_levels, circuit);
        for (source_pos, &source_idx) in source_indices.iter().enumerate() {
            let source_modulus = q_moduli[source_idx];
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

            for &target_idx in target_indices {
                let target_modulus = q_moduli[target_idx];
                let q_hat_mod_target =
                    modular_product_except(&source_moduli, source_pos, target_modulus);
                let mut target_scale = vec![0u64; total_levels];
                target_scale[target_idx] = q_hat_mod_target;
                let target_term = source_term
                    .move_level_to_position(source_idx, total_levels, target_idx, circuit)
                    .const_mul(&target_scale, circuit);
                accumulator = accumulator.add(&target_term, circuit);
            }
        }

        accumulator.full_reduce(circuit)
    }

    /// Evaluate the paper's `Conv` algorithm for a destination basis consisting of the next
    /// contiguous `extra_levels` q-moduli in this context.
    ///
    /// If the input uses the active prefix `q_0..q_{l-1}`, this returns a value with
    /// `enable_levels = Some(l + extra_levels)`. The first `l` levels are zero placeholders and
    /// the converted residues are written only into the appended levels.
    pub fn conv_to_next_levels(&self, extra_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        assert!(extra_levels > 0, "Conv requires at least one target level");
        assert!(
            source_levels + extra_levels <= self.ctx.q_moduli_depth,
            "Conv requires enough extra q-levels in the context: source_levels={source_levels}, extra_levels={extra_levels}, q_moduli_depth={}",
            self.ctx.q_moduli_depth
        );

        let total_levels = source_levels + extra_levels;
        let source_indices = (0..source_levels).collect::<Vec<_>>();
        let target_indices = (source_levels..total_levels).collect::<Vec<_>>();
        self.conv_between_levels(&source_indices, &target_indices, total_levels, circuit)
    }

    pub fn conv_to_next_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.conv_to_next_levels(1, circuit)
    }

    /// Evaluate the paper's Algorithm 1 `ModUp` when a contiguous block of `extra_levels` moduli is
    /// appended. The new moduli are the next q-levels available in the context.
    ///
    /// If the input uses the active prefix `q_0..q_{l-1}`, the output uses
    /// `enable_levels = Some(l + extra_levels)` and preserves the original prefix while appending
    /// the converted residues.
    pub fn mod_up_levels(&self, extra_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        let converted = self.conv_to_next_levels(extra_levels, circuit);
        let mut inner = self.inner[..source_levels].to_vec();
        inner.extend(converted.inner[source_levels..source_levels + extra_levels].iter().cloned());

        let mut max_plaintexts = self.max_plaintexts[..source_levels].to_vec();
        max_plaintexts.extend(
            converted.max_plaintexts[source_levels..source_levels + extra_levels].iter().cloned(),
        );
        Self::new(self.ctx.clone(), inner, Some(source_levels + extra_levels), max_plaintexts)
    }

    pub fn mod_up_one_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.mod_up_levels(1, circuit)
    }

    /// Evaluate the paper's Algorithm 2 `ModDown` when the removable basis is the final
    /// contiguous block of `remove_levels` active q-levels.
    ///
    /// The method expects an active prefix `(b_tilde^(0), ..., b_tilde^(l-1))` and returns the
    /// kept prefix after dropping the final `remove_levels` q-levels.
    pub fn mod_down_levels(&self, remove_levels: usize, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(remove_levels > 0, "ModDown requires at least one removable level");
        assert!(
            remove_levels < active_levels,
            "ModDown requires at least one kept level: active_levels={active_levels}, remove_levels={remove_levels}"
        );

        let kept_levels = active_levels - remove_levels;
        let removed_indices = (kept_levels..active_levels).collect::<Vec<_>>();
        let target_indices = (0..kept_levels).collect::<Vec<_>>();
        let q_moduli = self.ctx.q_moduli();
        let removed_moduli = &q_moduli[kept_levels..active_levels];
        let prefix = self.prefix_levels(kept_levels);
        let converted_extra = if remove_levels == 1 {
            let extra_level = kept_levels;
            // Algorithm 2 in `references/full_rns_ckks.pdf` calls Conv_{B->C} on the removed basis
            // B. In this repository's one-modulus special case, B contains exactly one
            // modulus P = q_l, so \hat P_0 = \prod_{i' != 0} p_{i'} is an empty product
            // and therefore equals 1. The fast-conversion sum then collapses to the
            // same integer a^(0) = [\tilde b]_P reduced into each kept q_i. At the
            // `NestedRnsPoly` level, that specialization is exactly "repeat the removed
            // level across the kept prefix" before subtracting and multiplying by
            // P^{-1} mod q_i.
            self.repeat_level_as_prefix(extra_level, kept_levels, circuit)
        } else {
            self.conv_between_levels(&removed_indices, &target_indices, kept_levels, circuit)
        };
        let difference = prefix.sub(&converted_extra, circuit);
        let inverse_constants = q_moduli[..kept_levels]
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
        self.mod_down_levels(1, circuit)
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
        let impl_error_upper = &e_plus + full_reduce_wrap_slack;
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
            fast_conversion_unsigned_lift(source_moduli, &source_residues, &ctx.p_moduli);

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
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let converted = input.conv_to_next_levels(2, &mut circuit);
        let raised = input.mod_up_levels(2, &mut circuit);
        assert_eq!(raised.enable_levels, Some(4));
        let input_reconstructed = input.reconstruct(&mut circuit);
        let converted_reconstructed = converted.reconstruct(&mut circuit);
        let raised_reconstructed = raised.reconstruct(&mut circuit);
        circuit.output(vec![input_reconstructed, converted_reconstructed, raised_reconstructed]);

        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[..2];
        let target_moduli = &q_moduli[2..4];
        let source_residues = random_residues_in_ranges(source_moduli);
        let source_modulus = product_modulus(source_moduli);
        let value = crt_value_from_residues(source_moduli, &source_residues);
        let (lifted, e_plus, impl_error_upper) =
            fast_conversion_unsigned_lift(source_moduli, &source_residues, &ctx.p_moduli);
        let raised_modulus = product_modulus(&q_moduli[..4]);
        let smallest_target_modulus =
            BigUint::from(*target_moduli.iter().min().expect("target basis must be non-empty"));

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
        assert_eq!(eval_results.len(), 3);
        let input_output = eval_results[0].coeffs_biguints()[0].clone();
        let converted_output = eval_results[1].coeffs_biguints()[0].clone();
        let output = eval_results[2].coeffs_biguints()[0].clone();
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
            assert_eq!(
                output.clone() % BigUint::from(q_i),
                converted_output.clone() % BigUint::from(q_i)
            );
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
        let (params, ctx) = create_test_context(&mut circuit, Some(3));
        let input = NestedRnsPoly::input(ctx.clone(), Some(3), &mut circuit);
        let lowered = input.mod_down_one_level(&mut circuit);
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[..3];
        let input_modulus = product_modulus(all_moduli);
        let extra_modulus = q_moduli[2];
        let kept_moduli = &q_moduli[..2];
        let kept_modulus = product_modulus(kept_moduli);
        let input_residues = small_deterministic_residues(all_moduli);
        let value = crt_value_from_residues(all_moduli, &input_residues);
        assert!(value < input_modulus);

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
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[..4];
        let kept_moduli = &q_moduli[..2];
        let removed_moduli = &q_moduli[2..4];
        let kept_modulus = product_modulus(kept_moduli);
        let removed_modulus = product_modulus(removed_moduli);
        let input_residues = random_residues_in_ranges(all_moduli);
        let removed_residues = &input_residues[2..4];
        let value = crt_value_from_residues(all_moduli, &input_residues);
        let t = &value / &removed_modulus;
        let (_lifted, e_plus, _impl_error_upper) =
            fast_conversion_unsigned_lift(removed_moduli, removed_residues, &ctx.p_moduli);

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
}
