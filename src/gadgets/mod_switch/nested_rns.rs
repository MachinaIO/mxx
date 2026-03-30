//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues correspond to the contiguous active q-level window `q_{level_offset}, ...,
//!   q_{level_offset + active_levels - 1}` stored in `self.inner`.
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

use crate::{circuit::PolyCircuit, gadgets::arith::NestedRnsPoly, poly::Poly, utils::mod_inverse};
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
        Self::new(self.ctx.clone(), inner, Some(level_offset), Some(levels), max_plaintexts)
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
        Self::new(
            self.ctx.clone(),
            self.inner[..levels].to_vec(),
            Some(self.level_offset),
            Some(levels),
            self.max_plaintexts[..levels].to_vec(),
        )
    }

    fn suffix_levels(&self, skip_levels: usize) -> Self {
        assert!(skip_levels <= self.inner.len(), "requested suffix skip exceeds available levels");
        let levels = self.inner.len() - skip_levels;
        Self::new(
            self.ctx.clone(),
            self.inner[skip_levels..].to_vec(),
            Some(self.level_offset + skip_levels),
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
        let mut inner = converted.inner;
        inner.extend(self.inner.iter().cloned());

        let mut max_plaintexts = converted.max_plaintexts;
        max_plaintexts.extend(self.max_plaintexts.iter().cloned());
        Self::new(
            self.ctx.clone(),
            inner,
            Some(output_level_offset),
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
        let converted_extra =
            self.repeat_level_as_prefix(removed_local_idx, level_offset, kept_levels, circuit);
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
        __PAIR, __TestState,
        circuit::PolyCircuit,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::ToPrimitive;
    use std::sync::Arc;

    const P_MODULI_BITS: usize = 7;
    const MAX_UNREDUCED_MULS: usize = 4;
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

    fn test_mod_switch_nested_rns_mod_up_levels_generic(
        mut circuit: PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        ctx: Arc<crate::gadgets::arith::NestedRnsPolyContext>,
        value: BigUint,
    ) {
        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[2..];
        let source_modulus = product_modulus(source_moduli);
        let raised_modulus = product_modulus(&q_moduli);
        assert!(value < source_modulus, "input must be reduced modulo the active source basis");

        let input = NestedRnsPoly::input(ctx.clone(), Some(4), Some(2), &mut circuit);
        let raised = input.mod_up_levels(2, &mut circuit);
        assert_eq!(raised.enable_levels, Some(6));
        assert_eq!(raised.level_offset, 0);
        let input_reconstructed = input.reconstruct(&mut circuit);
        let raised_reconstructed = raised.reconstruct(&mut circuit);
        circuit.output(vec![input_reconstructed, raised_reconstructed]);

        let encoded_input = crate::gadgets::arith::encode_nested_rns_poly_with_offset(
            P_MODULI_BITS,
            MAX_UNREDUCED_MULS,
            &params,
            &value,
            2,
            Some(4),
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
            &ctx.full_reduce_max_plaintexts[2..],
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
        ctx: Arc<crate::gadgets::arith::NestedRnsPolyContext>,
        value: BigUint,
    ) {
        let q_moduli = ctx.q_moduli();
        let all_moduli = &q_moduli[1..4];
        let input_modulus = product_modulus(all_moduli);
        let extra_modulus = q_moduli[3];
        let kept_moduli = &q_moduli[1..3];
        let kept_modulus = product_modulus(kept_moduli);
        assert!(value < input_modulus, "input must be reduced modulo the active input basis");

        let input = NestedRnsPoly::input(ctx.clone(), Some(3), Some(1), &mut circuit);
        let lowered = input.mod_down_one_level(&mut circuit);
        assert_eq!(lowered.level_offset, 1);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

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

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[2..];
        let source_modulus = product_modulus(source_moduli);
        let value = random_value_for_modulus(&source_modulus);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_up_levels_max() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, None);
        let source_modulus = product_modulus(&ctx.q_moduli()[2..]);
        let value = max_value_for_modulus(&source_modulus);
        test_mod_switch_nested_rns_mod_up_levels_generic(circuit, params, ctx, value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let input_modulus = product_modulus(&ctx.q_moduli()[1..4]);
        let value = random_value_for_modulus(&input_modulus);
        test_mod_switch_nested_rns_mod_down_one_level_generic(circuit, params, ctx, value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_one_level_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        test_mod_switch_nested_rns_mod_down_one_level_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[sequential_test::sequential]
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
        ctx: Arc<crate::gadgets::arith::NestedRnsPolyContext>,
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

        let input = NestedRnsPoly::input(ctx.clone(), Some(4), None, &mut circuit);
        let lowered = input.mod_down_levels(2, &mut circuit);
        assert_eq!(lowered.level_offset, 2);
        assert_eq!(lowered.enable_levels, Some(2));
        let reconstructed = lowered.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

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

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_random() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let all_modulus = product_modulus(&ctx.q_moduli()[..4]);
        let value = random_value_for_modulus(&all_modulus);
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, BigUint::ZERO);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_mod_switch_nested_rns_mod_down_levels_max() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(4));
        let all_modulus = product_modulus(&ctx.q_moduli()[..4]);
        let value = max_value_for_modulus(&all_modulus);
        test_mod_switch_nested_rns_mod_down_levels_generic(circuit, params, ctx, value);
    }
}
