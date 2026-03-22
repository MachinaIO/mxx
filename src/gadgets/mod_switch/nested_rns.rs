//! Modulus-switching helpers over a single [`NestedRnsPoly`].
//!
//! Mapping from the paper notation to this repository:
//! - One `NestedRnsPoly` stores one integer in q-RNS form.
//! - The paper residues `a^(j) mod q_j` correspond to the active q-level prefix
//!   `self.inner[0..enable_levels_or_inner_len]`.
//! - This module only supports the special case requested by the user: `Conv` converts from the
//!   active prefix basis `C = {q_0, ..., q_{l-1}}` into exactly one extra modulus `q_l`, `ModUp`
//!   appends that one modulus, and `ModDown` removes exactly the final active modulus.
//!
//! Output layout conventions:
//! - `conv_to_next_level()` returns a value with `enable_levels = Some(l + 1)`. Levels `0..l` are
//!   zero-filled placeholders and level `l` stores the converted residue.
//! - `mod_up_one_level()` returns a value with `enable_levels = Some(l + 1)` where levels `0..l`
//!   preserve the original active prefix and level `l` stores the appended residue.
//! - `mod_down_one_level()` assumes the removable modulus is the final active level and returns
//!   only the kept prefix with `enable_levels = Some(l - 1)`.
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

    /// Evaluate the paper's `Conv` algorithm in the special case where the destination basis has
    /// exactly one modulus, chosen as the next q-level in this context.
    ///
    /// If the input uses the active prefix `q_0..q_{l-1}`, this returns a value with
    /// `enable_levels = Some(l + 1)`. The first `l` levels are zero placeholders and the new
    /// residue is written only into level `l`.
    pub fn conv_to_next_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        assert!(source_levels > 0, "Conv requires at least one active source level");
        assert!(
            source_levels < self.ctx.q_moduli_depth,
            "Conv requires one extra q-level in the context: source_levels={source_levels}, q_moduli_depth={}",
            self.ctx.q_moduli_depth
        );

        let q_moduli = self.ctx.q_moduli();
        let source_moduli = &q_moduli[..source_levels];
        let target_level = source_levels;
        let target_modulus = q_moduli[target_level];
        let total_levels = source_levels + 1;

        let mut accumulator = self.zero_poly_with_levels(total_levels, circuit);
        for source_idx in 0..source_levels {
            let q_hat_mod_q_i =
                modular_product_except(source_moduli, source_idx, q_moduli[source_idx]);
            let q_hat_inv_mod_q_i = mod_inverse(q_hat_mod_q_i, q_moduli[source_idx])
                .unwrap_or_else(|| {
                    panic!(
                        "q_hat inverse must exist for source_idx {} modulo {}",
                        source_idx, q_moduli[source_idx]
                    )
                });
            let mut source_scale = vec![0u64; source_levels];
            source_scale[source_idx] = q_hat_inv_mod_q_i;
            let source_term = self
                .retain_only_level(source_idx, circuit)
                .const_mul(&source_scale, circuit)
                .full_reduce(circuit);

            let q_hat_mod_target =
                modular_product_except(source_moduli, source_idx, target_modulus);
            let mut target_scale = vec![0u64; total_levels];
            target_scale[target_level] = q_hat_mod_target;
            let target_term = source_term
                .move_level_to_position(source_idx, total_levels, target_level, circuit)
                .const_mul(&target_scale, circuit);
            accumulator = accumulator.add(&target_term, circuit);
        }

        accumulator.full_reduce(circuit)
    }

    /// Evaluate the paper's Algorithm 1 `ModUp` in the special case where exactly one modulus is
    /// appended. The new modulus is the next q-level available in the context.
    ///
    /// If the input uses the active prefix `q_0..q_{l-1}`, the output uses
    /// `enable_levels = Some(l + 1)` and stores `(a^(0), ..., a^(l-1), a_tilde^(l))`.
    pub fn mod_up_one_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let source_levels = self.mod_switch_active_levels();
        let converted = self.conv_to_next_level(circuit);
        let mut inner = self.inner[..source_levels].to_vec();
        inner.push(converted.inner[source_levels].clone());

        let mut max_plaintexts = self.max_plaintexts[..source_levels].to_vec();
        max_plaintexts.push(converted.max_plaintexts[source_levels].clone());
        Self::new(self.ctx.clone(), inner, Some(source_levels + 1), max_plaintexts)
    }

    /// Evaluate the paper's Algorithm 2 `ModDown` in the special case where exactly one modulus is
    /// removed, assuming the extra modulus is the final active q-level.
    ///
    /// The method expects an active prefix `(b_tilde^(0), ..., b_tilde^(l))` and returns the kept
    /// prefix `(b^(0), ..., b^(l-1))` with `enable_levels = Some(l)`.
    pub fn mod_down_one_level(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let active_levels = self.mod_switch_active_levels();
        assert!(active_levels >= 2, "ModDown requires at least two active levels");

        let kept_levels = active_levels - 1;
        let extra_level = kept_levels;
        let q_moduli = self.ctx.q_moduli();
        let extra_modulus = q_moduli[extra_level];
        let prefix = self.prefix_levels(kept_levels);
        let converted_extra = self.repeat_level_as_prefix(extra_level, kept_levels, circuit);
        let difference = prefix.sub(&converted_extra, circuit);
        let inverse_constants = q_moduli[..kept_levels]
            .iter()
            .map(|&q_i| {
                mod_inverse(extra_modulus % q_i, q_i).unwrap_or_else(|| {
                    panic!("extra modulus {} must be invertible modulo {}", extra_modulus, q_i)
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

    #[test]
    fn test_mod_switch_nested_rns_conv_to_next_level() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(3));
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[..2];
        let converted = input.conv_to_next_level(&mut circuit);
        assert_eq!(converted.enable_levels, Some(3));
        assert_eq!(converted.inner.len(), 3);
        let reconstructed = converted.reconstruct(&mut circuit);
        circuit.output(vec![reconstructed]);

        let source_residues = [10u64, 5u64];
        assert!(source_residues[0] < source_moduli[0]);
        assert!(source_residues[1] < source_moduli[1]);
        let value = crt_value_from_residues(source_moduli, &source_residues);

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
        assert_ne!(output % BigUint::from(q_moduli[2]), BigUint::ZERO);
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_up() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit, Some(3));
        let input = NestedRnsPoly::input(ctx.clone(), Some(2), &mut circuit);
        let converted = input.conv_to_next_level(&mut circuit);
        let raised = input.mod_up_one_level(&mut circuit);
        assert_eq!(raised.enable_levels, Some(3));
        let input_reconstructed = input.reconstruct(&mut circuit);
        let converted_reconstructed = converted.reconstruct(&mut circuit);
        let raised_reconstructed = raised.reconstruct(&mut circuit);
        circuit.output(vec![input_reconstructed, converted_reconstructed, raised_reconstructed]);

        let q_moduli = ctx.q_moduli();
        let source_moduli = &q_moduli[..2];
        let source_residues = random_residues_in_ranges(source_moduli);
        let value = crt_value_from_residues(source_moduli, &source_residues);
        let target_modulus = q_moduli[2];

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
        for (idx, &q_i) in source_moduli.iter().enumerate() {
            assert_eq!(
                input_output.clone() % BigUint::from(q_i),
                BigUint::from(source_residues[idx])
            );
            assert_eq!(output.clone() % BigUint::from(q_i), BigUint::from(source_residues[idx]));
        }
        assert_eq!(
            output % BigUint::from(target_modulus),
            converted_output % BigUint::from(target_modulus)
        );
    }

    #[test]
    fn test_mod_switch_nested_rns_mod_down() {
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
        let input_residues = random_residues_in_ranges(all_moduli);
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

        let extra_residue = input_residues[2];
        for (idx, &q_i) in kept_moduli.iter().enumerate() {
            let residue = input_residues[idx];
            let diff = (residue + q_i - (extra_residue % q_i)) % q_i;
            let inv = mod_inverse(extra_modulus % q_i, q_i).expect("inverse must exist");
            let expected = BigUint::from((diff as u128 * inv as u128 % q_i as u128) as u64);
            assert_eq!(output.clone() % BigUint::from(q_i), expected);
        }
    }
}
