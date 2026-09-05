//! Untrusted numeric suggestions for the application-owned capped Lean checker.
//!
//! Every suggested matrix/scalar value has a kernel-checked equation. The number of
//! equations follows the binary lengths of the two counts, never their values.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct NumericCertificateInputs {
    pub cap: BigUint,
    pub n: BigUint,
    pub inner: BigUint,
    pub ell: BigUint,
    pub error_bound: BigUint,
    pub preimage_bound: BigUint,
    pub digit_bound: BigUint,
    pub injector_layers: BigUint,
    pub circuit_layers: BigUint,
}

#[derive(Clone, Debug)]
pub struct NumericCertificate {
    pub source: String,
    pub bound: BigUint,
    /// Includes the exponent-zero equation.
    pub matrix_steps: usize,
    /// Includes the exponent-zero equation.
    pub scalar_steps: usize,
}

fn cap(c: &BigUint, x: BigUint) -> BigUint {
    x.min(c.clone())
}

fn add(c: &BigUint, x: &BigUint, y: &BigUint) -> BigUint {
    cap(c, x + y)
}

fn mul(c: &BigUint, x: &BigUint, y: &BigUint) -> BigUint {
    cap(c, x * y)
}

type Matrix = [BigUint; 4];

fn matrix_mul(c: &BigUint, a: &Matrix, b: &Matrix) -> Matrix {
    std::array::from_fn(|index| {
        let row = index / 2;
        let column = index % 2;
        add(c, &mul(c, &a[row * 2], &b[column]), &mul(c, &a[row * 2 + 1], &b[2 + column]))
    })
}

fn matrix_literal(a: &Matrix) -> String {
    format!("{{ a00 := {}, a01 := {}, a10 := {}, a11 := {} }}", a[0], a[1], a[2], a[3])
}

fn exponents(count: &BigUint) -> Vec<BigUint> {
    let mut chain = Vec::new();
    let mut value = count.clone();
    while !value.is_zero() {
        chain.push(value.clone());
        value >>= 1usize;
    }
    chain.reverse();
    chain
}

const PRELUDE: &str = "import DiamondNumericProof

open MxxWe DiamondGeneratedProof
namespace DiamondNumericCertificate

private theorem matrix_step {C k : Nat} {A half : BoundMatrix2} (hk : k ≠ 0)
    (hh : matrixPowCap C A (k / 2) = half) : matrixPowCap C A k =
      if k % 2 = 0 then matrixMulCap C half half
      else matrixMulCap C (matrixMulCap C half half) (capMatrix C A) := by
  rw [matrixPowCap]
  simp only [hk, ↓reduceDIte, hh]

private theorem scalar_step {C a k half : Nat} (hk : k ≠ 0)
    (hh : cappedBinaryPow C a (k / 2) = half) : cappedBinaryPow C a k =
      if k % 2 = 0 then cmul C half half
      else cmul C (cmul C half half) (cap C a) := by
  rw [cappedBinaryPow]
  simp only [hk, ↓reduceDIte, hh]

";

/// Render an equality for the existing capped bound, not an acceptance certificate.
/// Callers must tie these numeric inputs to their generated graph and prove the strict gate.
pub fn render_numeric_certificate(inputs: &NumericCertificateInputs) -> NumericCertificate {
    let NumericCertificateInputs {
        cap: c,
        n,
        inner,
        ell,
        error_bound: e,
        preimage_bound: k,
        digit_bound: d,
        injector_layers: l,
        circuit_layers: h,
    } = inputs;
    let matrix = [n.clone(), BigUint::zero(), n * e * 2u8, inner * n * k];
    let matrix_text = matrix_literal(&matrix);
    let capped_matrix = matrix.clone().map(|entry| cap(c, entry));
    let one = cap(c, BigUint::one());
    let mut matrix_value = [one.clone(), BigUint::zero(), BigUint::zero(), one.clone()];
    let mut source = String::from(PRELUDE);
    writeln!(source, "private theorem matrix_0 : matrixPowCap {c} ({matrix_text}) 0 =\n    {} := by\n  rw [matrixPowCap]\n  norm_num [capMatrix, matrixOne, cap]\n", matrix_literal(&matrix_value)).unwrap();
    let matrix_chain = exponents(l);
    for (index, exponent) in matrix_chain.iter().enumerate() {
        let previous = matrix_literal(&matrix_value);
        matrix_value = matrix_mul(c, &matrix_value, &matrix_value);
        if exponent.bit(0) {
            matrix_value = matrix_mul(c, &matrix_value, &capped_matrix);
        }
        writeln!(source, "private theorem matrix_{} : matrixPowCap {c} ({matrix_text}) {exponent} =\n    {} := by\n  rw [matrix_step (k := {exponent}) (by norm_num)\n    (show matrixPowCap {c} ({matrix_text}) ({exponent} / 2) =\n      {previous} from matrix_{index})]\n  norm_num [matrixMulCap, capMatrix, cadd, cmul, cap]\n", index + 1, matrix_literal(&matrix_value)).unwrap();
    }
    let matrix_last = matrix_chain.len();
    let injector = add(c, &matrix_value[2], &mul(c, &matrix_value[3], &cap(c, e.clone())));
    writeln!(source, "private theorem injector_value : cappedInjectorN {c} {n} {inner} {e} {k} {l} = {injector} := by\n  unfold cappedInjectorN\n  have hm : injectorStepMatrix {n} {inner} {e} {k} = ({matrix_text}) := by\n    norm_num [injectorStepMatrix]\n  rw [hm, matrix_{matrix_last}]\n  norm_num [cadd, cmul, cap]\n").unwrap();
    let projected = mul(c, &cap(c, inner * n * k), &injector);
    writeln!(source, "private theorem projected_value : cappedProjectedInjectorBound {c} {n} {inner} {e} {k} {l} = {projected} := by\n  unfold cappedProjectedInjectorBound\n  rw [injector_value]\n  norm_num [cmul, cap]\n").unwrap();
    let a = ell * n * d;
    let factor = &a * 2u8 + 4u8;
    let mut scalar_value = one;
    writeln!(source, "private theorem scalar_0 : cappedBinaryPow {c} {factor} 0 = {scalar_value} := by\n  rw [cappedBinaryPow]\n  norm_num [cap]\n").unwrap();
    let scalar_chain = exponents(h);
    for (index, exponent) in scalar_chain.iter().enumerate() {
        let previous = scalar_value.clone();
        scalar_value = mul(c, &scalar_value, &scalar_value);
        if exponent.bit(0) {
            scalar_value = mul(c, &scalar_value, &cap(c, factor.clone()));
        }
        writeln!(source, "private theorem scalar_{} : cappedBinaryPow {c} {factor} {exponent} = {scalar_value} := by\n  rw [scalar_step (k := {exponent}) (by norm_num)\n    (show cappedBinaryPow {c} {factor} ({exponent} / 2) = {previous} from scalar_{index})]\n  norm_num [cmul, cap]\n", index + 1).unwrap();
    }
    let scalar_last = scalar_chain.len();
    let layer = mul(c, &scalar_value, &projected);
    let bound = add(
        c,
        &mul(c, &BigUint::from(2u8), &projected),
        &mul(c, &cap(c, a), &add(c, &projected, &layer)),
    );
    writeln!(source, "theorem numeric_bound : cappedDiamondBound {c} {n} {inner} {ell} {e} {k} {d} {l} {h} = {bound} := by\n  unfold cappedDiamondBound\n  rw [projected_value]\n  unfold cappedFinalBound cappedLayerBound\n  norm_num only [layerFactor]\n  rw [scalar_{scalar_last}]\n  norm_num [cadd, cmul, cap]\n\n#print axioms numeric_bound\n\nend DiamondNumericCertificate").unwrap();
    NumericCertificate {
        source,
        bound,
        matrix_steps: matrix_chain.len() + 1,
        scalar_steps: scalar_chain.len() + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> NumericCertificateInputs {
        NumericCertificateInputs {
            cap: 252u32.into(),
            n: 8u32.into(),
            inner: 10u32.into(),
            ell: 3u32.into(),
            error_bound: 6u32.into(),
            preimage_bound: 26u32.into(),
            digit_bound: 8u32.into(),
            injector_layers: 1u32.into(),
            circuit_layers: 2u32.into(),
        }
    }

    #[test]
    fn capped_arithmetic_preserves_zero_after_saturation() {
        for c in [0u32, 1, 10, 252] {
            let c = BigUint::from(c);
            let saturated = cap(&c, BigUint::from(10000u32));
            assert_eq!(mul(&c, &saturated, &BigUint::zero()), BigUint::zero());
            assert_eq!(mul(&c, &BigUint::zero(), &saturated), BigUint::zero());
        }
        let mut inputs = fixture();
        inputs.cap = BigUint::zero();
        assert_eq!(render_numeric_certificate(&inputs).bound, BigUint::zero());
        inputs.cap = 252u32.into();
        inputs.preimage_bound = BigUint::zero();
        assert_eq!(render_numeric_certificate(&inputs).bound, BigUint::zero());
    }

    #[test]
    fn fixture_threshold_equality_is_not_strict_acceptance() {
        let inputs = fixture();
        let certificate = render_numeric_certificate(&inputs);
        assert_eq!(certificate.bound, inputs.cap);
        assert_eq!((certificate.matrix_steps, certificate.scalar_steps), (2, 3));
        assert!(!certificate.source.contains("native_decide"));
        assert!(!certificate.source.contains("^"));
    }

    #[test]
    fn bit_chain_handles_zero_one_and_huge_counts() {
        let mut inputs = fixture();
        for (count, steps) in [(0u32, 1usize), (1, 2), (2, 3), (1 << 20, 22)] {
            inputs.injector_layers = count.into();
            inputs.circuit_layers = count.into();
            let certificate = render_numeric_certificate(&inputs);
            assert_eq!(certificate.matrix_steps, steps);
            assert_eq!(certificate.scalar_steps, steps);
            assert!(certificate.bound <= inputs.cap);
        }
        inputs.injector_layers = BigUint::one() << 256usize;
        inputs.circuit_layers = inputs.injector_layers.clone();
        let certificate = render_numeric_certificate(&inputs);
        assert_eq!((certificate.matrix_steps, certificate.scalar_steps), (258, 258));
        assert_eq!(certificate.source.matches("private theorem matrix_").count(), 259);
        assert!(certificate.source.len() < 500_000);
    }

    #[test]
    fn small_recurrences_match_direct_iteration() {
        for layers in 0u32..5 {
            for depth in 0u32..5 {
                let mut inputs = fixture();
                inputs.cap = BigUint::one() << 256usize;
                inputs.injector_layers = layers.into();
                inputs.circuit_layers = depth.into();
                let mut prefix = BigUint::one();
                let mut noise = inputs.error_bound.clone();
                for _ in 0..layers {
                    noise = &inputs.n * &inputs.error_bound * &prefix * 2u8 +
                        &inputs.inner * &inputs.n * &inputs.preimage_bound * noise;
                    prefix *= &inputs.n;
                }
                let b0 = &inputs.inner * &inputs.n * &inputs.preimage_bound * noise;
                let a = &inputs.ell * &inputs.n * &inputs.digit_bound;
                let factor = &a * 2u8 + 4u8;
                let mut bh = b0.clone();
                for _ in 0..depth {
                    bh *= &factor;
                }
                let expected = &b0 * 2u8 + a * (&b0 + bh);
                assert_eq!(render_numeric_certificate(&inputs).bound, cap(&inputs.cap, expected));
            }
        }
    }

    #[test]
    #[ignore = "requires Lean and a compiled DiamondNumericProof on LEAN_PATH"]
    fn generated_equations_reject_an_incorrect_suggested_numeral() {
        let directory = tempfile::tempdir().unwrap();
        let generated = render_numeric_certificate(&fixture());
        let mutated = generated.source.replacen(
            "{ a00 := 8, a01 := 0, a10 := 96, a11 := 252 } := by",
            "{ a00 := 9, a01 := 0, a10 := 96, a11 := 252 } := by",
            1,
        );
        assert_ne!(generated.source, mutated);
        for (name, source, succeeds) in
            [("Correct", generated.source, true), ("Incorrect", mutated, false)]
        {
            let path = directory.path().join(format!("{name}.lean"));
            std::fs::write(&path, source).unwrap();
            let output = std::process::Command::new("lean").arg(path).output().unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert_eq!(output.status.success(), succeeds, "{stdout}\n{stderr}");
            if succeeds {
                assert!(!stdout.contains("sorryAx"), "{stdout}");
                assert!(stdout.contains("[propext, Classical.choice, Quot.sound]"), "{stdout}");
            } else {
                assert!(stdout.contains("unsolved goals"), "{stdout}");
            }
        }
    }
}
