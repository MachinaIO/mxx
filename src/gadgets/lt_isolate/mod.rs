use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use std::collections::HashMap;

/// LT isolate gadget for extracting constant terms from polynomials.
pub struct LtIsolateGadget;

impl LtIsolateGadget {
    /// Extract the constant term (coefficient a_i).
    /// LT isolate sets all non constant coefficients to zero.
    ///
    /// This requires a public lookup table that maps each polynomial to its
    /// constant term version. The lookup_id parameter should be the ID
    /// returned by register_lt_isolate_lookup.
    pub fn apply_lt_isolate<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        input: GateId,
        lookup_id: usize,
    ) -> GateId {
        circuit.public_lookup_gate(input, lookup_id)
    }

    /// Register an LT_isolate lookup table for extracting constant terms.
    /// The table maps each possible polynomial value to a polynomial with only
    /// the constant term preserved (all other coefficients set to zero).
    pub fn register_lt_isolate_lookup<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        max_value: usize,
    ) -> usize {
        let mut lt_isolate_table = HashMap::new();

        // Create lookup entries for all possible polynomial values
        for i in 0..=max_value {
            let input_poly = P::from_usize_to_constant(params, i);
            let coeffs = input_poly.coeffs();
            let constant_coeff = if coeffs.is_empty() {
                P::Elem::zero(&params.modulus())
            } else {
                coeffs[0].clone()
            };

            // Create output polynomial with only constant term
            let mut output_coeffs =
                vec![P::Elem::zero(&params.modulus()); params.ring_dimension() as usize];
            output_coeffs[0] = constant_coeff;
            let output_poly = P::from_coeffs(params, &output_coeffs);

            lt_isolate_table.insert(input_poly, (i, output_poly));
        }

        let lt_isolate_lut = PublicLut::new(lt_isolate_table);
        circuit.register_public_lookup(lt_isolate_lut)
    }

    /// Register a general LT_isolate lookup table that works with arbitrary polynomials.
    /// This version creates a lookup table for all possible polynomials within the given
    /// norm bound T, preserving only their constant terms.
    ///
    /// WARNING: This can create very large lookup tables for large T and ring dimension n.
    pub fn register_general_lt_isolate_lookup<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        norm_bound: usize,
    ) -> usize {
        let mut lt_isolate_table = HashMap::new();
        let n = params.ring_dimension() as usize;

        // Helper to generate all polynomials with coefficients bounded by norm_bound
        fn generate_polys_recursive<P: Poly>(
            params: &P::Params,
            coeffs: &mut Vec<P::Elem>,
            idx: usize,
            n: usize,
            norm_bound: usize,
            table: &mut HashMap<P, (usize, P)>,
        ) {
            if idx == n {
                // We've filled all coefficients, create the polynomial
                let input_poly = P::from_coeffs(params, coeffs);

                // Create output with only constant term
                let mut output_coeffs = vec![P::Elem::zero(&params.modulus()); n];
                output_coeffs[0] = coeffs[0].clone();
                let output_poly = P::from_coeffs(params, &output_coeffs);

                // Use a hash of coefficients as the row index
                let row_idx = coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let digits = c.value().to_u64_digits();
                        let val = if digits.is_empty() { 0 } else { digits[0] };
                        val as usize * (i + 1)
                    })
                    .sum::<usize>() %
                    (norm_bound * n + 1);

                table.insert(input_poly, (row_idx, output_poly));
                return;
            }

            // Try all possible values for coefficient at index idx
            for val in 0..=norm_bound {
                coeffs[idx] = P::Elem::constant(&params.modulus(), val as u64);
                generate_polys_recursive::<P>(params, coeffs, idx + 1, n, norm_bound, table);
            }
        }

        let mut coeffs = vec![P::Elem::zero(&params.modulus()); n];
        generate_polys_recursive::<P>(params, &mut coeffs, 0, n, norm_bound, &mut lt_isolate_table);

        let lt_isolate_lut = PublicLut::new(lt_isolate_table);
        circuit.register_public_lookup(lt_isolate_lut)
    }

    /// Isolate and aggregate coefficients from a polynomial A(X).
    /// Returns a vector of gate IDs, where each gate contains one coefficient a_i.
    ///
    /// The circuit performs:
    /// - For i in 0..=k:
    ///   - rot(X) = rotate A(X) by X^{n-i} to send a_i coeff as constant term
    ///   - output[i] = LT_isolate(rot(X))
    ///
    /// This requires an LT_isolate lookup table to be registered first.
    pub fn isolate_coeffs<P: Poly>(
        circuit: &mut PolyCircuit<P>,
        input: GateId,
        degree_k: usize,
        lt_isolate_lookup_id: usize,
        ring_dim: usize,
    ) -> Vec<GateId> {
        let mut coefficients = Vec::with_capacity(degree_k + 1);

        for i in 0..=degree_k {
            let shift = if i == 0 { 0 } else { ring_dim - i };
            let rotated = if shift == 0 { input } else { circuit.rotate_gate(input, shift) };
            let isolated_coeff = Self::apply_lt_isolate(circuit, rotated, lt_isolate_lookup_id);
            coefficients.push(isolated_coeff);
        }

        coefficients
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::PolyCircuit,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    use super::*;

    #[test]
    fn test_apply_lt_isolate() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let lt_isolate_id = LtIsolateGadget::register_lt_isolate_lookup(&mut circuit, &params, 10);
        let inputs = circuit.input(1);
        let isolated = LtIsolateGadget::apply_lt_isolate(&mut circuit, inputs[0], lt_isolate_id);
        circuit.output(vec![isolated]);

        let test_poly = DCRTPoly::from_usize_to_constant(&params, 7);
        let plt_evaluator = PolyPltEvaluator::new();

        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[test_poly.clone()],
            Some(plt_evaluator),
        );

        let mut expected_coeffs = vec![
            <DCRTPoly as Poly>::Elem::zero(&params.modulus());
            params.ring_dimension() as usize
        ];
        expected_coeffs[0] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 7);
        let expected = DCRTPoly::from_coeffs(&params, &expected_coeffs);

        // verify
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_isolate_coeffs_circuit() {
        let params = DCRTPolyParams::default();
        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();

        // only assumes binary polynomials of degree n as input
        let norm_bound = 1;
        let lt_isolate_id =
            LtIsolateGadget::register_general_lt_isolate_lookup(&mut circuit, &params, norm_bound);

        let inputs = circuit.input(1);
        let a_poly_id = inputs[0];
        let n = params.ring_dimension() as usize;
        let isolated_coeffs =
            LtIsolateGadget::isolate_coeffs(&mut circuit, a_poly_id, n - 1, lt_isolate_id, n);
        circuit.output(isolated_coeffs);

        // Create a random binary polynomial
        let sampler = DCRTPolyUniformSampler::new();
        let a_poly = sampler.sample_poly(&params, &DistType::BitDist);

        assert!(a_poly.coeffs().len() == n);

        let plt_evaluator = PolyPltEvaluator::new();
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_poly.clone()],
            Some(plt_evaluator),
        );

        // verify
        // should return a vector of n constant polynomials
        assert_eq!(result.len(), n);

        for (i, output) in result.iter().enumerate() {
            let output_coeff = output.coeffs();
            assert_eq!(output_coeff.len(), n);
            for (j, coeff) in output_coeff.iter().enumerate() {
                if j == 0 {
                    assert_eq!(coeff, &a_poly.coeffs()[i]);
                } else {
                    assert_eq!(coeff, &<DCRTPoly as Poly>::Elem::zero(&params.modulus()));
                }
            }
        }
    }
}
