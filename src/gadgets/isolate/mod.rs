use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use std::{collections::HashMap, marker::PhantomData};

/// Isolation gadget for extracting constant terms from a low-degree polynomial.
pub struct IsolationGadget<P: Poly> {
    pub max_degree: u16, // the maximum degree of the input low-degree polynomial
    pub max_norm: u32,   /* each coefficient of the input polynomial should be less than or
                          * equal to `max_norm` */
    pub plt_ids: Vec<usize>, // the table ids registered in the circuit,
    _p: PhantomData<P>,
}

impl<P: Poly> IsolationGadget<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        max_degree: u16,
        max_norm: u32,
    ) -> Self {
        // Create lookup entries for all possible polynomial values for each degree
        let mut plt_ids = vec![];
        for i in 0..=max_degree {
            let mut fs = vec![];
            for j in 0..=max_degree {
                let mut f: HashMap<P::Elem, (usize, P::Elem)> = HashMap::new();
                for norm in 0..=max_norm {
                    let input = <P::Elem as PolyElem>::constant(&params.modulus(), norm as u64);
                    let output = if i == j {
                        input.clone()
                    } else {
                        <P::Elem as PolyElem>::zero(&params.modulus())
                    };
                    f.insert(input, (norm as usize, output));
                }
                fs.push(f);
            }
            let plt = PublicLut::<P>::new(fs);
            let plt_id = circuit.register_public_lookup(plt);
            plt_ids.push(plt_id);
        }
        Self { max_degree, max_norm, plt_ids, _p: PhantomData }
    }

    /// Extract a single term the input polynomial, where the coefficients of the other terms are
    /// set to zeros.
    pub fn isolate_single_term(
        &self,
        circuit: &mut PolyCircuit<P>,
        term_idx: usize,
        input: GateId,
    ) -> GateId {
        circuit.public_lookup_gate(input, self.plt_ids[term_idx])
    }

    // /// Register an LT_isolate lookup table for extracting constant terms.
    // /// The table maps each possible polynomial value to a polynomial with only
    // /// the constant term preserved (all other coefficients set to zero).
    // pub fn register_table<P: Poly>(
    //     &self,
    //     circuit: &mut PolyCircuit<P>,
    //     params: &P::Params,
    //     max_value: usize,
    // ) -> usize {
    //     let mut lt_isolate_table = HashMap::new();

    //     // Create lookup entries for all possible polynomial values
    //     for i in 0..=max_value {
    //         let input_poly = P::from_usize_to_constant(params, i);
    //         let coeffs = input_poly.coeffs();
    //         let constant_coeff = if coeffs.is_empty() {
    //             P::Elem::zero(&params.modulus())
    //         } else {
    //             coeffs[0].clone()
    //         };

    //         // Create output polynomial with only constant term
    //         let mut output_coeffs =
    //             vec![P::Elem::zero(&params.modulus()); params.ring_dimension() as usize];
    //         output_coeffs[0] = constant_coeff;
    //         let output_poly = P::from_coeffs(params, &output_coeffs);

    //         lt_isolate_table.insert(input_poly, (i, output_poly));
    //     }

    //     let lt_isolate_lut = PublicLut::new(lt_isolate_table);
    //     circuit.register_public_lookup(lt_isolate_lut)
    // }

    // /// Register an isolate lookup table that outputs a constant term with low-degree
    // polynomials. /// This version creates a lookup table for all possible polynomials within
    // the given /// norm bound T, preserving only their constant terms.
    // ///
    // /// WARNING: This can create very large lookup tables for large T and ring dimension n.
    // pub fn register_general_lt_isolate_lookup<P: Poly>(
    //     circuit: &mut PolyCircuit<P>,
    //     params: &P::Params,
    //     norm_bound: usize,
    // ) -> usize {
    //     let mut lt_isolate_table = HashMap::new();
    //     let n = params.ring_dimension() as usize;

    //     // Helper to generate all polynomials with coefficients bounded by norm_bound
    //     fn generate_polys_recursive<P: Poly>(
    //         params: &P::Params,
    //         coeffs: &mut Vec<P::Elem>,
    //         idx: usize,
    //         n: usize,
    //         norm_bound: usize,
    //         table: &mut HashMap<P, (usize, P)>,
    //     ) {
    //         if idx == n {
    //             // We've filled all coefficients, create the polynomial
    //             let input_poly = P::from_coeffs(params, coeffs);

    //             // Create output with only constant term
    //             let mut output_coeffs = vec![P::Elem::zero(&params.modulus()); n];
    //             output_coeffs[0] = coeffs[0].clone();
    //             let output_poly = P::from_coeffs(params, &output_coeffs);

    //             // Use a hash of coefficients as the row index
    //             let row_idx = coeffs
    //                 .iter()
    //                 .enumerate()
    //                 .map(|(i, c)| {
    //                     let digits = c.value().to_u64_digits();
    //                     let val = if digits.is_empty() { 0 } else { digits[0] };
    //                     val as usize * (i + 1)
    //                 })
    //                 .sum::<usize>() %
    //                 (norm_bound * n + 1);

    //             table.insert(input_poly, (row_idx, output_poly));
    //             return;
    //         }

    //         // Try all possible values for coefficient at index idx
    //         for val in 0..=norm_bound {
    //             coeffs[idx] = P::Elem::constant(&params.modulus(), val as u64);
    //             generate_polys_recursive::<P>(params, coeffs, idx + 1, n, norm_bound, table);
    //         }
    //     }

    //     let mut coeffs = vec![P::Elem::zero(&params.modulus()); n];
    //     generate_polys_recursive::<P>(params, &mut coeffs, 0, n, norm_bound, &mut
    // lt_isolate_table);

    //     let lt_isolate_lut = PublicLut::new(lt_isolate_table);
    //     circuit.register_public_lookup(lt_isolate_lut)
    // }

    /// Isolate coefficients from a polynomial A(X).
    /// Returns a vector of gate IDs, where each gate contains one coefficient a_i.
    ///
    /// The circuit performs:
    /// - For i in 0..=k:
    ///   - rot(X) = rotate A(X) by X^{n-i} to send a_i coeff as constant term
    ///   - output[i] = LT_isolate(rot(X))
    ///
    /// This requires an LT_isolate lookup table to be registered first.
    pub fn isolate_terms(&self, circuit: &mut PolyCircuit<P>, input: GateId) -> Vec<GateId> {
        let max_degree = self.max_degree as usize;
        let mut terms = Vec::with_capacity(max_degree + 1);

        for i in 0..=max_degree {
            let extracted = self.isolate_single_term(circuit, i, input);
            let rotated = circuit.rotate_gate(extracted, -(i as i32));
            terms.push(rotated);
        }

        terms
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::PolyCircuit,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    use super::*;

    #[test]
    fn test_isolate_single() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let isolate_gadget = IsolationGadget::setup(&mut circuit, &params, 2, 10);
        let inputs = circuit.input(1);
        let isolated0 = isolate_gadget.isolate_single_term(&mut circuit, 0, inputs[0]);
        let isolated1 = isolate_gadget.isolate_single_term(&mut circuit, 1, inputs[0]);
        let isolated2 = isolate_gadget.isolate_single_term(&mut circuit, 2, inputs[0]);
        circuit.output(vec![isolated0, isolated1, isolated2]);

        let mut input_coeffs = vec![
            <DCRTPoly as Poly>::Elem::zero(&params.modulus());
            params.ring_dimension() as usize
        ];
        input_coeffs[0] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 7);
        input_coeffs[1] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 3);
        input_coeffs[2] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 4);
        let input_poly = DCRTPoly::from_coeffs(&params, &input_coeffs);
        let plt_evaluator = PolyPltEvaluator::new();
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[input_poly.clone()],
            Some(plt_evaluator),
        );

        let expected0_poly = DCRTPoly::from_usize_to_constant(&params, 7);
        let mut expected1_coeffs = vec![
            <DCRTPoly as Poly>::Elem::zero(&params.modulus());
            params.ring_dimension() as usize
        ];
        expected1_coeffs[1] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 3);
        let expected1_poly = DCRTPoly::from_coeffs(&params, &expected1_coeffs);
        let mut expected2_coeffs = vec![
            <DCRTPoly as Poly>::Elem::zero(&params.modulus());
            params.ring_dimension() as usize
        ];
        expected2_coeffs[2] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 4);
        let expected2_poly = DCRTPoly::from_coeffs(&params, &expected2_coeffs);

        // verify
        assert_eq!(result.len(), 3);
        assert_eq!(result, [expected0_poly, expected1_poly, expected2_poly]);
    }

    #[test]
    fn test_isolate_coeffs() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let isolate_gadget = IsolationGadget::setup(&mut circuit, &params, 2, 10);
        let inputs = circuit.input(1);
        let isolated_coeffs = isolate_gadget.isolate_terms(&mut circuit, inputs[0]);
        circuit.output(isolated_coeffs);

        let mut input_coeffs = vec![
            <DCRTPoly as Poly>::Elem::zero(&params.modulus());
            params.ring_dimension() as usize
        ];
        input_coeffs[0] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 7);
        input_coeffs[1] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 3);
        input_coeffs[2] = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 4);
        let input_poly = DCRTPoly::from_coeffs(&params, &input_coeffs);
        let plt_evaluator = PolyPltEvaluator::new();
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[input_poly.clone()],
            Some(plt_evaluator),
        );

        let expected = vec![
            DCRTPoly::from_usize_to_constant(&params, 7),
            DCRTPoly::from_usize_to_constant(&params, 3),
            DCRTPoly::from_usize_to_constant(&params, 4),
        ];

        // verify
        assert_eq!(result.len(), 3);
        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_isolate_coeffs_circuit() {
    //     let params = DCRTPolyParams::default();
    //     let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();

    //     // only assumes binary polynomials of degree n as input
    //     let norm_bound = 1;
    //     let lt_isolate_id =
    //         LtIsolateGadget::register_general_lt_isolate_lookup(&mut circuit, &params,
    // norm_bound);

    //     let inputs = circuit.input(1);
    //     let a_poly_id = inputs[0];
    //     let n = params.ring_dimension() as usize;
    //     let isolated_coeffs =
    //         LtIsolateGadget::isolate_coeffs(&mut circuit, a_poly_id, n - 1, lt_isolate_id, n);
    //     circuit.output(isolated_coeffs);

    //     // Create a random binary polynomial
    //     let sampler = DCRTPolyUniformSampler::new();
    //     let a_poly = sampler.sample_poly(&params, &DistType::BitDist);

    //     assert!(a_poly.coeffs().len() == n);

    //     let plt_evaluator = PolyPltEvaluator::new();
    //     let result = circuit.eval(
    //         &params,
    //         &DCRTPoly::const_one(&params),
    //         &[a_poly.clone()],
    //         Some(plt_evaluator),
    //     );

    //     // verify
    //     // should return a vector of n constant polynomials
    //     assert_eq!(result.len(), n);

    //     for (i, output) in result.iter().enumerate() {
    //         let output_coeff = output.coeffs();
    //         assert_eq!(output_coeff.len(), n);
    //         for (j, coeff) in output_coeff.iter().enumerate() {
    //             if j == 0 {
    //                 assert_eq!(coeff, &a_poly.coeffs()[i]);
    //             } else {
    //                 assert_eq!(coeff, &<DCRTPoly as Poly>::Elem::zero(&params.modulus()));
    //             }
    //         }
    //     }
    // }
}
