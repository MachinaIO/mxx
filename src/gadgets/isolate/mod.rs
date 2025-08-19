use crate::{
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use std::{collections::HashMap, marker::PhantomData};

/// Isolation gadget for extracting constant terms from a low-degree polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Isolate terms from a polynomial A(X).
    /// Returns a vector of gate IDs, where each gate contains one coefficient a_i in the constant
    /// term.
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
}
