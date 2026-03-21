use crate::{
    circuit::{PolyVec, gate::GateId},
    lookup::{PltEvaluator, PublicLut, poly::PolyPltEvaluator},
    poly::Poly,
};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PolyVecPltEvaluator {
    pub plt_evaluator: PolyPltEvaluator,
}

impl<P: Poly + 'static> PltEvaluator<PolyVec<P>> for PolyVecPltEvaluator {
    fn public_lookup(
        &self,
        params: &P::Params,
        plt: &PublicLut<P>,
        one: &PolyVec<P>,
        input: &PolyVec<P>,
        gate_id: GateId,
        lut_id: usize,
    ) -> PolyVec<P> {
        let output_vec = input
            .slots
            .as_slice()
            .par_iter()
            .map(|input_poly| {
                self.plt_evaluator.public_lookup(
                    params,
                    plt,
                    &one.slots[0], // Assuming all slots in `one` are the same
                    input_poly,
                    gate_id,
                    lut_id,
                )
            })
            .collect();
        PolyVec::new(output_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    fn lut_output(k: u64) -> u64 {
        k % 2
    }

    #[test]
    fn test_poly_vec_plt_public_lookup_evaluates_each_coefficient_of_each_slot_with_lut() {
        let params = DCRTPolyParams::new(8, 2, 17, 1);
        let input_coeffs_by_slot: Vec<Vec<u32>> =
            vec![vec![0, 15, 1, 14, 2, 13, 3, 12], vec![4, 11, 5, 10, 6, 9, 7, 8]];
        let input = PolyVec::new(
            input_coeffs_by_slot
                .iter()
                .map(|coeffs| DCRTPoly::from_u32s(&params, coeffs))
                .collect(),
        );

        let lut_len = 16u64;
        let lut = PublicLut::<DCRTPoly>::new(
            &params,
            lut_len,
            move |params, k| {
                if k >= lut_len {
                    return None;
                }
                let y_elem = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(k));
                Some((k, y_elem))
            },
            Some((
                lut_len - 1,
                <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(lut_len - 1)),
            )),
        );

        let evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let one = PolyVec::new(vec![DCRTPoly::const_one(&params); input.len()]);
        let output = evaluator.public_lookup(&params, &lut, &one, &input, GateId(7), 42);

        let expected = PolyVec::new(
            input_coeffs_by_slot
                .iter()
                .map(|coeffs| {
                    let expected_coeffs = coeffs
                        .iter()
                        .map(|&coeff| lut_output(coeff as u64) as u32)
                        .collect::<Vec<_>>();
                    DCRTPoly::from_u32s(&params, &expected_coeffs)
                })
                .collect(),
        );

        assert_eq!(output, expected);
    }
}
