use crate::{
    circuit::gate::GateId,
    element::PolyElem,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};
use num_traits::ToPrimitive;

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly + 'static> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(
        &self,
        params: &P::Params,
        plt: &PublicLut<P>,
        _: &P,
        input: &P,
        gate_id: GateId,
        lut_id: usize,
    ) -> P {
        let output_coeffs = input
            .coeffs()
            .into_iter()
            .enumerate()
            .map(|(coeff_idx, coeff)| {
                let x_i = coeff.value().to_u64().unwrap_or_else(|| {
                    panic!(
                        "lookup input coefficient must fit in u64; gate_id: {:?}, lut_id: {:?}, coeff_idx: {:?}, coeff: {:?}",
                        gate_id, lut_id, coeff_idx, coeff
                    )
                });
                plt.get(params, x_i)
                    .unwrap_or_else(|| {
                        panic!(
                            "output of the lookup evaluation not found; gate_id: {:?}, lut_id: {:?}, coeff_idx: {:?}, input_coeff_u64: {:?}",
                            gate_id, lut_id, coeff_idx, x_i
                        )
                    })
                    .1
            })
            .collect::<Vec<_>>();
        P::from_coeffs(params, &output_coeffs)
    }
}

impl Default for PolyPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyPltEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use std::sync::{Arc, Mutex};

    fn lut_output(k: u64) -> u64 {
        k + 10
    }

    #[test]
    fn public_lookup_evaluates_each_coefficient_with_lut() {
        let params = DCRTPolyParams::new(8, 2, 17, 1);
        let input_coeffs: Vec<u32> = vec![0, 15, 1, 14, 2, 13, 3, 12];
        let input = DCRTPoly::from_u32s(&params, &input_coeffs);
        let observed_queries = Arc::new(Mutex::new(Vec::<u64>::new()));

        let lut_len = 16u64;
        let observed_queries_for_lut = Arc::clone(&observed_queries);
        let lut = PublicLut::<DCRTPoly>::new(
            &params,
            lut_len,
            move |params, k| {
                if k >= lut_len {
                    return None;
                }
                observed_queries_for_lut.lock().expect("failed to lock observed query log").push(k);
                let y_elem = <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(k));
                Some((k, y_elem))
            },
            Some((
                lut_len - 1,
                <DCRTPoly as Poly>::Elem::constant(&params.modulus(), lut_output(lut_len - 1)),
            )),
        );

        let evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let output = evaluator.public_lookup(&params, &lut, &one, &input, GateId(7), 42);

        let expected_input_queries: Vec<u64> =
            input_coeffs.iter().map(|&coeff| coeff as u64).collect();
        let expected_coeffs: Vec<u32> =
            input_coeffs.iter().map(|&coeff| lut_output(coeff as u64) as u32).collect();
        let expected = DCRTPoly::from_u32s(&params, &expected_coeffs);

        assert_eq!(
            observed_queries.lock().expect("failed to lock observed query log").as_slice(),
            expected_input_queries.as_slice()
        );
        assert_eq!(output, expected);
    }
}
