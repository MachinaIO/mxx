use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};

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
        let input_const = input.const_coeff_u64();
        let output_const = plt
            .get(params, input_const)
            .unwrap_or_else(|| {
                panic!(
                    "lookup output not found for input constant term; gate_id: {:?}, lut_id: {:?}, input_const_u64: {:?}",
                    gate_id, lut_id, input_const
                )
            })
            .1;
        P::from_elem_to_constant(params, &output_const)
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
    fn test_poly_plt_public_lookup_uses_only_constant_term() {
        let params = DCRTPolyParams::new(8, 2, 17, 1);
        let input_coeffs: Vec<u32> = vec![0, 15, 1, 14, 2, 13, 3, 12];
        let input = DCRTPoly::from_u32s(&params, &input_coeffs);

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

        let evaluator = PolyPltEvaluator::new();
        let one = DCRTPoly::const_one(&params);
        let output = evaluator.public_lookup(&params, &lut, &one, &input, GateId(7), 42);

        let expected =
            DCRTPoly::from_usize_to_constant(&params, lut_output(input_coeffs[0] as u64) as usize);

        assert_eq!(output, expected);
    }
}
