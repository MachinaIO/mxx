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
