use mxx::{
    circuit::evaluable::PolyVec,
    gadgets::{
        arith::{DecomposeArithmeticGadget, MontgomeryPoly, encode_montgomery_poly_with_window},
        fhe::ring_gsw::RingGswContext,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly,
        dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams},
    },
};
use num_bigint::BigUint;

pub type MontgomeryRingGswContext = RingGswContext<GpuDCRTPoly, MontgomeryPoly<GpuDCRTPoly>>;

pub fn montgomery_ring_gsw_plaintext_ciphertext(
    params: &GpuDCRTPolyParams,
    ctx: &MontgomeryRingGswContext,
    plaintext: u64,
) -> [Vec<GpuDCRTPoly>; 2] {
    let gadget_row = MontgomeryPoly::<GpuDCRTPoly>::gadget_matrix::<GpuDCRTPolyMatrix>(
        params,
        ctx.arith_ctx.as_ref(),
        Some(ctx.active_levels),
        Some(ctx.level_offset),
    )
    .get_row(0);
    let gadget_len = gadget_row.len();
    let plaintext_poly = GpuDCRTPoly::from_biguint_to_constant(params, BigUint::from(plaintext));
    let zero = GpuDCRTPoly::const_zero(params);

    let mut top =
        gadget_row.iter().map(|entry| entry.clone() * plaintext_poly.clone()).collect::<Vec<_>>();
    top.extend((0..gadget_len).map(|_| zero.clone()));

    let mut bottom = vec![zero.clone(); gadget_len];
    bottom.extend(
        gadget_row.iter().map(|entry| entry.clone() * plaintext_poly.clone()).collect::<Vec<_>>(),
    );

    [top, bottom]
}

pub fn montgomery_ciphertext_inputs_from_native(
    params: &GpuDCRTPolyParams,
    ctx: &MontgomeryRingGswContext,
    ciphertext: &[Vec<GpuDCRTPoly>; 2],
) -> Vec<PolyVec<GpuDCRTPoly>> {
    ciphertext
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|poly| {
            let coeff_encodings = poly
                .coeffs_biguints()
                .into_iter()
                .map(|coeff| {
                    encode_montgomery_poly_with_window::<GpuDCRTPoly>(
                        ctx.arith_ctx.carry_arith_ctx.limb_bit_size,
                        params,
                        &coeff,
                        Some(ctx.active_levels),
                        ctx.level_offset,
                    )
                })
                .collect::<Vec<_>>();
            let encoded_len = coeff_encodings
                .first()
                .map(|encoding| encoding.len())
                .expect("Montgomery Ring-GSW polynomials must have at least one coefficient");
            (0..encoded_len)
                .map(|gate_idx| {
                    PolyVec::new(
                        coeff_encodings
                            .iter()
                            .map(|encoding| encoding[gate_idx].clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
