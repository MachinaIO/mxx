//! Hash-sampled public matrices and the RHS and LUT-switching ciphertexts
//! derived from them.
use super::*;

pub(super) fn public_lut_branch<D: super::RightAction>(
    public_data: RnsBggMatrix,
    switched_key_decomposition: ((D, D), (D, D)),
) -> RnsBggMatrix {
    multiply_public_action(public_data, switched_key_decomposition)
}

pub(in crate::rns_bgg) fn matrix_from_hash(
    ring: &mxx_dsl::Ring,
    key: Bytes,
    tag: impl Into<mxx_dsl::HashTag>,
    columns: usize,
) -> RnsBggMatrix {
    // Each component has its own tag, so the four components are independent.
    let tag = tag.into();
    let component = |index: u64| {
        let mut tag = tag.clone();
        tag.push("component");
        tag.push(mxx_ir_core::IntExpr::constant(index));
        ring.hash_matrix(key.clone(), tag, (1, columns))
    };
    ((component(0), component(1)), (component(2), component(3)))
}

/// The deterministic public `A_sigma` of evaluation level `level`. Every
/// stage that uses it samples it from the shared hash key, so preprocess,
/// encryption, and online evaluation derive the same matrix without
/// publishing it.
/// `sigma` is always a runtime integer tag component, so a constant sigma and
/// one selected by a loop index derive the same tag.
pub(in crate::rns_bgg) fn lut_public_matrix(
    parameters: &RnsBggParameters,
    key: Bytes,
    level: usize,
    sigma: Int,
) -> Result<RnsBggMatrix, RnsBggError> {
    let ring = parameters.ring(level)?;
    let columns = parameters.component_columns(level)?;
    let mut tag = mxx_dsl::HashTag::from(b"rns-bgg/lut-a".to_vec());
    tag.push(mxx_ir_core::IntExpr::constant(level));
    tag.push(sigma);
    Ok(matrix_from_hash(&ring, key, tag, columns))
}

/// The member enters the tag as an index expression in every stage, so a
/// member read by a loop index and one read by a table of indices share it.
fn input_hash_tag(wire: WireId, level: usize, member: Option<IntExpr>) -> mxx_dsl::HashTag {
    let mut tag = mxx_dsl::HashTag::from(format!("rns-bgg/input/{wire:?}/{level}").into_bytes());
    if let Some(member) = member {
        tag.push("member");
        tag.push(member);
    }
    tag
}

/// The top row of RHS `rhs`'s compact ciphertext consumed at `level`, or of
/// member `member` of a family RHS. It is sampled from the hash key, so only
/// the bottom row is stored and each reader derives this one.
///
/// The tag names the parameter set, the Program, the RHS, and the member, so
/// no two ciphertexts under one payload secret share a top row: the bottom
/// rows of two that did would differ by their plaintext difference times the
/// secret, revealing whether the plaintexts differ.
pub(crate) fn rhs_top(
    parameters: &RnsBggParameters,
    program: crate::program::ExponentLutProgramId,
    hash_key: Bytes,
    rhs: RhsId,
    level: usize,
    member: Option<IntExpr>,
) -> Result<RnsBggRow, RnsBggError> {
    let next_level = level.checked_add(1).ok_or(RnsBggError::Parameters("RHS level overflow"))?;
    let ring = parameters.ring(next_level)?;
    let columns = parameters.component_columns(next_level)?;
    // The parameter set and the Program are integer components, so chains of one refresh (which
    // differ in them) differ only in integers.
    let mut tag =
        mxx_dsl::HashTag::from(format!("rns-bgg/rhs/{rhs:?}/{level}/{next_level}").into_bytes());
    tag.push("parameters");
    crate::rns_bgg::push_digest(&mut tag, &parameters.parameter_identity());
    tag.push("program");
    crate::rns_bgg::push_digest(&mut tag, program.as_bytes());
    if let Some(member) = member {
        tag.push("member");
        tag.push(member);
    }
    let component = |index: u64| {
        let mut tag = tag.clone();
        tag.push("component");
        tag.push(mxx_ir_core::IntExpr::constant(index));
        ring.hash_matrix(hash_key.clone(), tag, (1, columns))
    };
    Ok((component(0), component(1)))
}

pub(super) fn decompose_matrix(
    parameters: &RnsBggParameters,
    level: usize,
    matrix: RnsBggMatrix,
) -> Result<RnsBggDigits, RnsBggError> {
    let p = parameters.level(level)?;
    let base = 1u64
        .checked_shl(p.base_bits())
        .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
    let decomposition_count = p.modulus_digits();
    Ok((
        (
            matrix.0.0.decompose(base, decomposition_count),
            matrix.0.1.decompose(base, decomposition_count),
        ),
        (
            matrix.1.0.decompose(base, decomposition_count),
            matrix.1.1.decompose(base, decomposition_count),
        ),
    ))
}

/// The RHS ciphertext `C` of `plaintext` consumed at `level`, sampled at
/// `q_(level+1)` with top row `top()` under payload row `payload`:
/// `(t_bar, 1) C = Delta_(level+1) y t H_(level+1) + e_C`. The top row is
/// sampled after the target, so the Static preprocessing graph keeps the
/// operation order its exported correctness claims state.
pub(in crate::rns_bgg) fn rhs_ciphertext(
    parameters: &RnsBggParameters,
    level: usize,
    payload: &RnsBggRow,
    plaintext: Mat,
    top: impl FnOnce() -> Result<RnsBggRow, RnsBggError>,
) -> Result<RnsBggMatrix, RnsBggError> {
    let next_level =
        level.checked_add(1).ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
    let next_ring = parameters.ring(next_level)?;
    let target_columns = parameters.component_columns(next_level)?;
    // Plaintext handles are canonicalized in the caller's natural input
    // ring (currently level zero).  Do not require them to already carry
    // the RHS consumer's logical level: the ciphertext is intentionally
    // sampled at q_(level + 1), and this conversion is the boundary where
    // the plaintext is represented in that ring.
    let nu = checked_modulus_convert(plaintext, &next_ring)?;
    let payload = checked_modulus_convert_row(payload.clone(), &next_ring)?;
    let gadget = parameters.state_gadget(next_level)?;
    let scale = parameters.scalar(next_level, parameters.message_scale(next_level)?)?;
    let target = (payload.0.clone() * gadget.clone(), payload.1.clone() * gadget);
    let desired = (target.0 * scale.clone() * nu.clone(), target.1 * scale * nu);
    super::ciphertext_matrix(parameters, next_level, target_columns, payload.0, top()?, desired)
}

/// The LUT switching ciphertext `C_sigma` of branch `sigma` of `table` for
/// a LUT evaluated at `level`, sampled at `q_(level+1)` with top row `top()`
/// under payload row `payload = (t_bar, 1)`:
/// `(sigma(t_bar), 1) C_sigma = Delta_(level+1) a_sigma t H_(level+1) + e`.
/// The operations keep the order the exported correctness claims of the
/// Static preprocessing graph state.
pub(in crate::rns_bgg) fn lut_switching_ciphertext(
    parameters: &RnsBggParameters,
    level: usize,
    table: &LutTable,
    payload: &RnsBggRow,
    sigma: IntExpr,
    top: impl FnOnce() -> Result<RnsBggRow, RnsBggError>,
) -> Result<RnsBggMatrix, RnsBggError> {
    let next_level = level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
    let coefficient = super::EncodingLowering::lut_coefficient(
        parameters,
        next_level,
        table.input_layout(),
        table.output_representation(),
        sigma.clone(),
        table.values().ok_or(RnsBggError::Binding(
            "preprocessing needs the LUT values, which are withheld",
        ))?,
    )?;
    // C_sigma is sampled in the next ring and constrained by the payload
    // secret. Its signed switching decomposition has each compact digit
    // centered-rebased to the source ring; the full matrix is never rebased
    // as a dense matrix.
    let payload = checked_modulus_convert_row(payload.clone(), &parameters.ring(next_level)?)?;
    let representation = table.output_representation();
    let (scale, gadget) = parameters.lut_switching_target(next_level, representation)?;
    let scale = parameters.scalar(next_level, scale)?;
    let target = lut_payload_target(scale, coefficient, payload.clone(), gadget);
    let columns = parameters.lut_output_columns(next_level, representation)?;
    let top = top()?;
    let prefix = payload.0.ring_automorphism(sigma);
    let error = super::encoding::fresh_error(parameters, next_level, columns)?;
    Ok((
        top.clone(),
        (-prefix.clone() * top.0 + target.0 + error.0, -prefix * top.1 + target.1 + error.1),
    ))
}

fn lut_payload_target(scale: Mat, coefficient: Mat, payload: RnsBggRow, gadget: Mat) -> RnsBggRow {
    (
        scale.clone() * coefficient.clone() * payload.0 * gadget.clone(),
        scale * coefficient * payload.1 * gadget,
    )
}

/// The public key of input `wire` at `level`, or of member `member` of an
/// input family. It is sampled from the hash key alone, so every stage derives
/// it where it needs it and no stage stores it.
pub(crate) fn sampled_key(
    parameters: &RnsBggParameters,
    level: usize,
    hash_key: Bytes,
    wire: WireId,
    knowledge: AttributeKnowledge,
    member: Option<IntExpr>,
) -> Result<RnsBggPublicKey, RnsBggError> {
    let ring = parameters.ring(level)?;
    let columns = parameters.key_component_columns(level, knowledge)?;
    Ok(RnsBggPublicKey::new(
        wire,
        matrix_from_hash(&ring, hash_key, input_hash_tag(wire, level, member), columns),
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    use mxx_ir_core::ParamEnv;

    #[test]
    fn rhs_payload_ciphertext_has_zero_noise_private_product_residual() {
        // This is the scalar oracle for the graph construction above.  The
        // top sample U cancels from the payload product; the remaining term
        // is the desired next-level payload target (up to the configured
        // fresh error).
        let p = 11i64;
        let q = 121i64;
        let nu = 3i64;
        let delta_next = 5i64;
        let gadget_next = 7i64;
        let r0 = 2i64;
        let r1 = 1i64;
        let u = 4i64;
        let desired = nu * delta_next * gadget_next;
        let bottom = desired - r0 * u;
        let payload_product = (r0 * u + r1 * bottom).rem_euclid(p);
        assert_eq!(payload_product, desired.rem_euclid(p));

        let private_input = 9i64;
        let product = (private_input * payload_product).rem_euclid(q);
        assert_eq!(product.rem_euclid(p), (private_input * desired).rem_euclid(p));
        assert_ne!(desired.rem_euclid(p), 0);
        assert!(q > p && p > 1);
    }

    #[test]
    fn equal_width_lut_tables_produce_distinct_public_branch_coefficients() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let constant = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 0],
        )
        .unwrap();
        let identity = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(constant.flatten(), identity.flatten());
    }

    #[test]
    fn public_lut_branch_is_helper_action_only() {
        let ring = crate::test_support::explicit_ring(&[17], 4);
        let data = (
            (ring.input("lut-data-00", (1, 1)), ring.input("lut-data-01", (1, 1))),
            (ring.input("lut-data-10", (1, 1)), ring.input("lut-data-11", (1, 1))),
        );
        let switching_decomposition = (
            (
                ring.preimage_input("lut-switching-decomposition-00", (1, 1), 17),
                ring.preimage_input("lut-switching-decomposition-01", (1, 1), 17),
            ),
            (
                ring.preimage_input("lut-switching-decomposition-10", (1, 1), 17),
                ring.preimage_input("lut-switching-decomposition-11", (1, 1), 17),
            ),
        );
        let branch = public_lut_branch(data.clone(), switching_decomposition.clone());
        let expected = multiply_public_action(data, switching_decomposition);
        assert_eq!(branch.0.0.matrix_type(), expected.0.0.matrix_type());
        let graph = DslContext::new("public-lut-branch")
            .transferred_output("branch", branch)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn lut_payload_target_uses_distinct_r0_and_r1_components() {
        let ring = crate::test_support::explicit_ring(&[257], 4);
        let r0 = ring.input("lut-r0", (1, 1));
        let r1 = ring.identity(1);
        let one = ring.identity(1);
        let target = lut_payload_target(one.clone(), one, (r0, r1), ring.identity(1));
        assert_ne!(target.0.flatten(), target.1.flatten());
    }

    #[test]
    fn switching_decomposition_is_centered_rebased_into_the_source_ring() {
        use crate::test_support::{explicit_ring, modulus_of};
        let next = explicit_ring(&[17], 4);
        let compact = next.preimage_input("lut-switching-decomposition-next", (1, 1), 17);
        let decomposition = ((compact.clone(), compact.clone()), (compact.clone(), compact));
        let rebased = super::centered_rebase_compact(decomposition, &explicit_ring(&[257], 4));
        assert_eq!(modulus_of(rebased.0.0.matrix_type()), BigInt::from(257u16));
        assert_eq!(modulus_of(rebased.1.1.matrix_type()), BigInt::from(257u16));
    }
}
