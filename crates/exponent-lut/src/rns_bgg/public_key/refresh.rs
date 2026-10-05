//! The setup and preprocessing side of the masked trapdoor-decoder refresh
//! (see [`crate::noise_refresh`]).
//!
//! The decoded column is component 1 of the source output (its payload
//! secret component is 1) under the selection `G_(L+1)^-1(1)`, whose payload
//! is `Delta_L mu`, plus the two mask family sums, whose members were
//! finalized and weighted in their root bodies. Its key is the `2 x 1`
//! matrix `A = (A_01; A_11) G_(L+1)^-1(1) + sum (A_01; A_11)_mask`.
use super::*;
use crate::noise_refresh::refresh::RefreshDeclaration;

/// The setup's trapdoor artifact: `.0` is the public `B`, `.1` the trapdoor.
const DECODER_TRAPDOOR: &str = "decoder/trapdoor";

/// Run the trusted decoder setup of `declaration`'s refresh of
/// `source_program`: sample the public `B` and its trapdoor at the source
/// level with trapdoor sigma `trapdoor_sigma`. Only the trusted preprocessor
/// may read the returned production; it republishes `B` and the preimage of
/// the refresh key.
#[allow(clippy::too_many_arguments)]
pub fn sample_refresh_decoder<E, S>(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    source_program: &ExponentLutProgram,
    context: [u8; 32],
    trapdoor_sigma: &bigdecimal::BigDecimal,
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<RnsBggDecoderSetupDescriptor>, ApiError>
where
    E: crate::api::StageExecution<S>,
    S: mxx_backends::SessionStore,
{
    let level = RnsBggParameters::refresh_source_level(declaration, source_program)
        .map_err(ApiError::scheme)?;
    let sigma = trapdoor_sigma
        .to_f64()
        .ok_or_else(|| ApiError::input("the decoder trapdoor sigma is not finite"))?;
    let (base, digits, bound) =
        parameters.refresh_decoder_gadget(level, sigma).map_err(ApiError::scheme)?;
    let ring = parameters.ring(level).map_err(ApiError::scheme)?;
    let trapdoor = crate::noise_refresh::decoder::sample_trapdoor(
        &ring,
        decimal_sigma(trapdoor_sigma)?,
        base,
        digits,
        BigInt::from(bound.clone()),
    );
    let graph = DslContext::new("rns-bgg-refresh-decoder-setup")
        .transferred_output(DECODER_TRAPDOOR, trapdoor)
        .map_err(ApiError::scheme)?
        .build()
        .map_err(ApiError::scheme)?;
    let descriptor = RnsBggDecoderSetupDescriptor {
        context,
        parameters: parameters.parameter_identity(),
        refresh_commitment: *declaration.commitment(),
        level,
        sigma: trapdoor_sigma.normalized().to_string(),
        base,
        digits,
        bound,
    };
    crate::api::execute_artifacts(
        StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        descriptor,
        crate::api::domain_identity(*declaration.commitment(), b"refresh-decoder-setup"),
        execution,
        store,
        before_execute,
    )
}

fn decimal_sigma(sigma: &bigdecimal::BigDecimal) -> Result<mxx_ir_core::RealExpr, ApiError> {
    crate::utils::decimal_to_real_expr(sigma)
        .map_err(|_| ApiError::input("the decoder trapdoor sigma is not a decimal"))
}

/// Import the trapdoor of `setup` at its decoding level.
pub(super) fn import_decoder_trapdoor(
    parameters: &RnsBggParameters,
    setup: &StageArtifacts<RnsBggDecoderSetupDescriptor>,
) -> Result<Trapdoor, RnsBggError> {
    let descriptor = setup.descriptor();
    let sigma = descriptor
        .sigma
        .parse::<bigdecimal::BigDecimal>()
        .map_err(|_| RnsBggError::Binding("decoder setup sigma"))?;
    let sigma = crate::utils::decimal_to_real_expr(&sigma)
        .map_err(|_| RnsBggError::Binding("decoder setup sigma"))?;
    Ok(parameters.ring(descriptor.level)?.trapdoor_artifact_input(
        setup.production().clone(),
        format!("{DECODER_TRAPDOOR}.0"),
        format!("{DECODER_TRAPDOOR}.1"),
        crate::noise_refresh::decoder::DECODER_ROWS,
        sigma,
        descriptor.base,
        descriptor.digits,
        BigInt::from(descriptor.bound.clone()),
    ))
}

/// The component-1 key column `(A_01; A_11)` of `matrix`.
fn decoded_key_column(matrix: &RnsBggMatrix) -> (Mat, Mat) {
    (matrix.0.1.clone(), matrix.1.1.clone())
}

/// The preprocessing stage's refresh: the summed decoder key `A`.
impl NoiseRefresh for RnsBggRefresh<'_, PublicKeyLowering<'_>> {
    /// The two lowered mask family sums.
    type Mask = Vec<RnsBggPublicKey>;
    /// The `2 x 1` summed key `A`.
    type Output = Mat;
    type Error = RnsBggError;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        mask: Self::Mask,
    ) -> Result<Self::Output, Self::Error> {
        let parameters = self.lowerer.parameters;
        let level = self.lowerer.workload.level(self.source.wire)?;
        if self.source.wire != declaration.source_output() ||
            mask.len() != crate::noise_refresh::refresh::MaskFamily::ALL.len()
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let ring = parameters.ring(level)?;
        let plaintext = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let selection = parameters.gadget_inverse(
            plaintext,
            level,
            parameters.ring(plaintext)?.polynomial([IntExpr::constant(1)]),
        )?;
        let (top, bottom) = decoded_key_column(&self.source.matrix);
        let (mut top, mut bottom) =
            (top.mul_small_rhs(selection.clone()), bottom.mul_small_rhs(selection));
        for key in mask {
            let (mask_top, mask_bottom) = decoded_key_column(&key.matrix);
            top = top + checked_modulus_convert(mask_top, &ring)?;
            bottom = bottom + checked_modulus_convert(mask_bottom, &ring)?;
        }
        Ok(Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![top, bottom]))
    }
}
