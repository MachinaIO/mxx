//! The setup and preprocessing side of the smudged trapdoor decoder
//! (see [`crate::decoder`]).
//!
//! The decoded column is component 1 of the source output (its payload
//! secret component is 1) under the selection `G_(L+1)^-1(1)`, whose payload
//! is `Delta_L mu`, plus the smudge sum, whose members were
//! finalized and weighted in their root bodies. Its key is the `2 x 1`
//! matrix `A = (A_01; A_11) G_(L+1)^-1(1) + sum (A_01; A_11)_smudge`.
use super::*;
use crate::decoder::decoding::DecoderDeclaration;

/// The setup's trapdoor artifact: `.0` is the public `B`, `.1` the trapdoor, which also carries
/// `B`.
const UNMASK_TRAPDOOR: &str = "unmask/trapdoor";

/// Run the trusted unmask setup of `declaration`'s decoder of
/// `source_program`: sample the public `B` and its trapdoor at the source
/// level with trapdoor sigma `trapdoor_sigma`. Only the trusted preprocessor
/// may read the returned production; it republishes `B` and the preimage of
/// the decoder key.
#[allow(clippy::too_many_arguments)]
pub fn sample_unmask_trapdoor<E, S>(
    parameters: &RnsBggParameters,
    declaration: &DecoderDeclaration,
    source_program: &ExponentLutProgram,
    context: [u8; 32],
    trapdoor_sigma: &bigdecimal::BigDecimal,
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<StageArtifacts<RnsBggUnmaskSetupDescriptor>, ApiError>
where
    E: crate::api::StageExecution<S>,
    S: mxx_backends::SessionStore,
{
    let level = RnsBggParameters::decoder_source_level(declaration, source_program)
        .map_err(ApiError::scheme)?;
    let sigma = trapdoor_sigma
        .to_f64()
        .ok_or_else(|| ApiError::input("the unmask trapdoor sigma is not finite"))?;
    let (base, digits, bound) = parameters.unmask_gadget(level, sigma).map_err(ApiError::scheme)?;
    let ring = parameters.ring(level).map_err(ApiError::scheme)?;
    let trapdoor = crate::decoder::unmask::sample_trapdoor(
        &ring,
        decimal_sigma(trapdoor_sigma)?,
        base,
        digits,
        BigInt::from(bound.clone()),
    );
    let graph = DslContext::new("rns-bgg-unmask-setup")
        .transferred_output(UNMASK_TRAPDOOR, trapdoor)
        .map_err(ApiError::scheme)?
        .build()
        .map_err(ApiError::scheme)?;
    let descriptor = RnsBggUnmaskSetupDescriptor {
        context,
        parameters: parameters.parameter_identity(),
        decoder_commitment: *declaration.commitment(),
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
        crate::api::domain_identity(*declaration.commitment(), b"unmask-setup"),
        execution,
        store,
        before_execute,
    )
}

fn decimal_sigma(sigma: &bigdecimal::BigDecimal) -> Result<mxx_ir_core::RealExpr, ApiError> {
    crate::utils::decimal_to_real_expr(sigma)
        .map_err(|_| ApiError::input("the unmask trapdoor sigma is not a decimal"))
}

/// Import the trapdoor of `setup` at its decoding level.
pub(super) fn import_unmask_trapdoor(
    parameters: &RnsBggParameters,
    setup: &StageArtifacts<RnsBggUnmaskSetupDescriptor>,
) -> Result<Trapdoor, RnsBggError> {
    let descriptor = setup.descriptor();
    let sigma = descriptor
        .sigma
        .parse::<bigdecimal::BigDecimal>()
        .map_err(|_| RnsBggError::Binding("unmask setup sigma"))?;
    let sigma = crate::utils::decimal_to_real_expr(&sigma)
        .map_err(|_| RnsBggError::Binding("unmask setup sigma"))?;
    Ok(parameters.ring(descriptor.level)?.trapdoor_artifact_input(
        setup.production().clone(),
        format!("{UNMASK_TRAPDOOR}.1"),
        crate::decoder::unmask::UNMASK_ROWS,
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

/// The preprocessing stage's decoder: the summed unmask key `A`.
impl Decoder for RnsBggDecoder<'_, PublicKeyLowering<'_>> {
    /// The lowered smudge sum.
    type Smudge = Vec<RnsBggPublicKey>;
    /// The `2 x 1` summed key `A`.
    type Output = Mat;
    type Error = RnsBggError;

    fn finish_decoding(
        &mut self,
        declaration: &DecoderDeclaration,
        smudge: Self::Smudge,
    ) -> Result<Self::Output, Self::Error> {
        let parameters = self.lowerer.parameters;
        let level = self.lowerer.workload.level(self.source.wire)?;
        if self.source.wire != declaration.source_output() || smudge.len() != 1 {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
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
        for key in smudge {
            let (smudge_top, smudge_bottom) = decoded_key_column(&key.matrix);
            top = top + checked_modulus_convert(smudge_top, &ring)?;
            bottom = bottom + checked_modulus_convert(smudge_bottom, &ring)?;
        }
        Ok(Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![top, bottom]))
    }
}
