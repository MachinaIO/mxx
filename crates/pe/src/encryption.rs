//! Encryption: `Enc(mpk, x, m)`, with no predicate.
//!
//! Under a fresh secret `s = (s_bar, 1)` and encryption session, one prelude
//! production computes
//!
//! - the anchor encoding `b = s B + e_b = s_bar B^(1) + B^(2) + e_b`;
//! - for every attribute symbol `x_t`, the gadget ciphertext `C_t = P Z_t + X^(Bz x_t) G` with `P =
//!   [B^(1); B^(2) - b]` and a fresh `Z_t` of shape `m_B x m`: `s P = -e_b`, so `s C_t = X^(Bz x_t)
//!   s G - e_b Z_t`;
//! - the message component `c_m = s d^T + e_m + floor(q/2) m = s_bar d_1 + d_2 + e_m + floor(q/2)
//!   m` for binary coefficients `m`.
//!
//! The prelude exports `C_t` under the Dynamic setup's supplied-ciphertext
//! role, and the exponent-lut Dynamic encryption, bound to the prelude and
//! to the same secret and session, adds the initial state `c_0 = s (A_0 -
//! G) + e_0`, the attribute packages' digit companions, the helpers `h_k`,
//! and the companions of the derived `C_k`. Neither production exports `s`,
//! a `Z_t`, or any sampling transcript.
use crate::{PeError, PePublicKey, setup::digest};
use mxx_backends::{
    RuntimeValue, SessionStore,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
};
use mxx_dsl::{DslContext, Mat};
use mxx_exponent_lut::{
    api::{self, EncryptBuild, StageArtifacts, StageExecution, StageGraph},
    program::InputBinding,
    rns_bgg::{
        RnsBggDynamicSetup, RnsBggEncryptionDescriptor, RnsBggPayloadSecret, RnsEncryptBuild,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv, node::IndexRange};
use num_bigint::BigInt;
use num_traits::Zero;
use std::collections::BTreeMap;

/// `b` in the prelude production.
pub(crate) const ANCHOR_ENCODING: &str = "mxx-pe/anchor-encoding";
/// `c_m` in the prelude production.
pub(crate) const MESSAGE_ENCODING: &str = "mxx-pe/message-encoding";
const ATTRIBUTE_INPUT: &str = "mxx-pe/attribute-exponents";
const MESSAGE_INPUT: &str = "mxx-pe/message-coefficients";

/// A ciphertext: the prelude production (`b`, every `C_t`, `c_m`) and the
/// Dynamic package production, both of one encryption session.
pub struct PeCiphertext {
    public_key: [u8; 32],
    prelude: StageArtifacts<()>,
    packages: StageArtifacts<RnsBggEncryptionDescriptor>,
}

impl PeCiphertext {
    /// The public key the ciphertext was encrypted under.
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public_key
    }

    /// The prelude production.
    pub const fn prelude(&self) -> &StageArtifacts<()> {
        &self.prelude
    }

    /// The Dynamic package production.
    pub const fn packages(&self) -> &StageArtifacts<RnsBggEncryptionDescriptor> {
        &self.packages
    }
}

/// The first `columns` and the next `columns` columns of a row.
pub(crate) fn split(row: Mat, columns: usize) -> (Mat, Mat) {
    let block = |start: usize| {
        row.clone().slice(
            None,
            Some(IndexRange {
                start: IntExpr::constant(start),
                end: IntExpr::constant(start + columns),
            }),
        )
    };
    (block(0), block(columns))
}

/// `Enc(mpk, x, m)` for an attribute of the class and a message of at most
/// `n` binary coefficients.
pub fn encrypt<E, S>(
    public_key: &PePublicKey,
    attribute: &[usize],
    message: &[bool],
    execution: &mut E,
    store: &mut S,
) -> Result<PeCiphertext, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    Ok(encrypt_with_secret(public_key, attribute, message, execution, store)?.0)
}

/// [`encrypt`], also returning the resident secret for tests that check the
/// encryption relations.
pub(crate) fn encrypt_with_secret<E, S>(
    public_key: &PePublicKey,
    attribute: &[usize],
    message: &[bool],
    execution: &mut E,
    store: &mut S,
) -> Result<(PeCiphertext, RnsBggPayloadSecret), PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let parameters = public_key.parameters();
    let class = parameters.class();
    class.validate_attribute(attribute)?;
    let level = parameters.level();
    let dimension = level.ring_dimension() as usize;
    if message.len() > dimension {
        return Err(PeError::Message);
    }
    let rns = parameters.rns();
    let secret = RnsEncryptBuild::new(rns, *public_key.context()).sample_payload_secret(
        execution,
        store,
        |_| Ok(()),
    )?;
    let session: [u8; 32] = rand::random();
    let prelude = prelude(public_key, &secret, attribute, message, session, execution, store)?;
    let setup = public_key.setup();
    let initial = *setup.encryption_inputs().inputs().keys().next().ok_or(PeError::Mismatch)?;
    let lhs = BTreeMap::from([(
        initial,
        InputBinding::Scalar(DCRTPoly::from_usize_to_constant(level, 1)),
    )]);
    let packages = api::encrypt(
        &setup.encryption_inputs(),
        &ParamEnv::default(),
        &lhs,
        None,
        &secret,
        None,
        public_key.hash_key(),
        RnsEncryptBuild::new(rns, *public_key.context()).with_dynamic_setup(setup)?,
        execution,
        store,
        |build| {
            <RnsEncryptBuild<'_> as EncryptBuild<DCRTPoly>>::bind_prelude(
                build,
                prelude.production().clone(),
                prelude.manifest().clone(),
            )?;
            build.bind_rhs_inputs::<DCRTPoly>(&ParamEnv::default(), &BTreeMap::new(), session)
        },
        |_| Ok(()),
    )?;
    Ok((PeCiphertext { public_key: *public_key.identity(), prelude, packages }, secret))
}

/// The prelude production of one encryption session.
fn prelude<E, S>(
    public_key: &PePublicKey,
    secret: &RnsBggPayloadSecret,
    attribute: &[usize],
    message: &[bool],
    session: [u8; 32],
    execution: &mut E,
    store: &mut S,
) -> Result<StageArtifacts<()>, PeError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let parameters = public_key.parameters();
    let class = parameters.class();
    let ring = public_key.ring()?;
    let columns = parameters.digits();
    let m_b = parameters.anchor_columns();
    let dimension = parameters.level().ring_dimension() as usize;
    let mut runtime = BTreeMap::new();
    secret.bind_runtime(&mut runtime);
    let hash_key = public_key.hash_key_input(&mut runtime)?;
    let context = DslContext::new("mxx-pe/encrypt-prelude");
    let exponents = context.int_family_input(ATTRIBUTE_INPUT, attribute.len());
    runtime.insert(
        ATTRIBUTE_INPUT.to_owned(),
        RuntimeValue::integer_values(
            attribute.iter().map(|symbol| BigInt::from(class.state_width() * symbol)).collect(),
        ),
    );
    let coefficients = context.int_family_input(MESSAGE_INPUT, dimension);
    runtime.insert(
        MESSAGE_INPUT.to_owned(),
        RuntimeValue::integer_values(
            (0..dimension)
                .map(|index| BigInt::from(u8::from(message.get(index).copied().unwrap_or(false))))
                .collect(),
        ),
    );
    let (s_bar, _) = secret.row().clone();
    let first = public_key.anchor_row()?;
    let second = public_key.anchor_second_row(hash_key.clone())?;
    let (d_1, d_2) = public_key.message_row(hash_key)?;
    // `chi`; a zero sigma (noise-free parameters) adds no error.
    let error = |columns: usize| -> Result<Mat, PeError> {
        let contract = parameters.error();
        if contract.sigma().is_zero() {
            return Ok(ring.zero((1, columns)));
        }
        Ok(ring.gaussian(
            (1, columns),
            crate::params::real_expr(contract.sigma())?,
            contract.cutoff().clone(),
        ))
    };
    let anchor = s_bar.clone() * first.clone() + second.clone() + error(m_b)?;
    let lower = second - anchor.clone();
    let gadget = parameters.rns().public_gadget(0)?;
    let programming = parameters.programming();
    let programming_sigma = crate::params::real_expr(programming.sigma())?;
    let ciphertexts = mxx_dsl::parallel(attribute.len(), |member| {
        let z = ring.gaussian(
            (m_b, 2 * columns),
            programming_sigma.clone(),
            programming.cutoff().clone(),
        );
        let symbol = ring.identity(1).multiply_monomial(exponents.at(member));
        let target = symbol * gadget.clone();
        let (top_left, top_right) = split(first.clone() * z.clone(), columns);
        let (bottom_left, bottom_right) = split(lower.clone() * z, columns);
        Ok(((top_left + target.clone(), top_right), (bottom_left, bottom_right + target)))
    })?;
    let half = ring.polynomial([IntExpr::constant(parameters.modulus() / 2u8)]);
    let message = s_bar * d_1 + d_2 + error(1)? + half * ring.from_coefficients(&coefficients);
    let graph = context
        .transferred_output(ANCHOR_ENCODING, anchor)?
        .transferred_output(
            RnsBggDynamicSetup::rhs_ciphertext_role(
                crate::branching_program::ATTRIBUTE_COORDINATE,
                true,
            ),
            ciphertexts,
        )?
        .transferred_output(MESSAGE_ENCODING, message)?
        .build()?;
    let (anchor_production, anchor_manifest) = public_key.anchor_production().clone();
    Ok(api::execute_artifacts(
        StageGraph::new(graph, runtime, BTreeMap::from([(anchor_production, anchor_manifest)])),
        &ParamEnv::default(),
        (),
        digest(b"mxx-pe/encrypt-prelude/v1", &[public_key.identity(), &session]),
        execution,
        store,
        |_| Ok(()),
    )?)
}
