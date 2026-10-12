//! Encryption: `Enc(mpk, x, m)`, with no predicate.
//!
//! Under a fresh secret `s = (s_bar, 1)` and encryption session, one prelude
//! production computes
//!
//! - the anchor encoding `b = s B + e_b = s_bar B^(1) + B^(2) + e_b`;
//! - for every attribute symbol `x_t`, the bottom row `c_t = X^(Bz x_t) s G - s_bar a_t + e_t` of
//!   the gadget ciphertext `C_t = [a_t; c_t]`, whose top row `a_t` is public and hashed by the
//!   setup (`RnsBggDynamicSetup::rhs_public_top`), with a fresh `e_t` from `chi`, both `1 x m`, so
//!   `s C_t = X^(Bz x_t) s G + e_t`, if a lookup step reads it, and the same ciphertexts of the
//!   distances `X^(popcount(x_t xor z))` to every symbol `z` if an add step does
//!   ([`crate::branching_program::StepKind`]). Since `a_t` is public, the digit rows of `a_t` need
//!   no companion, which halves the companions of every package;
//! - the message component `c_m = s d^T + e_m + floor(q/2) m = s_bar d_1 + d_2 + e_m + floor(q/2)
//!   m` for binary coefficients `m`.
//!
//! The prelude samples `c_t` with the Dynamic setup's `rhs_bottom` and exports
//! it under the setup's bottom-row role, and the exponent-lut Dynamic encryption, bound to the
//! prelude and to the same secret and session, adds the initial state `c_0 = s (A_0 -
//! G) + e_0`, the attribute packages' digit companions, the helpers `h_k`,
//! and the companions of the derived `C_k`. Neither production exports `s`
//! or any sampling transcript.
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
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::BigInt;
use num_traits::Zero;
use std::collections::BTreeMap;

/// `b` in the prelude production.
pub(crate) const ANCHOR_ENCODING: &str = "mxx-pe/anchor-encoding";
/// `c_m` in the prelude production.
pub(crate) const MESSAGE_ENCODING: &str = "mxx-pe/message-encoding";
const ATTRIBUTE_INPUT: &str = "mxx-pe/attribute-exponents";
const ADD_ATTRIBUTE_INPUT: &str = "mxx-pe/add-exponents";
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
    let m_b = parameters.anchor_columns();
    let dimension = parameters.level().ring_dimension() as usize;
    let mut runtime = BTreeMap::new();
    secret.bind_runtime(&mut runtime);
    let hash_key = public_key.hash_key_input(&mut runtime)?;
    let context = DslContext::new("mxx-pe/encrypt-prelude");
    // The exponents of the lookup packages, `Bz x_t`, and of the add packages of the `i`-th add
    // step, `popcount(x_i xor z)` at member `z`, for the packages the class reads.
    let mut families = Vec::new();
    if class.has_lookup_packages() {
        families.push((
            crate::branching_program::ATTRIBUTE_COORDINATE,
            ATTRIBUTE_INPUT.to_owned(),
            attribute.iter().map(|symbol| class.state_width() * symbol).collect::<Vec<_>>(),
        ));
    }
    for (read, coordinate) in
        (0..class.add_step_count()).zip(crate::branching_program::ADD_ATTRIBUTE_COORDINATE..)
    {
        let symbol = attribute[read];
        families.push((
            coordinate,
            format!("{ADD_ATTRIBUTE_INPUT}/{read}"),
            (0..class.alphabet_size())
                .map(|template| (symbol ^ template).count_ones() as usize)
                .collect(),
        ));
    }
    let exponents = families
        .iter()
        .map(|(coordinate, name, values)| {
            runtime.insert(
                name.clone(),
                RuntimeValue::integer_values(
                    values.iter().map(|value| BigInt::from(*value)).collect(),
                ),
            );
            (*coordinate, context.int_family_input(name, values.len()))
        })
        .collect::<Vec<_>>();
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
    let (d_1, d_2) = public_key.message_row(hash_key.clone())?;
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
    let anchor = s_bar.clone() * first + second + error(m_b)?;
    let mut context = context.transferred_output(ANCHOR_ENCODING, anchor)?;
    let setup = public_key.setup();
    for (coordinate, family) in exponents {
        let bottoms = mxx_dsl::parallel(family.count().clone(), |member| {
            // The bottom row `X^v s G - s_bar a_t + e_t` of `C_t = [a_t; ...]`, whose top row
            // `a_t` is the setup's: `s C_t = X^v s G + e_t` for `s = (s_bar, 1)`.
            setup
                .rhs_bottom(
                    parameters.rns(),
                    hash_key.clone(),
                    coordinate,
                    Some(member.expression()?),
                    secret.row(),
                    ring.identity(1).multiply_monomial(family.at(member)),
                )
                .map_err(|_| mxx_dsl::DslError::Schema)
        })?;
        context = context
            .transferred_output(RnsBggDynamicSetup::rhs_bottom_role(coordinate, true), bottoms)?;
    }
    let half = ring.polynomial([IntExpr::constant(parameters.modulus() / 2u8)]);
    let message = s_bar * d_1 + d_2 + error(1)? + half * ring.from_coefficients(&coefficients);
    let graph = context.transferred_output(MESSAGE_ENCODING, message)?.build()?;
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
