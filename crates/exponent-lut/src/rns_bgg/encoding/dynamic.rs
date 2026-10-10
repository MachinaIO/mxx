//! Dynamic packages: their generation at encryption and their evaluation at
//! online evaluation.
//!
//! Encryption under a [`RnsBggDynamicSetup`] produces, under one secret
//! `s = (s_bar, 1)` and in one encryption session:
//!
//! - every input coordinate's encoding, under its setup key;
//! - one package per RHS coordinate (member): its gadget ciphertext `C` with `s C = nu s G + e_C`
//!   (sampled here from the plaintext, or supplied by the enclosing scheme through the prelude
//!   production) and the digit companions `z_{a,j} = s (B_{a,j} - d_{a,j} G) + eps` of `d =
//!   Dec_G(C)`;
//! - for every nonidentity automorphism `k` of the setup, the helper `h_k = s H_k - s_k G + e_h`
//!   and the digit companions, under `s_k = (sigma_k(s_bar), 1)`, of the switching ciphertext `C_k`
//!   that every reader derives from the helper of `k^-1` (see
//!   [`crate::rns_bgg::dynamic::automorphism_ciphertext`]); `C_k` itself is not published.
//!
//! None of these names a Program, a LUT, or a logical level, so one
//! ciphertext serves every Program bound to the setup.
use super::*;
use crate::rns_bgg::dynamic::{
    automorphism_ciphertext, lut_automorphisms, scalar_action, scalar_action_matrix,
    scalar_action_row, select,
};

/// Artifact role of the encoding of input coordinate `coordinate`.
pub(crate) fn dynamic_input_role(coordinate: u32) -> String {
    format!("dynamic/input/{coordinate}")
}

/// Artifact role of the digit companions of RHS coordinate `coordinate`.
pub(crate) fn dynamic_companion_role(coordinate: u32) -> String {
    format!("dynamic/rhs/{coordinate}/digit-encoding")
}

/// Artifact role of the automorphism helpers.
pub(crate) const AUTOMORPHISM_HELPER_ROLE: &str = "dynamic/automorphism/helper";

/// Artifact role of the digit companions of the automorphism switching
/// ciphertexts.
pub(crate) const AUTOMORPHISM_COMPANION_ROLE: &str = "dynamic/automorphism/digit-encoding";

fn quotient(index: &IntExpr, divisor: usize) -> IntExpr {
    IntExpr::FloorDiv(Box::new(index.clone()), Box::new(IntExpr::constant(divisor))).canonicalize()
}

fn remainder(index: IntExpr, modulus: usize) -> IntExpr {
    IntExpr::Rem(Box::new(index), Box::new(IntExpr::constant(modulus))).canonicalize()
}

impl<'a> RnsEncryptBuild<'a> {
    /// This builder for Dynamic encryption under `setup`, whose parameters
    /// and context must be the builder's. Dynamic encryption binds no
    /// Program and no preprocessing artifact.
    pub fn with_dynamic_setup(mut self, setup: &RnsBggDynamicSetup) -> Result<Self, RnsBggError> {
        setup.validate_parameters(self.parameters)?;
        if setup.context() != &self.context {
            return Err(RnsBggError::Binding("the Dynamic setup has another context"));
        }
        self.dynamic_setup = Some(Arc::new(setup.clone()));
        Ok(self)
    }

    pub(super) fn bound_dynamic_setup(&self) -> Result<&Arc<RnsBggDynamicSetup>, ApiError> {
        self.dynamic_setup.as_ref().ok_or_else(|| {
            ApiError::input("Dynamic encryption needs its setup (with_dynamic_setup)")
        })
    }

    /// Bind the RHS plaintexts of a Dynamic encryption, keyed by setup RHS
    /// coordinate, and its session identifier, e.g. in the `api::encrypt`
    /// compose callback.
    ///
    /// Every [`RnsBggPackageSource::Sampled`] coordinate needs a plaintext;
    /// a [`RnsBggPackageSource::Supplied`] coordinate takes none, and its
    /// gadget ciphertexts must be exported by the production bound with
    /// [`EncryptBuild::bind_prelude`] under
    /// [`RnsBggDynamicSetup::rhs_ciphertext_role`]. `session` is a fresh
    /// public identifier of this encryption request, retained only to resume
    /// it: it selects the sampling session, so every encryption under one
    /// setup draws its own secret-dependent material.
    pub fn bind_rhs_inputs<P: Poly>(
        &mut self,
        parameters: &ParamEnv,
        rhs: &PlaintextInputs<u32, P>,
        session: [u8; 32],
    ) -> Result<(), ApiError> {
        if self.parameters.rhs_mode() != RhsMode::Dynamic {
            return Err(ApiError::input("Static RHS plaintexts are bound by preprocessing"));
        }
        if self.dynamic_rhs.is_some() {
            return Err(ApiError::input("Dynamic RHS inputs are already bound"));
        }
        let setup = self.bound_dynamic_setup()?.clone();
        if rhs.keys().any(|coordinate| !setup.rhs().contains_key(coordinate)) {
            return Err(ApiError::input("an RHS binding is not a setup coordinate"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let prefix = super::plaintext_input_prefix();
        let mut values = BTreeMap::new();
        for (coordinate, specification) in setup.rhs() {
            let value = match (&specification.source, rhs.get(coordinate)) {
                (RnsBggPackageSource::Supplied { .. }, None) => DynamicRhsValue::Supplied,
                (RnsBggPackageSource::Supplied { .. }, Some(_)) => {
                    return Err(ApiError::input("a supplied RHS coordinate takes no plaintext"));
                }
                (RnsBggPackageSource::Sampled, None) => {
                    return Err(ApiError::input("missing Dynamic RHS plaintext"));
                }
                (RnsBggPackageSource::Sampled, Some(value)) => {
                    match (&specification.shape, value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {}
                        (InputBinding::Family(_), InputBinding::Family(members)) => {
                            if resolved_input_cardinality(&specification.shape, parameters)
                                .map_err(ApiError::scheme)? !=
                                Some(members.len())
                            {
                                return Err(ApiError::input(
                                    "RHS family count does not match the setup",
                                ));
                            }
                        }
                        _ => return Err(ApiError::input("RHS plaintext shape does not match")),
                    }
                    DynamicRhsValue::Plaintext(
                        super::plaintext_inputs(
                            &ring,
                            &format!("{prefix}/rhs/{coordinate}"),
                            value,
                            &mut self.runtime,
                        )
                        .map_err(ApiError::input)?,
                    )
                }
            };
            values.insert(*coordinate, value);
        }
        self.dynamic_rhs = Some((session, values));
        Ok(())
    }

    pub(super) fn bound_dynamic_rhs(&self) -> Result<&BTreeMap<u32, DynamicRhsValue>, ApiError> {
        self.dynamic_rhs.as_ref().map(|(_, values)| values).ok_or_else(|| {
            ApiError::input("Dynamic encryption requires bind_rhs_inputs with a session identifier")
        })
    }

    /// Export the gadget ciphertexts of every sampled RHS family from one
    /// root `parallel` per family into `graph`, the encryption prelude,
    /// under `name`. With no in-graph consumer, each family is an
    /// export-only streamed port. Returns whether any family was exported.
    pub(super) fn dynamic_family_ciphertexts(
        &self,
        mut graph: DslContext,
        name: impl Fn(String) -> String,
    ) -> Result<(DslContext, bool), ApiError> {
        let setup = self.bound_dynamic_setup()?;
        let payload = &self
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("secret is not bound"))?
            .row;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let mut exported = false;
        for (coordinate, value) in self.bound_dynamic_rhs()? {
            let DynamicRhsValue::Plaintext(InputBinding::Family(family)) = value else {
                continue;
            };
            let ciphertexts = mxx_dsl::parallel(family.count().clone(), |member| {
                self.sampled_rhs_ciphertext(setup, payload, family.member(&ring, member))
                    .map_err(|_| DslError::Schema)
            })
            .map_err(ApiError::scheme)?;
            graph = graph
                .transferred_output(
                    name(RnsBggDynamicSetup::rhs_ciphertext_role(*coordinate, true)),
                    ciphertexts,
                )
                .map_err(ApiError::scheme)?;
            exported = true;
        }
        Ok((graph, exported))
    }

    /// The sampled gadget ciphertext `C = [a; nu s_bar G - s_bar a + e_C]`
    /// of plaintext `nu`, with a uniform top row `a` and its error under the
    /// setup's ciphertext contract, so `s C = nu s G + e_C`.
    fn sampled_rhs_ciphertext(
        &self,
        setup: &RnsBggDynamicSetup,
        payload: &RnsBggRow,
        plaintext: Mat,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let ring = self.parameters.ring(0)?;
        let columns = self.parameters.component_columns(0)?;
        let secret = reduce_secret(self.parameters, payload, 0)?;
        let gadget = self.parameters.public_gadget(0)?;
        let target =
            (secret.0.clone() * gadget.clone() * plaintext.clone(), secret.1 * gadget * plaintext);
        super::ciphertext_matrix(
            self.parameters,
            0,
            columns,
            secret.0,
            uniform_row(&ring, columns),
            target,
            &setup.distributions().ciphertext,
        )
    }

    /// The Dynamic material of this encryption, added to `graph` under
    /// `name`: the input encodings, the RHS packages, and the automorphism
    /// packages of the setup.
    pub(super) fn dynamic_material(
        &self,
        mut graph: DslContext,
        hash_key: &Bytes,
        name: impl Fn(String) -> String,
    ) -> Result<DslContext, ApiError> {
        let setup = self.bound_dynamic_setup()?.clone();
        let payload = self
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("secret is not bound"))?
            .row
            .clone();
        graph = self.dynamic_inputs(graph, &setup, hash_key, &name)?;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let columns = self.parameters.component_columns(0).map_err(ApiError::scheme)?;
        let secret = |_: IntExpr| Ok(payload.clone());
        for (coordinate, value) in self.bound_dynamic_rhs()? {
            let shape = &setup.rhs()[coordinate].shape;
            let members = resolved_input_cardinality(shape, &ParamEnv::default())
                .map_err(ApiError::scheme)?;
            let family = members.is_some();
            let package = |member: IntExpr| RnsBggPackage::Rhs {
                coordinate: *coordinate,
                member: family.then_some(member),
            };
            // The gadget ciphertexts: computed here for a sampled scalar,
            // otherwise read from the prelude production that exports them.
            let ciphertexts = match value {
                DynamicRhsValue::Plaintext(InputBinding::Scalar(plaintext)) => {
                    let ciphertext = self
                        .sampled_rhs_ciphertext(&setup, &payload, plaintext.clone())
                        .map_err(ApiError::scheme)?;
                    graph = graph
                        .transferred_output(
                            name(RnsBggDynamicSetup::rhs_ciphertext_role(*coordinate, false)),
                            ciphertext.clone(),
                        )
                        .map_err(ApiError::scheme)?;
                    PackageCiphertexts::Scalar(ciphertext)
                }
                _ => {
                    let (prelude, _) = self.prelude.as_ref().ok_or_else(|| {
                        ApiError::input("Dynamic RHS ciphertexts need the prelude production")
                    })?;
                    let role = RnsBggDynamicSetup::rhs_ciphertext_role(*coordinate, family);
                    let ports =
                        (0..4).map(|port| name(format!("{role}.{port}"))).collect::<Vec<_>>();
                    match members {
                        None => PackageCiphertexts::Scalar(prelude_matrix(
                            &ring, prelude, &ports, columns,
                        )),
                        Some(count) => PackageCiphertexts::Family(
                            prelude_matrix_family(&ring, prelude, &ports, count, columns)
                                .map_err(ApiError::scheme)?,
                        ),
                    }
                }
            };
            let companions = self
                .companion_encodings(
                    &setup,
                    members.unwrap_or(1),
                    &|member| ciphertexts.member(member),
                    &package,
                    &secret,
                    hash_key,
                    0,
                )
                .map_err(ApiError::scheme)?;
            graph = graph
                .transferred_output(name(dynamic_companion_role(*coordinate)), companions)
                .map_err(ApiError::scheme)?;
        }
        // The switching packages of `switching_automorphisms` and the helpers of
        // `helper_automorphisms` (all of `H \ {1}` without a LUT radix).
        let automorphisms = setup.switching_automorphisms();
        if automorphisms.is_empty() {
            return Ok(graph);
        }
        let helpers =
            self.automorphism_helpers(&setup, &payload, hash_key).map_err(ApiError::scheme)?;
        let inverses = automorphisms
            .iter()
            .map(|k| setup.automorphism_inverse(*k))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::scheme)?;
        let inverse_members = inverses
            .iter()
            .map(|k| setup.helper_member(*k))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::scheme)?;
        // Every `C_k` once, from the helper of `k^-1`; the companion lanes
        // read the member of their package.
        let ciphertexts = mxx_dsl::parallel(automorphisms.len(), |member| {
            let member = member.expression()?;
            automorphism_ciphertext(
                self.parameters,
                &setup,
                hash_key.clone(),
                select(&automorphisms, member.clone()),
                select(&inverses, member.clone()),
                stored_row(
                    self.parameters,
                    &setup,
                    helpers.at(Int::evaluate(select(&inverse_members, member))),
                )
                .map_err(|_| DslError::Schema)?,
            )
            .map_err(|_| DslError::Schema)
        })
        .map_err(ApiError::scheme)?;
        let automorphed_secret = |member: IntExpr| {
            let secret = reduce_secret(self.parameters, &payload, 0)?;
            Ok((secret.0.ring_automorphism(select(&automorphisms, member)), secret.1))
        };
        let companions = self
            .companion_encodings(
                &setup,
                automorphisms.len(),
                &|member| ciphertexts.at(member),
                &|member| RnsBggPackage::Automorphism(select(&automorphisms, member)),
                &automorphed_secret,
                hash_key,
                super::automorphism_public_rows(self.parameters, 0).map_err(ApiError::scheme)?,
            )
            .map_err(ApiError::scheme)?;
        graph = graph
            .transferred_output(name(AUTOMORPHISM_HELPER_ROLE.to_owned()), helpers)
            .map_err(ApiError::scheme)?
            .transferred_output(name(AUTOMORPHISM_COMPANION_ROLE.to_owned()), companions)
            .map_err(ApiError::scheme)?;
        Ok(graph)
    }

    /// Encode every input coordinate of the setup under its setup key at the
    /// setup ring, with its error under the setup's input contract.
    fn dynamic_inputs(
        &self,
        mut graph: DslContext,
        setup: &RnsBggDynamicSetup,
        hash_key: &Bytes,
        name: &impl Fn(String) -> String,
    ) -> Result<DslContext, ApiError> {
        let payload = self
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("payload secret is not bound"))?;
        let mask =
            self.mask_secret.as_ref().ok_or_else(|| ApiError::input("mask secret is not bound"))?;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let contract = &setup.distributions().input;
        for (coordinate, declaration) in setup.inputs() {
            let public = matches!(declaration.knowledge(), AttributeKnowledge::PublicPayload);
            let value = self
                .plaintexts
                .get(&WireId(*coordinate))
                .ok_or_else(|| ApiError::input("missing encryption plaintext"))?;
            let encode = |member: Option<IntExpr>, plaintext: Mat| {
                let key = RnsBggPublicKey::new(
                    WireId(*coordinate),
                    setup.input_key(self.parameters, hash_key.clone(), *coordinate, member)?,
                );
                if public {
                    encode_public_inner(self.parameters, &key, mask, plaintext, 0, contract)
                } else {
                    encode_private_inner(
                        self.parameters,
                        &key,
                        payload,
                        mask,
                        plaintext,
                        0,
                        contract,
                    )
                }
            };
            let role = name(dynamic_input_role(*coordinate));
            graph = match (value, public) {
                (InputBinding::Scalar(value), true) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    let payload = encoding
                        .public_payload
                        .ok_or_else(|| ApiError::input("public encoding has no payload"))?;
                    graph.transferred_output(role, (encoding.row.0, encoding.row.1, payload))
                }
                (InputBinding::Scalar(value), false) if setup.compact_input(*coordinate) => {
                    // `s A~ - p (Delta x) r + e` over `pq`: the private
                    // payload row `Delta x r`, expanded under the state gadget.
                    let compact = setup
                        .compact_ring(self.parameters, 0)
                        .map_err(ApiError::scheme)?
                        .ok_or_else(|| ApiError::input("compact input without a compact ring"))?;
                    let p = crate::rns_bgg::dynamic::compact_modulus(setup)
                        .ok_or_else(|| ApiError::input("compact input without a modulus"))?;
                    let lift = |row: &RnsBggRow| {
                        map_row(row.clone(), |entry| entry.centered_rebase(&compact))
                    };
                    let s = lift(
                        &reduce_secret(self.parameters, &mask.row, 0).map_err(ApiError::scheme)?,
                    );
                    let r = lift(
                        &reduce_secret(self.parameters, &payload.row, 0)
                            .map_err(ApiError::scheme)?,
                    );
                    let delta = self
                        .parameters
                        .scalar(0, self.parameters.message_scale(0).map_err(ApiError::scheme)?)
                        .map_err(ApiError::scheme)?;
                    let scale = (delta * value.clone()).centered_rebase(&compact);
                    let key = setup
                        .compact_input_key(self.parameters, hash_key.clone(), *coordinate)
                        .map_err(ApiError::scheme)?;
                    let row = crate::rns_bgg::dynamic::compact_encoding_row(
                        &compact,
                        key,
                        &s,
                        (scale.clone() * r.0, scale * r.1),
                        &p,
                        contract,
                    )
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(role, row)
                }
                (InputBinding::Scalar(value), false) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    graph.transferred_output(role, encoding.row)
                }
                (InputBinding::Family(plaintexts), true) => {
                    let stored = mxx_dsl::parallel(plaintexts.count().clone(), |index| {
                        let plaintext = plaintexts.member(&ring, index.clone());
                        let encoding = encode(Some(index.expression()?), plaintext)
                            .map_err(|_| DslError::Schema)?;
                        let payload = encoding.public_payload.ok_or(DslError::Schema)?;
                        Ok((encoding.row.0, encoding.row.1, payload))
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(role, stored)
                }
                (InputBinding::Family(plaintexts), false) => {
                    let stored = mxx_dsl::parallel(plaintexts.count().clone(), |index| {
                        let plaintext = plaintexts.member(&ring, index.clone());
                        encode(Some(index.expression()?), plaintext)
                            .map(|encoding| encoding.row)
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(role, stored)
                }
            }
            .map_err(ApiError::scheme)?;
        }
        Ok(graph)
    }

    /// The helper `h_k = s H_k - s_k G + e_h` of every helper automorphism `k`
    /// of the setup ([`RnsBggDynamicSetup::helper_automorphisms`]), member `i`
    /// for the `i`-th, with its error under the setup's helper contract.
    /// `s_k G = (sigma_k(s_bar) g, g)`.
    fn automorphism_helpers(
        &self,
        setup: &RnsBggDynamicSetup,
        payload: &RnsBggRow,
        hash_key: &Bytes,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        let automorphisms = &setup.helper_automorphisms();
        let ring = self.parameters.ring(0)?;
        let columns = self.parameters.component_columns(0)?;
        let gadget = self.parameters.public_gadget(0)?;
        let secret = reduce_secret(self.parameters, payload, 0)?;
        if let (Some(compact), Some(p)) = (
            setup.compact_ring(self.parameters, 0)?,
            crate::rns_bgg::dynamic::compact_modulus(setup),
        ) {
            // `s H~_k - p s_k + e` over `pq` for `s_k = (sigma_k(s_bar), 1)`.
            let lifted = map_row(secret.clone(), |entry| entry.centered_rebase(&compact));
            return mxx_dsl::parallel(automorphisms.len(), |member| {
                let k = select(automorphisms, member.expression()?);
                let schema = |_| DslError::Schema;
                let public = setup
                    .compact_helper_key(self.parameters, hash_key.clone(), k.clone())
                    .map_err(schema)?;
                let payload = (
                    secret.0.clone().ring_automorphism(k).centered_rebase(&compact),
                    lifted.1.clone(),
                );
                crate::rns_bgg::dynamic::compact_encoding_row(
                    &compact,
                    public,
                    &lifted,
                    payload,
                    &p,
                    &setup.distributions().helper,
                )
                .map_err(schema)
            })
            .map_err(graph);
        }
        mxx_dsl::parallel(automorphisms.len(), |member| {
            let k = select(automorphisms, member.expression()?);
            let schema = |_| DslError::Schema;
            let public =
                setup.helper_key(self.parameters, hash_key.clone(), k.clone()).map_err(schema)?;
            let error = setup.distributions().helper.sample_row(&ring, columns).map_err(schema)?;
            Ok((
                secret.0.clone() * public.0.0 + secret.1.clone() * public.1.0 -
                    secret.0.clone().ring_automorphism(k) * gadget.clone() +
                    error.0,
                secret.0.clone() * public.0.1 + secret.1.clone() * public.1.1 - gadget.clone() +
                    error.1,
            ))
        })
        .map_err(graph)
    }

    /// The digit companions of `members` packages: member `t`'s gadget
    /// ciphertext is `ciphertext(t)`, its package `package(t)`, and its
    /// payload secret `secret(t)`. Only the digit rows `a >= first_row` have
    /// companions (an automorphism package's leading rows are public, see
    /// [`super::automorphism_public_rows`]); with `r = m - first_row` of them,
    /// element `(t r + a - first_row) m + j` (the order
    /// [`super::companion_columns`] reads) is
    /// `z_{a,j} = s' B_{a,j} - d_{a,j} s' G + eps` for the digit
    /// `d_{a,j} = Dec_G(C)_{a,j}`, with its error under the setup's companion
    /// contract. Each lane decomposes only the entry of column `j` of `C`
    /// that holds digit row `a`.
    fn companion_encodings(
        &self,
        setup: &RnsBggDynamicSetup,
        members: usize,
        ciphertext: &dyn Fn(Int) -> RnsBggMatrix,
        package: &dyn Fn(IntExpr) -> RnsBggPackage,
        secret: &dyn Fn(IntExpr) -> Result<RnsBggRow, RnsBggError>,
        hash_key: &Bytes,
        first_row: usize,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        let parameters = self.parameters;
        let ring = parameters.ring(0)?;
        let current = parameters.level(0)?;
        let base = 1u64
            .checked_shl(current.base_bits())
            .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
        let digits = current.modulus_digits();
        let columns = parameters.component_columns(0)?;
        let width = 2 * columns;
        let rows = width - first_row;
        let count = rows * width;
        let gadget = parameters.public_gadget(0)?;
        let contract = &setup.distributions().companion;
        let compact = setup.compact_ring(parameters, 0)?;
        let p = crate::rns_bgg::dynamic::compact_modulus(setup);
        // Member `lane / count` of the ciphertexts is read through this outer
        // index family, the Gather form a wave importer reads ahead when the
        // ciphertexts are an imported family. One scalar loop computes it,
        // so the graph does not grow with the lane count.
        let member_of = mxx_dsl::parallel(IntExpr::constant(members * count), |lane| {
            Ok(Int::evaluate(quotient(&lane.expression()?, count)))
        })
        .map_err(graph)?;
        mxx_dsl::parallel(IntExpr::constant(members * count), |lane| {
            let schema = |_| DslError::Schema;
            let member_index = member_of.at(lane.clone());
            let index = lane.expression()?;
            let member = quotient(&index, count);
            let a = (remainder(quotient(&index, width), rows) + IntExpr::constant(first_row))
                .canonicalize();
            let j = remainder(index, width);
            let ciphertext = ciphertext(member_index);
            let [top, bottom] = [&ciphertext.0, &ciphertext.1].map(super::join_row);
            // Digit row `a` is digit `a mod D` of entry `a / D` of the stacked
            // column `j`.
            let entry_digits = super::row_entry(
                &Mat::concat(
                    mxx_ir_core::node::ConcatAxis::Columns,
                    vec![super::row_entry(&top, j.clone()), super::row_entry(&bottom, j.clone())],
                ),
                quotient(&a, digits),
            )
            .decompose(base, digits);
            let digit = super::unit_row(&ring, digits, remainder(a.clone(), digits))
                .mul_small_rhs(entry_digits);
            if let (Some(compact), Some(p)) = (&compact, &p) {
                // `s' (A~ - p d I_2) + eps` over `pq`: the payload row `d s'`.
                let tag = super::companion_tag(&setup.package_tag(&package(member.clone())), a, j);
                let key =
                    setup.compact_matrix(parameters, 0, hash_key.clone(), tag).map_err(schema)?;
                let lifted = map_row(secret(member).map_err(schema)?, |entry| {
                    entry.centered_rebase(compact)
                });
                let digit = digit.centered_rebase(compact);
                let payload = (digit.clone() * lifted.0.clone(), digit * lifted.1.clone());
                return crate::rns_bgg::dynamic::compact_encoding_row(
                    compact, key, &lifted, payload, p, contract,
                )
                .map_err(schema);
            }
            let key = super::companion_key(
                parameters,
                setup,
                0,
                hash_key.clone(),
                &setup.package_tag(&package(member.clone())),
                a,
                j,
            )
            .map_err(schema)?;
            public_encoding_row(
                parameters,
                key,
                &secret(member).map_err(schema)?,
                digit.tensor(gadget.clone()),
                digits,
                0,
                contract,
            )
            .map_err(schema)
        })
        .map_err(graph)
    }
}

/// A stored companion or helper row at the setup ring as its readers use it:
/// expanded from its two-column form over `pq` if the setup's encodings are
/// compact ([`crate::rns_bgg::dynamic::expand_compact_row`]), else as stored.
fn stored_row(
    parameters: &RnsBggParameters,
    setup: &RnsBggDynamicSetup,
    row: RnsBggRow,
) -> Result<RnsBggRow, RnsBggError> {
    match setup.compact_extension() {
        Some(_) => crate::rns_bgg::dynamic::expand_compact_row(parameters, setup, 0, row),
        None => Ok(row),
    }
}

/// The gadget ciphertexts of one RHS coordinate, read by member.
enum PackageCiphertexts {
    Scalar(RnsBggMatrix),
    Family(Family<RnsBggMatrix>),
}

impl PackageCiphertexts {
    fn member(&self, member: Int) -> RnsBggMatrix {
        match self {
            Self::Scalar(ciphertext) => ciphertext.clone(),
            Self::Family(family) => family.at(member),
        }
    }
}

/// A `2 x 2` block matrix imported from four ports of `production`.
pub(super) fn prelude_matrix(
    ring: &Ring,
    production: &ProductionId,
    ports: &[String],
    columns: usize,
) -> RnsBggMatrix {
    let port = |index: usize| {
        ring.artifact_input(
            production.clone(),
            ports[index].clone(),
            (1, columns),
            ArtifactAvailability::Transferred,
        )
    };
    ((port(0), port(1)), (port(2), port(3)))
}

/// A family of `2 x 2` block matrices imported from four family ports of
/// `production`, read member by member.
pub(super) fn prelude_matrix_family(
    ring: &Ring,
    production: &ProductionId,
    ports: &[String],
    count: usize,
    columns: usize,
) -> Result<Family<RnsBggMatrix>, RnsBggError> {
    let parts = ports
        .iter()
        .map(|port| {
            ring.family_artifact_input(
                production.clone(),
                port.clone(),
                IntExpr::constant(count),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        })
        .collect::<Vec<_>>();
    let first = |part: &Family<Mat>| part.at(Int::constant(0));
    let element: RnsBggMatrix =
        ((first(&parts[0]), first(&parts[1])), (first(&parts[2]), first(&parts[3])));
    let fields = parts.iter().map(|part| part.value_handle().clone()).collect::<Vec<_>>();
    family_of_fields(&element, IntExpr::constant(count), &fields).map_err(graph)
}

impl EncodingLowering<'_> {
    pub(super) fn dynamic_state(&self) -> Result<&DynamicEvaluation, RnsBggError> {
        self.dynamic.as_ref().ok_or(RnsBggError::Binding("the Dynamic setup is not bound"))
    }

    /// A stored encoding row as evaluation reads it (see [`stored_row`]).
    fn read_row(&self, row: RnsBggRow) -> Result<RnsBggRow, RnsBggError> {
        stored_row(self.parameters, &self.dynamic_state()?.setup, row)
    }

    /// A family of `count` rows of `columns` columns per component imported
    /// from the encryption under `role`.
    /// A compact setup stores its companions and helpers in two columns over
    /// `pq` (`columns` is then ignored).
    fn encryption_row_family(
        &self,
        role: String,
        count: usize,
        columns: usize,
    ) -> Result<Family<RnsBggRow>, RnsBggError> {
        let setup = &self.dynamic_state()?.setup;
        let (ring, columns) = match setup.compact_ring(self.parameters, 0)? {
            Some(compact) => (compact, 1),
            None => (self.parameters.ring(0)?, columns),
        };
        let families = [0, 1].map(|port| {
            ring.family_artifact_input(
                self.encryption.clone(),
                self.encryption_artifact_name(format!("{role}.{port}")),
                IntExpr::constant(count),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        });
        let element: RnsBggRow =
            (families[0].at(Int::constant(0)), families[1].at(Int::constant(0)));
        let fields =
            families.iter().map(|family| family.value_handle().clone()).collect::<Vec<_>>();
        family_of_fields(&element, IntExpr::constant(count), &fields).map_err(graph)
    }

    /// Import, where the Program is bound, the digit companions of every RHS
    /// package the Program reads and the automorphism packages of the setup,
    /// so a Program family body reads their members in place.
    pub(super) fn import_dynamic_packages(&mut self) -> Result<(), RnsBggError> {
        let state = self.dynamic_state()?.clone();
        let columns = self.parameters.component_columns(0)?;
        let count = companion_count(self.parameters, 0, columns)?;
        let digits = self.parameters.level(0)?.modulus_digits();
        let program = self.program.clone();
        for (rhs, shape) in program.rhs_inputs() {
            let coordinate = state.binding.rhs(*rhs)?;
            let members = resolved_input_cardinality(shape, &ParamEnv::default())?.unwrap_or(1);
            let family = self.encryption_row_family(
                dynamic_companion_role(coordinate),
                members * count,
                digits,
            )?;
            self.rhs_companions.insert(*rhs, family);
        }
        let switching = state.setup.switching_automorphisms().len();
        let helpers = state.setup.helper_automorphisms().len();
        let imported = (switching > 0)
            .then(|| {
                Ok::<_, RnsBggError>((
                    self.encryption_row_family(
                        AUTOMORPHISM_HELPER_ROLE.to_owned(),
                        helpers,
                        columns,
                    )?,
                    self.encryption_row_family(
                        AUTOMORPHISM_COMPANION_ROLE.to_owned(),
                        switching * automorphism_companion_count(self.parameters, 0, columns)?,
                        digits,
                    )?,
                ))
            })
            .transpose()?;
        if let Some(dynamic) = self.dynamic.as_mut() {
            dynamic.automorphisms = imported;
        }
        Ok(())
    }

    /// The Dynamic product of `input` by the RHS package of `reference`,
    /// whose gadget ciphertext is `ciphertext`: `c' = E + c Dec(C)` with the
    /// companion row `E_j = sum_a z_{a,j} U_{:,a}` and the key
    /// `T_j = sum_a B_{a,j} U_{:,a}` of `U = Dec(A)` (see
    /// [`super::companion_columns`]). The package is the same at every
    /// consuming level.
    pub(super) fn dynamic_mul(
        &self,
        input: &RnsBggEncoding,
        ciphertext: &RnsBggMatrix,
        reference: &RhsRef<Int>,
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let state = self.dynamic_state()?;
        let level = self.workload.level(input.wire())?;
        self.parameters.dynamic_route(level)?;
        self.parameters.dynamic_route(self.workload.level(self.site()?.output)?)?;
        let coordinate = state.binding.rhs(reference.id)?;
        // A member selected at run time (`Operand::Selected`) has no constant tag; its key is
        // the one preprocessing published for the root site.
        let selected = reference.index.as_ref().filter(|index| index.expression().is_err());
        let member = match selected {
            Some(_) => None,
            None => reference
                .index
                .as_ref()
                .map(Int::expression)
                .transpose()
                .map_err(|_| RnsBggError::Binding("RHS member index"))?,
        };
        let companions = self
            .rhs_companions
            .get(&reference.id)
            .ok_or(RnsBggError::Binding("Dynamic RHS companions are not imported"))?;
        let columns = self.parameters.component_columns(level)?;
        let count = companion_count(self.parameters, level, columns)?;
        let tag =
            state.setup.package_tag(&RnsBggPackage::Rhs { coordinate, member: member.clone() });
        let key_rows =
            [&input.public_key.matrix.0, &input.public_key.matrix.1].map(super::join_row);
        let offset = member.unwrap_or(IntExpr::constant(0)) * IntExpr::constant(count);
        let imported = self.imported_public_action(level, 0)?;
        if selected.is_some() && imported.is_none() {
            return Err(RnsBggError::Binding(
                "an RHS member selected at run time is read only at root scope",
            ));
        }
        let companion_index = |index: IntExpr| match selected {
            Some(member) => member.clone().mul(Int::constant(count)).add(Int::evaluate(index)),
            None => Int::evaluate((offset.clone() + index).canonicalize()),
        };
        let companion_key = |_, a, j| {
            super::companion_key(
                self.parameters,
                &state.setup,
                level,
                self.hash_key.clone(),
                &tag,
                a,
                j,
            )
        };
        let rows = super::companion_columns(
            self.parameters,
            level,
            1,
            columns,
            &|_, r, a| Ok(super::row_entry(&key_rows[r], a)),
            imported.is_none().then_some(&companion_key as _),
            Some(&|_, a: IntExpr, j: IntExpr| {
                let index = a * IntExpr::constant(2 * columns) + j;
                self.read_row(companions.at(companion_index(index)))
            }),
            None,
        )?;
        let product = super::column_action(
            self.parameters,
            level,
            level,
            ciphertext,
            columns,
            vec![super::join_row(&input.row)],
        )?;
        let split = |row: &Mat| super::split_row(row, columns);
        let companion = rows.last().ok_or(RnsBggError::Binding("companion row"))?;
        let key = match imported {
            Some((key, _)) => key,
            None => (split(&rows[0]), split(&rows[1])),
        };
        Ok((split(&(product[0].clone() + companion.clone())), key))
    }

    /// The public action preprocessing published for the current site if
    /// it is at root scope (see [`super::public_action_role`]): its output
    /// key at `level` and, for `switched > 0` branches, the rows of their
    /// product keys `T_k`. A site in a family body returns `None`, and its
    /// evaluation recomputes the same action from the setup matrices.
    fn imported_public_action(
        &self,
        level: usize,
        switched: usize,
    ) -> Result<Option<(RnsBggMatrix, Option<[Mat; 2]>)>, RnsBggError> {
        let output = self.site()?.output;
        if !self.program.root_private_sites().contains(&output) {
            return Ok(None);
        }
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let import = |part: &str, port: usize, width: usize| {
            ring.artifact_input(
                self.preprocess.clone(),
                self.preprocess_artifact_name(format!(
                    "{}.{port}",
                    super::public_action_role(output, part)
                )),
                (1, width),
                ArtifactAvailability::Transferred,
            )
        };
        let key = [0, 1, 2, 3].map(|port| import("key", port, columns));
        let [k00, k01, k10, k11] = key;
        let branches = (switched > 0)
            .then(|| [0, 1].map(|port| import("branches", port, switched * 2 * columns)));
        Ok(Some(((k00, k01), (k10, k11))).map(|key| (key, branches)))
    }

    /// `Aut_k` of an encoding row (see [`crate::rns_bgg::dynamic`]): `r_k =
    /// sigma_k(c) Dec(C_k) + E_k` with the companion block `companion()` of
    /// `k`'s package, then the mask switch `h_k Dec(T_k) + r_k` against the
    /// product key `pre_key()`. Returns the switched row and, if `recompute`,
    /// the two rows of the switched key `H_k Dec(T_k)` (each `1 x m`). The
    /// closures run where their values are first needed.
    #[allow(clippy::too_many_arguments)]
    fn automorphism_branch(
        &self,
        level: usize,
        row: &RnsBggRow,
        switch: &AutomorphismSwitch,
        companion: &dyn Fn() -> RnsBggRow,
        pre_key: &dyn Fn() -> RnsBggMatrix,
        recompute: bool,
    ) -> Result<(RnsBggRow, Vec<Mat>), RnsBggError> {
        let state = self.dynamic_state()?;
        let setup = &state.setup;
        let (helpers, _) = state
            .automorphisms
            .as_ref()
            .ok_or(RnsBggError::Binding("the automorphism packages are not imported"))?;
        let columns = self.parameters.component_columns(level)?;
        let split = |row: &Mat| super::split_row(row, columns);
        let k = switch.k.clone();
        // `r_k = sigma_k(c) Dec(C_k) + E_k`, under the key `T_k`.
        let routed = map_row(row.clone(), |entry| entry.ring_automorphism(k.clone()));
        let switching = automorphism_ciphertext(
            self.parameters,
            setup,
            self.hash_key.clone(),
            k.clone(),
            switch.inverse.clone(),
            self.read_row(helpers.at(Int::evaluate(switch.inverse_helper.clone())))?,
        )?;
        let product = super::column_action(
            self.parameters,
            level,
            level,
            &switching,
            columns,
            vec![super::join_row(&routed)],
        )?;
        let switched_row = add_encoding_rows(split(&product[0]), companion());
        let pre_key = pre_key();
        // `Aut_k(c) = h_k Dec(T_k) + r_k` under `A_k = H_k Dec(T_k)`.
        let helper = self.read_row(helpers.at(Int::evaluate(switch.helper.clone())))?;
        let mut acted = vec![super::join_row(&helper)];
        if recompute {
            let public = setup.helper_key(self.parameters, self.hash_key.clone(), k)?;
            acted.extend([super::join_row(&public.0), super::join_row(&public.1)]);
        }
        let mut actions =
            super::column_action(self.parameters, level, level, &pre_key, columns, acted)?;
        let automorphed_row = add_encoding_rows(split(&actions[0]), switched_row);
        Ok((automorphed_row, actions.split_off(1)))
    }

    /// The Dynamic LUT `sum_{k in H_L} SMul_{D_{k,L}}(Aut_k(c))` of `input`
    /// (see [`crate::rns_bgg::dynamic`]), its row and key. One companion loop
    /// nest first builds every nonidentity branch's key rows `T_k` and
    /// companion row `E_k` (block `s` of each row), so the stored companions
    /// are read in one loop nest; the branch loop then adds
    /// `sigma_k(c) Dec(C_k) + E_k`, switches the mask with the helper, and
    /// applies the public scalar action, one column at a time.
    pub(super) fn dynamic_lut(
        &self,
        input: &RnsBggEncoding,
        table: &LutTable,
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let state = self.dynamic_state()?;
        let setup = &state.setup;
        let level = self.workload.level(input.wire())?;
        let normalized = level + usize::from(table.output_representation().is_coefficient());
        self.parameters.dynamic_route(normalized.max(self.workload.level(self.site()?.output)?))?;
        let automorphisms =
            lut_automorphisms(self.parameters, setup.automorphisms(), level, table)?;
        let values = crate::rns_bgg::dynamic::lut_entries(table, &self.dynamic_state()?.data)?;
        let layout = table.input_layout();
        let representation = table.output_representation();
        let columns = self.parameters.component_columns(level)?;
        let coefficient = |k: IntExpr| {
            EncodingLowering::lut_coefficient(
                self.parameters,
                level,
                layout,
                representation,
                k,
                values.clone(),
            )
        };
        // The identity branch: `Aut_1 = id`.
        let identity = scalar_action(self.parameters, level, coefficient(IntExpr::constant(1))?)?;
        let initial = (
            scalar_action_row(input.row.clone(), &identity),
            scalar_action_matrix(input.public_key.matrix.clone(), &identity),
        );
        let switched = automorphisms[1..].to_vec();
        if switched.is_empty() {
            return Ok(initial);
        }
        let (_, companions) = state
            .automorphisms
            .as_ref()
            .ok_or(RnsBggError::Binding("the automorphism packages are not imported"))?;
        let plan = crate::rns_bgg::dynamic::radix_plan(setup, &switched)?;
        if plan.len() > 1 {
            return self.radix_dynamic_lut(input, table, &plan, initial);
        }
        // Every branch is switched directly by its own package.
        let members =
            switched.iter().map(|k| setup.switching_member(*k)).collect::<Result<Vec<_>, _>>()?;
        let helper_members =
            switched.iter().map(|k| setup.helper_member(*k)).collect::<Result<Vec<_>, _>>()?;
        let inverses = switched
            .iter()
            .map(|k| setup.automorphism_inverse(*k))
            .collect::<Result<Vec<_>, _>>()?;
        let inverse_members =
            inverses.iter().map(|k| setup.helper_member(*k)).collect::<Result<Vec<_>, _>>()?;
        let count = automorphism_companion_count(self.parameters, level, columns)?;
        let public_rows = super::automorphism_public_rows(self.parameters, level)?;
        let width = 2 * columns;
        let key_rows =
            [&input.public_key.matrix.0, &input.public_key.matrix.1].map(super::join_row);
        let imported = self.imported_public_action(level, switched.len())?;
        let public_top = |branch: IntExpr| {
            crate::rns_bgg::dynamic::automorphism_public_top(
                self.parameters,
                setup,
                self.hash_key.clone(),
                select(&switched, branch.clone()),
                select(&inverses, branch),
            )
        };
        let companion_key = |branch, a, j| {
            let tag = setup.package_tag(&RnsBggPackage::Automorphism(select(&switched, branch)));
            super::companion_key(self.parameters, setup, level, self.hash_key.clone(), &tag, a, j)
        };
        let rows = super::companion_columns(
            self.parameters,
            level,
            switched.len(),
            columns,
            &|branch, r, a| {
                Ok(super::row_entry(&key_rows[r], a).ring_automorphism(select(&switched, branch)))
            },
            imported.is_none().then_some(&companion_key as _),
            Some(&|branch, a: IntExpr, j: IntExpr| {
                let index = select(&members, branch) * IntExpr::constant(count) +
                    (a - IntExpr::constant(public_rows)) * IntExpr::constant(width) +
                    j;
                self.read_row(companions.at(Int::evaluate(index.canonicalize())))
            }),
            Some((public_rows, &public_top)),
        )?;
        let split = |row: &Mat| super::split_row(row, columns);
        let schema = |_| DslError::Schema;
        let companion = rows.last().ok_or(RnsBggError::Binding("companion row"))?;
        // With the imported public action, the key is preprocessing's and
        // only the row is accumulated.
        let (cached, key_rows) = match &imported {
            Some((key, Some(branches))) => (Some(key.clone()), branches.clone()),
            Some((_, None)) => return Err(RnsBggError::Binding("LUT branch keys are missing")),
            None => (None, [rows[0].clone(), rows[1].clone()]),
        };
        let recompute = cached.is_none();
        // Branch `index`: its row term and, unless the key is imported,
        // its key term.
        let branch_terms = |index: Int| -> Result<(RnsBggRow, Option<RnsBggMatrix>), DslError> {
            let branch = index.expression()?;
            let k = select(&switched, branch.clone());
            let start = (branch.clone() * IntExpr::constant(width)).canonicalize();
            let block = |row: &Mat| {
                split(&row.clone().slice(
                    None,
                    Some(mxx_ir_core::node::IndexRange {
                        start: start.clone(),
                        end: (start.clone() + IntExpr::constant(width)).canonicalize(),
                    }),
                ))
            };
            let (automorphed_row, key_rows) = self
                .automorphism_branch(
                    level,
                    &input.row,
                    &AutomorphismSwitch {
                        k: k.clone(),
                        inverse: select(&inverses, branch.clone()),
                        inverse_helper: select(&inverse_members, branch.clone()),
                        helper: select(&helper_members, branch),
                    },
                    &|| block(companion),
                    &|| (block(&key_rows[0]), block(&key_rows[1])),
                    recompute,
                )
                .map_err(schema)?;
            let action = scalar_action(self.parameters, level, coefficient(k).map_err(schema)?)
                .map_err(schema)?;
            let key = recompute
                .then(|| scalar_action_matrix((split(&key_rows[0]), split(&key_rows[1])), &action));
            Ok((scalar_action_row(automorphed_row, &action), key))
        };
        let count = IntExpr::constant(switched.len());
        match cached {
            Some(key) => {
                let row = mxx_dsl::iterate(count, initial.0, |index, sum| {
                    Ok(add_encoding_rows(sum, branch_terms(index)?.0))
                })
                .map_err(graph)?;
                Ok((row, key))
            }
            None => mxx_dsl::iterate(count, initial, |index, (row_sum, key_sum)| {
                let (row, key) = branch_terms(index)?;
                let key = key.ok_or(DslError::Schema)?;
                Ok((add_encoding_rows(row_sum, row), add_public_matrices(key_sum, key)))
            })
            .map_err(graph),
        }
    }
}

impl EncodingLowering<'_> {
    /// The Dynamic LUT of `input` by the radix plan `plan` of more than one
    /// depth (see [`crate::rns_bgg::dynamic`]), continuing from the identity
    /// branch `initial`. Per depth, one companion loop nest builds the
    /// product keys of its switches from the input key (first depth) or the
    /// parents' keys `H_p Dec(T_p)`; the parents of the next depth are
    /// switched one by one and kept with their keys, and the branch loops add
    /// the scalar actions of the parents and then switch and add the others.
    fn radix_dynamic_lut(
        &self,
        input: &RnsBggEncoding,
        table: &LutTable,
        plan: &[crate::rns_bgg::dynamic::RadixDepth],
        initial: (RnsBggRow, RnsBggMatrix),
    ) -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
        let state = self.dynamic_state()?;
        let setup = &state.setup;
        let (_, companions) = state
            .automorphisms
            .as_ref()
            .ok_or(RnsBggError::Binding("the automorphism packages are not imported"))?;
        let level = self.workload.level(input.wire())?;
        let values = crate::rns_bgg::dynamic::lut_entries(table, &self.dynamic_state()?.data)?;
        let layout = table.input_layout();
        let representation = table.output_representation();
        let columns = self.parameters.component_columns(level)?;
        let count = automorphism_companion_count(self.parameters, level, columns)?;
        let public_rows = super::automorphism_public_rows(self.parameters, level)?;
        let width = 2 * columns;
        let coefficient = |k: IntExpr| {
            EncodingLowering::lut_coefficient(
                self.parameters,
                level,
                layout,
                representation,
                k,
                values.clone(),
            )
        };
        let members = |ks: &[usize], member: &dyn Fn(usize) -> Result<usize, RnsBggError>| {
            ks.iter().map(|k| member(*k)).collect::<Result<Vec<_>, _>>()
        };
        let switch = |ks: &[usize], index: IntExpr| -> Result<AutomorphismSwitch, RnsBggError> {
            let inverses = members(ks, &|k| setup.automorphism_inverse(k))?;
            Ok(AutomorphismSwitch {
                k: select(ks, index.clone()),
                inverse: select(&inverses, index.clone()),
                inverse_helper: select(
                    &members(&inverses, &|k| setup.helper_member(k))?,
                    index.clone(),
                ),
                helper: select(&members(ks, &|k| setup.helper_member(k))?, index),
            })
        };
        let total = plan.iter().map(|depth| depth.branches.len()).sum();
        let imported = self.imported_public_action(level, total)?;
        let recompute = imported.is_none();
        let split = |row: &Mat| super::split_row(row, columns);
        let block = |row: &Mat, index: IntExpr| {
            let start = (index * IntExpr::constant(width)).canonicalize();
            split(&row.clone().slice(
                None,
                Some(mxx_ir_core::node::IndexRange {
                    start: start.clone(),
                    end: (start + IntExpr::constant(width)).canonicalize(),
                }),
            ))
        };
        let companion_key = |ks: &[usize], branch: IntExpr, a, j| {
            let tag = setup.package_tag(&RnsBggPackage::Automorphism(select(ks, branch)));
            super::companion_key(self.parameters, setup, level, self.hash_key.clone(), &tag, a, j)
        };
        let companion = |ks_members: &[usize], branch: IntExpr, a: IntExpr, j: IntExpr| {
            let index = select(ks_members, branch) * IntExpr::constant(count) +
                (a - IntExpr::constant(public_rows)) * IntExpr::constant(width) +
                j;
            self.read_row(companions.at(Int::evaluate(index.canonicalize())))
        };
        let schema = |_| DslError::Schema;
        let input_rows =
            [&input.public_key.matrix.0, &input.public_key.matrix.1].map(super::join_row);
        let mut sum = initial;
        // The switched parents of the current depth: their rows and keys.
        let mut parents: Option<(Family<RnsBggRow>, Family<RnsBggMatrix>)> = None;
        let mut offset = 0;
        for depth in plan {
            let switches = &depth.switches;
            let switch_members = members(switches, &|k| setup.switching_member(k))?;
            let switch_inverses = members(switches, &|k| setup.automorphism_inverse(k))?;
            let public_top = |branch: IntExpr| {
                crate::rns_bgg::dynamic::automorphism_public_top(
                    self.parameters,
                    setup,
                    self.hash_key.clone(),
                    select(switches, branch.clone()),
                    select(&switch_inverses, branch),
                )
            };
            let parent_row = |branch: IntExpr| match &parents {
                None => input.row.clone(),
                Some((rows, _)) => rows.at(Int::evaluate(select(&depth.parents, branch))),
            };
            let rows = super::companion_columns(
                self.parameters,
                level,
                depth.branches.len(),
                columns,
                &|branch, r, a| {
                    let entry = match &parents {
                        None => super::row_entry(&input_rows[r], a),
                        Some((_, keys)) => {
                            let key =
                                keys.at(Int::evaluate(select(&depth.parents, branch.clone())));
                            super::row_entry(
                                &super::join_row(if r == 0 { &key.0 } else { &key.1 }),
                                a,
                            )
                        }
                    };
                    Ok(entry.ring_automorphism(select(switches, branch)))
                },
                recompute.then_some(&|branch, a, j| companion_key(switches, branch, a, j) as _),
                Some(&|branch, a, j| companion(&switch_members, branch, a, j)),
                Some((public_rows, &public_top)),
            )?;
            let companion_row = rows.last().ok_or(RnsBggError::Binding("companion row"))?;
            let product_key = |branch: IntExpr| match &imported {
                Some((_, Some(branches))) => {
                    let position = (IntExpr::constant(offset) + branch).canonicalize();
                    Ok((block(&branches[0], position.clone()), block(&branches[1], position)))
                }
                Some((_, None)) => Err(RnsBggError::Binding("LUT branch keys are missing")),
                None => Ok((block(&rows[0], branch.clone()), block(&rows[1], branch))),
            };
            // The parents of the next depth, switched one by one with their keys.
            let mut switched_rows = Vec::with_capacity(depth.internal);
            let mut switched_keys = Vec::with_capacity(depth.internal);
            for position in 0..depth.internal {
                let index = IntExpr::constant(position);
                let pre_key = product_key(index.clone())?;
                let (row, key) = self.automorphism_branch(
                    level,
                    &parent_row(index.clone()),
                    &switch(switches, index.clone())?,
                    &|| block(companion_row, index.clone()),
                    &|| pre_key.clone(),
                    true,
                )?;
                switched_rows.push(row);
                switched_keys.push((split(&key[0]), split(&key[1])));
            }
            let switched = (depth.internal > 0)
                .then(|| {
                    Ok::<_, RnsBggError>((
                        Family::pack(switched_rows).map_err(graph)?,
                        Family::pack(switched_keys).map_err(graph)?,
                    ))
                })
                .transpose()?;
            let ks = &depth.branches;
            // Each branch's row term and, unless the key is imported, its key term.
            let internal_terms =
                |index: Int| -> Result<(RnsBggRow, Option<RnsBggMatrix>), DslError> {
                    let (rows, keys) = switched.as_ref().ok_or(DslError::Schema)?;
                    let branch = index.expression()?;
                    let action = scalar_action(
                        self.parameters,
                        level,
                        coefficient(select(ks, branch)).map_err(schema)?,
                    )
                    .map_err(schema)?;
                    let key =
                        recompute.then(|| scalar_action_matrix(keys.at(index.clone()), &action));
                    Ok((scalar_action_row(rows.at(index), &action), key))
                };
            let leaf_terms = |index: Int| -> Result<(RnsBggRow, Option<RnsBggMatrix>), DslError> {
                let branch =
                    (IntExpr::constant(depth.internal) + index.expression()?).canonicalize();
                let (row, key) = self
                    .automorphism_branch(
                        level,
                        &parent_row(branch.clone()),
                        &switch(switches, branch.clone()).map_err(schema)?,
                        &|| block(companion_row, branch.clone()),
                        &|| product_key(branch.clone()).expect("product key"),
                        recompute,
                    )
                    .map_err(schema)?;
                let action = scalar_action(
                    self.parameters,
                    level,
                    coefficient(select(ks, branch)).map_err(schema)?,
                )
                .map_err(schema)?;
                let key = recompute
                    .then(|| scalar_action_matrix((split(&key[0]), split(&key[1])), &action));
                Ok((scalar_action_row(row, &action), key))
            };
            if !recompute {
                product_key(IntExpr::constant(0))?;
            }
            let add =
                |terms: &dyn Fn(Int) -> Result<(RnsBggRow, Option<RnsBggMatrix>), DslError>,
                 count: usize,
                 state: (RnsBggRow, RnsBggMatrix)|
                 -> Result<(RnsBggRow, RnsBggMatrix), RnsBggError> {
                    if count == 0 {
                        return Ok(state);
                    }
                    mxx_dsl::iterate(
                        IntExpr::constant(count),
                        state,
                        |index, (row_sum, key_sum)| {
                            let (row, key) = terms(index)?;
                            let key_sum = match key {
                                Some(key) => add_public_matrices(key_sum, key),
                                None => key_sum,
                            };
                            Ok((add_encoding_rows(row_sum, row), key_sum))
                        },
                    )
                    .map_err(graph)
                };
            sum = add(&internal_terms, depth.internal, sum)?;
            sum = add(&leaf_terms, depth.branches.len() - depth.internal, sum)?;
            offset += depth.branches.len();
            parents = switched;
        }
        match imported {
            Some((cached, _)) => Ok((sum.0, cached)),
            None => Ok(sum),
        }
    }
}

/// A row of two uniform blocks of `columns` columns over `ring`: the top row
/// of a ciphertext sampled in the encryption that publishes it.
pub(super) fn uniform_row(ring: &Ring, columns: usize) -> RnsBggRow {
    (ring.uniform_residue((1, columns)), ring.uniform_residue((1, columns)))
}

/// The coordinates of one automorphism switch `Aut_k`: `k`, `k^-1`, and the
/// helper members of `k^-1` (for `C_k`) and of `k` (for the mask switch).
struct AutomorphismSwitch {
    k: IntExpr,
    inverse: IntExpr,
    inverse_helper: IntExpr,
    helper: IntExpr,
}

/// Shared helpers of the Dynamic stage tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use mxx_backends::{
        MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        poly::dcrt::poly::DCRTPoly,
    };

    fn read(name: &str, default: &str) -> String {
        std::env::var(format!("MXX_EXPONENT_LUT_DYNAMIC_TEST_{name}"))
            .unwrap_or_else(|_| default.to_owned())
    }

    /// An equal-basis Dynamic chain of `levels` levels: ring dimension,
    /// base bits, CRT depth and prime bits, and sigma overridable by
    /// `MXX_EXPONENT_LUT_DYNAMIC_TEST_{RING_DIMENSION,BASE_BITS,CRT_DEPTH,CRT_BITS,SIGMA}`.
    pub(crate) fn parameters(levels: usize) -> RnsBggParameters {
        let ring_dimension = read("RING_DIMENSION", "16").parse().unwrap();
        let base_bits = read("BASE_BITS", "6").parse().unwrap();
        let depth = read("CRT_DEPTH", "4").parse().unwrap();
        let bits = read("CRT_BITS", "17").parse().unwrap();
        let sigma = read("SIGMA", "1").parse::<bigdecimal::BigDecimal>().unwrap();
        let level = DCRTPolyParams::new(ring_dimension, depth, bits, base_bits, None, None);
        RnsBggParameters::new(vec![level; levels], sigma, levels)
            .unwrap()
            .with_rhs_mode(RhsMode::Dynamic)
    }

    /// A CPU execution registering the one ring of `parameters`.
    pub(crate) fn execution(parameters: &RnsBggParameters) -> CpuExecution {
        CpuExecution::new(cpu_backend(parameters.distinct_levels().into_iter().cloned()))
    }

    /// As many CRT primes as the ring of `parameters` has, of the same bit size, `1 mod 2n`,
    /// and disjoint from it: the auxiliary modulus of compact encodings.
    pub(crate) fn compact_extension(parameters: &RnsBggParameters) -> Vec<u64> {
        let level = parameters.level(0).unwrap();
        let base = level.to_crt().0;
        DCRTPolyParams::new(
            level.ring_dimension(),
            2 * level.crt_depth(),
            level.crt_bits(),
            level.base_bits(),
            None,
            None,
        )
        .to_crt()
        .0
        .into_iter()
        .filter(|prime| !base.contains(prime))
        .take(level.crt_depth())
        .collect()
    }

    /// A CPU execution registering the ring of `parameters` and its compact ring over
    /// `extension`.
    pub(crate) fn compact_execution(
        parameters: &RnsBggParameters,
        extension: &[u64],
    ) -> CpuExecution {
        let level = parameters.level(0).unwrap().clone();
        let moduli =
            level.to_crt().0.into_iter().chain(extension.iter().copied()).collect::<Vec<_>>();
        let compact = DCRTPolyParams::new(
            level.ring_dimension(),
            moduli.len(),
            level.crt_bits(),
            level.base_bits(),
            Some(moduli),
            None,
        );
        CpuExecution::new(cpu_backend([level, compact]))
    }

    /// The monomial `X^exponent` (negated past `n`) in the ring of `parameters`.
    pub(crate) fn monomial(parameters: &RnsBggParameters, exponent: usize) -> DCRTPoly {
        DCRTPoly::const_rotate_poly(&parameters.levels()[0], exponent)
    }

    /// One setup with its public seed.
    pub(crate) struct Setup {
        pub(crate) setup: RnsBggDynamicSetup,
        pub(crate) context: [u8; 32],
        pub(crate) hash_key: [u8; 32],
    }

    impl Setup {
        /// A setup over explicit coordinates.
        pub(crate) fn new(
            parameters: &RnsBggParameters,
            inputs: BTreeMap<u32, crate::program::ProgramInputDeclaration>,
            rhs: BTreeMap<u32, RnsBggDynamicRhs>,
            layout: ExpLayout,
        ) -> Self {
            let context: [u8; 32] = rand::random();
            let hash_key: [u8; 32] = rand::random();
            let setup = RnsBggDynamicSetup::new(
                parameters,
                context,
                &hash_key,
                inputs,
                rhs,
                layout,
                RnsBggDynamicDistributions::uniform(parameters.encoding_contract()),
            )
            .unwrap();
            Self { setup, context, hash_key }
        }

        /// The setup of `program` with the identity mapping.
        pub(crate) fn for_program(
            parameters: &RnsBggParameters,
            program: &ExponentLutProgram,
            layout: ExpLayout,
        ) -> (Self, RnsBggDynamicBinding) {
            let context: [u8; 32] = rand::random();
            let hash_key: [u8; 32] = rand::random();
            let (setup, binding) = RnsBggDynamicSetup::for_program(
                parameters,
                context,
                &hash_key,
                program,
                layout,
                RnsBggDynamicDistributions::uniform(parameters.encoding_contract()),
            )
            .unwrap();
            (Self { setup, context, hash_key }, binding)
        }

        /// Compile `program` against the setup.
        pub(crate) fn preprocess(
            &self,
            parameters: &RnsBggParameters,
            program: &ExponentLutProgram,
            binding: &RnsBggDynamicBinding,
            output_use: RnsOutputUse,
            execution: &mut CpuExecution,
            store: &mut MemoryArtifactStore,
        ) -> Result<
            (
                StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
                StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
            ),
            ApiError,
        > {
            crate::api::preprocess(
                program,
                &ParamEnv::default(),
                &PlaintextInputs::<RhsId, DCRTPoly>::new(),
                None,
                &self.hash_key,
                crate::rns_bgg::RnsPreprocessBuild::new(parameters, self.context)
                    .with_output_use(output_use)
                    .with_dynamic_setup(&self.setup, binding.clone())
                    .map_err(ApiError::scheme)?,
                execution,
                store,
                |_, outputs| Ok(outputs),
                |_| Ok(()),
            )
        }

        /// Encrypt under a fresh secret and session, with no Program.
        pub(crate) fn encrypt(
            &self,
            parameters: &RnsBggParameters,
            lhs: &PlaintextInputs<WireId, DCRTPoly>,
            rhs: &PlaintextInputs<u32, DCRTPoly>,
            execution: &mut CpuExecution,
            store: &mut MemoryArtifactStore,
        ) -> (StageArtifacts<RnsBggEncryptionDescriptor>, RnsBggPayloadSecret) {
            let secret = RnsEncryptBuild::new(parameters, self.context)
                .sample_payload_secret(execution, store, |_| Ok(()))
                .unwrap();
            let encrypted = crate::api::encrypt(
                &self.setup.encryption_inputs(),
                &ParamEnv::default(),
                lhs,
                None,
                &secret,
                None,
                &self.hash_key,
                RnsEncryptBuild::new(parameters, self.context)
                    .with_dynamic_setup(&self.setup)
                    .unwrap(),
                execution,
                store,
                |build| build.bind_rhs_inputs(&ParamEnv::default(), rhs, rand::random()),
                |_| Ok(()),
            )
            .unwrap();
            (encrypted, secret)
        }

        /// Evaluate `program` on `encrypted`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn evaluate(
            &self,
            parameters: &RnsBggParameters,
            program: &ExponentLutProgram,
            binding: &RnsBggDynamicBinding,
            output_use: RnsOutputUse,
            online_eval: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
            encrypted: &StageArtifacts<RnsBggEncryptionDescriptor>,
            execution: &mut CpuExecution,
            store: &mut MemoryArtifactStore,
        ) -> Result<StageOutputs, ApiError> {
            crate::api::online_eval(
                program,
                &ParamEnv::default(),
                online_eval,
                encrypted,
                &self.hash_key,
                RnsOnlineEvalBuild::new(parameters, self.context)
                    .with_output_use(output_use)
                    .with_dynamic_setup(&self.setup, binding.clone())
                    .map_err(ApiError::scheme)?,
                execution,
                store,
                |_, outputs| Ok(outputs),
                |_| Ok(()),
            )
        }
    }

    /// The four key ports of output `wire` that preprocessing published, as
    /// host matrices.
    pub(crate) fn preprocessed_key(
        parameters: &RnsBggParameters,
        online_eval: &StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
        wire: WireId,
        columns: usize,
        execution: &mut CpuExecution,
        store: &mut MemoryArtifactStore,
    ) -> Vec<mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix> {
        let ring = parameters.ring(0).unwrap();
        let mut graph = DslContext::new("dynamic-preprocessed-key");
        for port in 0..4 {
            graph = graph
                .output(
                    format!("key-{port}"),
                    ring.artifact_input(
                        online_eval.production().clone(),
                        format!("public-key/{wire:?}.{port}"),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    ),
                )
                .unwrap();
        }
        let outputs = crate::api::execute_graph(
            StageGraph::new(
                graph.build().unwrap(),
                BTreeMap::new(),
                BTreeMap::from([(
                    online_eval.production().clone(),
                    online_eval.manifest().clone(),
                )]),
            ),
            &ParamEnv::default(),
            execution,
            store,
            |_| Ok(()),
        )
        .unwrap();
        (0..4).map(|port| download(execution, &outputs, &format!("key-{port}"))).collect()
    }

    /// Download one matrix output.
    pub(crate) fn download(
        execution: &CpuExecution,
        outputs: &StageOutputs,
        name: &str,
    ) -> mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix {
        <CpuExecution as StageExecution<MemoryArtifactStore>>::download_matrix(
            execution,
            &outputs.outputs[name],
        )
        .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::{test_support::*, *};
    use crate::program::{FamilyRange, LutOutputRepresentation};
    use mxx_backends::{MemoryArtifactStore, poly::dcrt::poly::DCRTPoly};

    type Plaintexts = (PlaintextInputs<WireId, DCRTPoly>, PlaintextInputs<u32, DCRTPoly>);

    /// Compile `program` once against its setup, then encrypt twice under
    /// fresh secrets and sessions, with RHS values chosen only then, verify
    /// each output, and check that every online evaluation key is the key
    /// compiled once: it depends on setup matrices only.
    fn check_sessions(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
        layout: ExpLayout,
        batch: impl Fn() -> (Plaintexts, DCRTPoly, DCRTPoly),
    ) {
        let mut execution = execution(parameters);
        let mut store = MemoryArtifactStore::default();
        let (setup, binding) = Setup::for_program(parameters, program, layout);
        let (_, online_eval) = setup
            .preprocess(
                parameters,
                program,
                &binding,
                RnsOutputUse::DirectDecode,
                &mut execution,
                &mut store,
            )
            .unwrap();
        let Operand::Scalar(output) = program.outputs()[0] else { panic!("scalar output") };
        let key = preprocessed_key(parameters, &online_eval, output, 1, &mut execution, &mut store);
        let mut rows = Vec::new();
        for _ in 0..2 {
            let ((lhs, rhs), expected, wrong) = batch();
            let (encrypted, secret) =
                setup.encrypt(parameters, &lhs, &rhs, &mut execution, &mut store);
            let result = setup
                .evaluate(
                    parameters,
                    program,
                    &binding,
                    RnsOutputUse::DirectDecode,
                    &online_eval,
                    &encrypted,
                    &mut execution,
                    &mut store,
                )
                .unwrap();
            let verify = |plaintext: &DCRTPoly, execution: &mut _, store: &mut _| {
                verify_output(
                    program,
                    &result,
                    0,
                    plaintext,
                    parameters,
                    &secret,
                    None,
                    RnsOutputUse::DirectDecode,
                    execution,
                    store,
                )
            };
            verify(&expected, &mut execution, &mut store).unwrap();
            assert!(
                verify(&wrong, &mut execution, &mut store).is_err(),
                "a wrong Dynamic output must be rejected"
            );
            for (port, key) in key.iter().enumerate() {
                assert_eq!(
                    &download(&execution, &result, &format!("output-0.{port}")),
                    key,
                    "the online key must be the compiled key"
                );
            }
            rows.push(download(&execution, &result, "output-0.5"));
        }
        assert_ne!(rows[0], rows[1], "two sessions must sample their own material");
    }

    /// Two successive products by a scalar RHS and a family RHS member,
    /// both chosen at encryption after compilation.
    #[test]
    fn dynamic_private_products_verify_late_rhs_under_one_setup() {
        let parameters = parameters(4);
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let first = builder.rhs_input().unwrap();
        let second = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(first)).unwrap();
        let output = builder
            .mul_private(
                Operand::Scalar(product),
                Operand::Member { family: second, index: IntExpr::constant(1) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_sessions(&parameters, &program, ExpLayout { stride_log2: 0, width_log2: 0 }, || {
            // Products past X^n are negated monomials.
            let exponents = [0; 4].map(|_| rand::random::<u32>() as usize % dimension);
            let [x, y, z0, z1] = exponents.map(|exponent| monomial(&parameters, exponent));
            let expected = x.clone() * y.clone() * z1.clone();
            let wrong = expected.clone() * monomial(&parameters, 1);
            (
                (
                    BTreeMap::from([(input, InputBinding::Scalar(x))]),
                    BTreeMap::from([
                        (first.0, InputBinding::Scalar(y)),
                        (second.0, InputBinding::Family(vec![z0, z1])),
                    ]),
                ),
                expected,
                wrong,
            )
        });
    }

    /// A product inside a Program family body over member 1 only, by the RHS
    /// member the body index selects, then by a scalar RHS.
    #[test]
    fn dynamic_family_body_products_verify_indexed_rhs_members() {
        let parameters = parameters(4);
        let mut builder = ExponentLutProgramBuilder::new();
        let inputs =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let scale = builder.rhs_input().unwrap();
        let range =
            FamilyRange::bounded(IntExpr::constant(1), IntExpr::constant(1), IntExpr::constant(2))
                .unwrap();
        let products = builder
            .parallel(range, |builder, index| {
                builder
                    .mul_private(
                        Operand::Member { family: inputs, index: index.clone() },
                        Operand::Member { family: rhs, index },
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let sum = builder.sum(products).unwrap();
        let output = builder.mul_private(Operand::Scalar(sum), Operand::Scalar(scale)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_sessions(&parameters, &program, ExpLayout { stride_log2: 0, width_log2: 0 }, || {
            let exponents = [0; 5].map(|_| rand::random::<u32>() as usize % dimension);
            let [x0, x1, y0, y1, z] = exponents.map(|exponent| monomial(&parameters, exponent));
            let expected = x1.clone() * y1.clone() * z.clone();
            let wrong = expected.clone() * monomial(&parameters, 1);
            (
                (
                    BTreeMap::from([(inputs, InputBinding::Family(vec![x0, x1]))]),
                    BTreeMap::from([
                        (rhs.0, InputBinding::Family(vec![y0, y1])),
                        (scale.0, InputBinding::Scalar(z)),
                    ]),
                ),
                expected,
                wrong,
            )
        });
    }

    /// A LUT on a width-four group, `Sum_k SMul(Aut_k(c))`, then a product
    /// by a late RHS: the automorphism packages switch both secrets.
    #[test]
    fn dynamic_lut_verifies_through_a_late_rhs_product() {
        let parameters = parameters(4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let values = (0..4).map(|_| rand::random::<u32>() as usize % 4).collect::<Vec<_>>();
        let table = builder
            .register_lut(
                LutTable::unary(
                    4,
                    layout,
                    4,
                    values.clone(),
                    LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let looked_up = builder.lut(Operand::Scalar(input), table).unwrap();
        let output = builder.mul_private(Operand::Scalar(looked_up), Operand::Scalar(rhs)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        check_sessions(&parameters, &program, layout, || {
            let u = rand::random::<u32>() as usize % 4;
            let exponent = rand::random::<u32>() as usize % dimension;
            let expected = monomial(&parameters, values[u]) * monomial(&parameters, exponent);
            let wrong =
                monomial(&parameters, (values[u] + 1) % 4) * monomial(&parameters, exponent);
            (
                (
                    BTreeMap::from([(input, InputBinding::Scalar(monomial(&parameters, u)))]),
                    BTreeMap::from([(
                        rhs.0,
                        InputBinding::Scalar(monomial(&parameters, exponent)),
                    )]),
                ),
                expected,
                wrong,
            )
        });
    }

    /// The largest centered coefficient of a downloaded row.
    fn row_norm(
        parameters: &RnsBggParameters,
        row: &[mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix],
    ) -> num_bigint::BigUint {
        use mxx_backends::{matrix::PolyMatrix, poly::Poly};
        let q = parameters.level(0).unwrap().modulus();
        let q = q.as_ref();
        row.iter()
            .flat_map(|block| {
                (0..block.col_size())
                    .flat_map(move |column| block.entry(0, column).coeffs_biguints())
            })
            .map(|coefficient| coefficient.clone().min(q - &coefficient))
            .max()
            .unwrap()
    }

    /// Under the encryption secret, every nonidentity automorphism `k` of
    /// the setup has a helper with `h_k - (s H_k - s_k G)` within the helper
    /// cutoff (the mask switch from `s_k` back to `s`), and the switching
    /// ciphertext `C_k` derived from the helper of `k^-1` has
    /// `s_k C_k - s G = -sigma_k(e_(h, k^-1))` within the same cutoff (the
    /// payload switch). The group has nontrivial inverses, and `k = 1` has
    /// no package.
    #[test]
    fn automorphism_packages_switch_both_secrets() {
        let parameters = parameters(2);
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let setup = shared_setup(&parameters);
        let group = &setup.setup;
        assert!(group.automorphism_member(1).is_err());
        let switched = group.nonidentity_automorphisms().to_vec();
        assert_eq!(switched.len(), 3);
        assert!(switched.iter().any(|k| group.automorphism_inverse(*k).unwrap() != *k));
        let rhs = BTreeMap::from([
            (0, InputBinding::Family(vec![monomial(&parameters, 0); 3])),
            (1, InputBinding::Scalar(monomial(&parameters, 0))),
        ]);
        let lhs = BTreeMap::from([(WireId(0), InputBinding::Scalar(monomial(&parameters, 1)))]);
        let (encrypted, secret) =
            setup.encrypt(&parameters, &lhs, &rhs, &mut execution, &mut store);
        let ring = parameters.ring(0).unwrap();
        let columns = parameters.component_columns(0).unwrap();
        let gadget = parameters.public_gadget(0).unwrap();
        let mut runtime = BTreeMap::new();
        secret.bind_runtime(&mut runtime);
        runtime.insert(
            HASH_KEY_INPUT.to_owned(),
            mxx_backends::RuntimeValue::Bytes(setup.hash_key.as_slice().into()),
        );
        let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
        let helpers = [0, 1].map(|port| {
            ring.family_artifact_input(
                encrypted.production().clone(),
                format!("{AUTOMORPHISM_HELPER_ROLE}.{port}"),
                IntExpr::constant(switched.len()),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        });
        let helper = |k: usize| {
            let member = Int::constant(group.automorphism_member(k).unwrap());
            (helpers[0].at(member.clone()), helpers[1].at(member))
        };
        let s = secret.row().clone();
        let times = |row: &RnsBggRow, matrix: &RnsBggMatrix| {
            (
                row.0.clone() * matrix.0.0.clone() + row.1.clone() * matrix.1.0.clone(),
                row.0.clone() * matrix.0.1.clone() + row.1.clone() * matrix.1.1.clone(),
            )
        };
        let mut graph = DslContext::new("dynamic-automorphism-relations");
        for k in &switched {
            let routed = IntExpr::constant(*k);
            let s_k = (s.0.clone().ring_automorphism(routed.clone()), s.1.clone());
            let s_k_gadget = (s_k.0.clone() * gadget.clone(), gadget.clone());
            let public = group.helper_key(&parameters, hash_key.clone(), routed.clone()).unwrap();
            let inverse = group.automorphism_inverse(*k).unwrap();
            let ciphertext = automorphism_ciphertext(
                &parameters,
                group,
                hash_key.clone(),
                routed,
                IntExpr::constant(inverse),
                helper(inverse),
            )
            .unwrap();
            let payload = times(&s_k, &ciphertext);
            let residuals = [
                (
                    "helper",
                    (
                        helper(*k).0 - times(&s, &public).0 + s_k_gadget.0.clone(),
                        helper(*k).1 - times(&s, &public).1 + s_k_gadget.1.clone(),
                    ),
                ),
                ("switch", (payload.0 - s.0.clone() * gadget.clone(), payload.1 - gadget.clone())),
            ];
            for (name, residual) in residuals {
                graph = graph
                    .output(format!("{name}-{k}.0"), residual.0)
                    .unwrap()
                    .output(format!("{name}-{k}.1"), residual.1)
                    .unwrap();
            }
        }
        let outputs = crate::api::execute_graph(
            StageGraph::new(
                graph.build().unwrap(),
                runtime,
                BTreeMap::from([(encrypted.production().clone(), encrypted.manifest().clone())]),
            ),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let cutoff = group.distributions().helper.cutoff();
        for k in &switched {
            for name in ["helper", "switch"] {
                let row = [0, 1]
                    .map(|port| download(&execution, &outputs, &format!("{name}-{k}.{port}")));
                let norm = row_norm(&parameters, &row);
                assert!(&norm <= cutoff, "{name} residual of k = {k}: {norm} > {cutoff}");
            }
        }
    }

    /// The expansion `ModSwitch_(pq -> q)(c G~)` of compact entries agrees on the CPU and the
    /// GPU, multi-limb on both sides, for entries at the rounding boundaries (`p/2` and its
    /// neighbours, `pq - 1`, `(pq +- 1)/2`) and for a hashed compact key entry.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(dcrt_runtime)]
    fn compact_expansion_agrees_on_the_cpu_and_the_gpu() {
        use mxx_backends::{
            GpuRuntime,
            backend::poly_gpu::gpu_backend,
            matrix::PolyMatrix,
            poly::{Poly, PolyParams, dcrt::gpu::GpuDCRTPolyParams},
        };
        let parameters = parameters(2);
        let extension = compact_extension(&parameters);
        let setup = Setup::new(
            &parameters,
            BTreeMap::new(),
            BTreeMap::new(),
            ExpLayout { stride_log2: 0, width_log2: 1 },
        );
        let setup = setup.setup.with_compact_encodings(&parameters, extension.clone()).unwrap();
        let level = parameters.level(0).unwrap().clone();
        let q = level.modulus().as_ref().clone();
        let p = extension
            .iter()
            .map(|prime| num_bigint::BigUint::from(*prime))
            .product::<num_bigint::BigUint>();
        let pq = &p * &q;
        let moduli =
            level.to_crt().0.into_iter().chain(extension.iter().copied()).collect::<Vec<_>>();
        let compact = DCRTPolyParams::new(
            level.ring_dimension(),
            moduli.len(),
            level.crt_bits(),
            level.base_bits(),
            Some(moduli),
            None,
        );
        let half = &p / 2u8;
        let mut values = vec![
            num_bigint::BigUint::from(0u8),
            num_bigint::BigUint::from(1u8),
            half.clone(),
            &half + 1u8,
            &half - 1u8,
            &p + &half,
            &pq - 1u8,
            &pq - &half,
            (&pq - 1u8) / 2u8,
            (&pq + 1u8) / 2u8,
        ];
        while values.len() < level.ring_dimension() as usize {
            values.push(num_bigint::BigUint::from(rand::random::<u64>()) % &pq);
        }
        let ring = setup.compact_ring(&parameters, 0).unwrap().unwrap();
        let entry = ring.polynomial(
            values
                .iter()
                .map(|value| num_bigint::BigInt::from(value.clone()).into())
                .collect::<Vec<_>>(),
        );
        let gadget = parameters.public_gadget(0).unwrap();
        let hash_key = ring.bytes_input("hash-key", 32);
        let hashed = setup.compact_helper_key(&parameters, hash_key, IntExpr::constant(3)).unwrap();
        let expand = |entry| {
            crate::rns_bgg::dynamic::expand_compact_entry(&parameters, &setup, 0, &gadget, entry)
                .unwrap()
        };
        let graph = || {
            DslContext::new("compact-expansion")
                .output("crafted", expand(entry.clone()))
                .unwrap()
                .output("hashed", expand(hashed.0.0.clone()))
                .unwrap()
                .build()
                .unwrap()
        };
        let key: [u8; 32] = rand::random();
        let runtime = || {
            BTreeMap::from([(
                "hash-key".to_owned(),
                mxx_backends::RuntimeValue::Bytes(key.as_slice().into()),
            )])
        };
        let mut cpu = compact_execution(&parameters, &extension);
        let mut store = MemoryArtifactStore::default();
        let cpu_outputs = crate::api::execute_graph(
            StageGraph::new(graph(), runtime(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut cpu,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        let mut gpu = GpuRuntime::new(gpu_backend([level.clone(), compact].map(|ring| {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        })))
        .unwrap();
        let mut gpu_store = MemoryArtifactStore::default();
        let gpu_outputs = crate::api::execute_graph(
            StageGraph::new(graph(), runtime(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut gpu,
            &mut gpu_store,
            |_| Ok(()),
        )
        .unwrap();
        for name in ["crafted", "hashed"] {
            let host = download(&cpu, &cpu_outputs, name);
            let device = <GpuRuntime as StageExecution<MemoryArtifactStore>>::download_matrix(
                &gpu,
                &gpu_outputs.outputs[name],
            )
            .unwrap();
            assert_eq!(host.col_size(), device.col_size());
            for column in 0..host.col_size() {
                assert_eq!(
                    host.entry(0, column).coeffs_biguints(),
                    device.entry(0, column).coeffs_biguints(),
                    "{name} column {column}"
                );
            }
        }
    }

    /// The measured error of a Dynamic product and of a Dynamic exponent
    /// LUT (flat, compact, and over a radix-2 decomposition), `c - s (A - mu G)` of each
    /// full-gadget output, is within the noise simulator's bound under the setup's contracts.
    #[test]
    fn dynamic_residuals_stay_within_the_simulated_bounds() {
        let parameters = parameters(4);
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let u = rand::random::<u32>() as usize % 4;
        let v = rand::random::<u32>() as usize % dimension;
        let values = (0..4).map(|_| rand::random::<u32>() as usize % 4).collect::<Vec<_>>();
        let product = {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let rhs = builder.rhs_input().unwrap();
            let output = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
            // Past `X^n`, the product is a negated monomial.
            let payload = monomial(&parameters, u) * monomial(&parameters, v);
            (builder.finish(vec![Operand::Scalar(output)]).unwrap(), payload)
        };
        let lookup = {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let table = builder
                .register_lut(
                    LutTable::unary(
                        4,
                        layout,
                        4,
                        values.clone(),
                        LutOutputRepresentation::Exponent(layout),
                    )
                    .unwrap(),
                )
                .unwrap();
            let output = builder.lut(Operand::Scalar(input), table).unwrap();
            (
                builder.finish(vec![Operand::Scalar(output)]).unwrap(),
                monomial(&parameters, values[u]),
            )
        };
        let extension = compact_extension(&parameters);
        // Flat and compact encodings, and the lookup over a radix-2 decomposition, whose branch
        // `g^3` is switched twice.
        for ((program, payload), (compact, radix)) in [product.clone(), lookup.clone()]
            .into_iter()
            .zip([(false, None); 2])
            .chain([product, lookup.clone()].into_iter().zip([(true, None); 2]))
            .chain([(lookup, (false, Some(2)))])
        {
            let mut execution = if compact {
                compact_execution(&parameters, &extension)
            } else {
                execution(&parameters)
            };
            let mut store = MemoryArtifactStore::default();
            let (mut setup, binding) = Setup::for_program(&parameters, &program, layout);
            if compact {
                setup.setup =
                    setup.setup.with_compact_encodings(&parameters, extension.clone()).unwrap();
            }
            if let Some(radix) = radix {
                setup.setup = setup.setup.with_lut_radix(radix).unwrap();
                assert_eq!(setup.setup.switching_automorphisms().len(), 2);
                assert_eq!(setup.setup.radix_depths().values().max(), Some(&2));
            }
            let (_, online_eval) = setup
                .preprocess(
                    &parameters,
                    &program,
                    &binding,
                    RnsOutputUse::Gadget,
                    &mut execution,
                    &mut store,
                )
                .unwrap();
            let input = *program.inputs().keys().next().unwrap();
            let lhs = BTreeMap::from([(input, InputBinding::Scalar(monomial(&parameters, u)))]);
            let rhs = program
                .rhs_inputs()
                .keys()
                .map(|id| (id.0, InputBinding::Scalar(monomial(&parameters, v))))
                .collect();
            let (encrypted, secret) =
                setup.encrypt(&parameters, &lhs, &rhs, &mut execution, &mut store);
            let result = setup
                .evaluate(
                    &parameters,
                    &program,
                    &binding,
                    RnsOutputUse::Gadget,
                    &online_eval,
                    &encrypted,
                    &mut execution,
                    &mut store,
                )
                .unwrap();

            let distributions = setup.setup.distributions();
            // A compact input is read through its expansion, bounded without moments.
            let initial = if compact {
                crate::rns_bgg::RnsBggNoiseValue::public(
                    crate::rns_bgg::compact_expansion_bound(
                        &parameters,
                        &extension,
                        distributions.input.cutoff(),
                    )
                    .unwrap(),
                    &[BigInt::from(1u8)],
                )
            } else {
                crate::rns_bgg::RnsBggNoiseValue::sampled_gaussian(
                    &distributions.input,
                    b"dynamic-residual-input",
                    &[BigInt::from(1u8)],
                )
            };
            let bindings = crate::rns_bgg::RnsBggNoiseBindings::from_inputs(
                &program,
                &[initial],
                &vec![
                    crate::rns_bgg::RnsBggNoiseRhs {
                        payload_l1: 1u8.into(),
                        payload_l2_squared: 1u8.into(),
                    };
                    program.rhs_inputs().len()
                ],
            )
            .unwrap()
            .with_output_use(RnsOutputUse::Gadget)
            .with_dynamic(
                crate::rns_bgg::RnsBggDynamicNoise::from_setup(&setup.setup, &binding, &program)
                    .unwrap(),
            );
            let workload =
                RnsBggDerivedWorkload::from_program_with_parameters(&parameters, &program).unwrap();
            let report = crate::api::noise_simulate(
                &program,
                crate::rns_bgg::RnsNoiseBuild::new(&parameters, &program, &workload, &bindings)
                    .unwrap(),
                |_, outputs| Ok(outputs),
            )
            .unwrap();
            let bound = report.outputs[0].bound.error.clone();
            let modulus = parameters.level(0).unwrap().modulus();
            assert!(&bound * 2u8 < *modulus.as_ref(), "the bound {bound} constrains nothing");

            let ring = parameters.ring(0).unwrap();
            let columns = parameters.component_columns(0).unwrap();
            let gadget = parameters.public_gadget(0).unwrap();
            let mut runtime = BTreeMap::new();
            secret.bind_runtime(&mut runtime);
            for port in 0..6 {
                runtime.insert(
                    format!("port-{port}"),
                    result.outputs[&format!("output-0.{port}")].clone(),
                );
            }
            let port = |index: usize| ring.input(format!("port-{index}"), (1, columns));
            let s = secret.row().clone();
            let mu = plaintext_constant(&ring, &payload);
            let residual = |component: usize, s_i: Mat| {
                port(4 + component) -
                    (s.0.clone() * port(component) + s.1.clone() * port(2 + component)) +
                    s_i * mu.clone() * gadget.clone()
            };
            let graph = DslContext::new("dynamic-residual")
                .output("residual-0", residual(0, s.0.clone()))
                .unwrap()
                .output("residual-1", residual(1, s.1.clone()))
                .unwrap()
                .build()
                .unwrap();
            let outputs = crate::api::execute_graph(
                StageGraph::new(graph, runtime, BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let row =
                [0, 1].map(|port| download(&execution, &outputs, &format!("residual-{port}")));
            let measured = row_norm(&parameters, &row);
            assert!(
                measured <= bound,
                "measured {measured} exceeds the simulated {bound} (compact {compact})"
            );
        }
    }

    /// A setup of one private input, a scalar RHS (coordinate 1), and an
    /// RHS family of three (coordinate 0), with the width-four group.
    fn shared_setup(parameters: &RnsBggParameters) -> Setup {
        Setup::new(
            parameters,
            BTreeMap::from([(
                0,
                crate::program::ProgramInputDeclaration::new(
                    InputBinding::Scalar(()),
                    AttributeKnowledge::PrivatePayload,
                ),
            )]),
            BTreeMap::from([
                (
                    0,
                    RnsBggDynamicRhs {
                        shape: InputBinding::Family(IntExpr::constant(3)),
                        source: RnsBggPackageSource::Sampled,
                    },
                ),
                (
                    1,
                    RnsBggDynamicRhs {
                        shape: InputBinding::Scalar(()),
                        source: RnsBggPackageSource::Sampled,
                    },
                ),
            ]),
            ExpLayout { stride_log2: 0, width_log2: 2 },
        )
    }

    /// Program A: `x * y`, an exponent LUT, `* y` again (the scalar package
    /// read at levels 0 and 2), `* z[2]`; Program B: `x * z[1]`, then a
    /// coefficient LUT. The Program-local ids differ from the setup
    /// coordinates, so the mappings are explicit.
    fn two_programs(
        values: &[usize],
        coefficients: &[usize],
    ) -> ((ExponentLutProgram, RnsBggDynamicBinding), (ExponentLutProgram, RnsBggDynamicBinding))
    {
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let mut builder = ExponentLutProgramBuilder::new();
        let x = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let z = builder.rhs_family(IntExpr::constant(3)).unwrap();
        let y = builder.rhs_input().unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(
                    4,
                    layout,
                    4,
                    values.to_vec(),
                    LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let first = builder.mul_private(Operand::Scalar(x), Operand::Scalar(y)).unwrap();
        let looked_up = builder.lut(Operand::Scalar(first), table).unwrap();
        let again = builder.mul_private(Operand::Scalar(looked_up), Operand::Scalar(y)).unwrap();
        let output = builder
            .mul_private(
                Operand::Scalar(again),
                Operand::Member { family: z, index: IntExpr::constant(2) },
            )
            .unwrap();
        let a = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let a_binding =
            RnsBggDynamicBinding::new(BTreeMap::from([(x, 0)]), BTreeMap::from([(z, 0), (y, 1)]));
        let mut builder = ExponentLutProgramBuilder::new();
        let y = builder.rhs_input().unwrap();
        let x = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let z = builder.rhs_family(IntExpr::constant(3)).unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(
                    4,
                    layout,
                    8,
                    coefficients.to_vec(),
                    LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let product = builder
            .mul_private(
                Operand::Scalar(x),
                Operand::Member { family: z, index: IntExpr::constant(1) },
            )
            .unwrap();
        let output = builder.lut(Operand::Scalar(product), table).unwrap();
        let b = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let b_binding =
            RnsBggDynamicBinding::new(BTreeMap::from([(x, 0)]), BTreeMap::from([(z, 0), (y, 1)]));
        ((a, a_binding), (b, b_binding))
    }

    /// One ciphertext, encrypted before either Program is compiled, serves
    /// two Programs that read different tables and RHS members, including a
    /// package read at two nonadjacent levels. The ciphertext holds one
    /// package per coordinate and one automorphism package set; its
    /// identity names no Program. Stage mismatches are rejected.
    #[test]
    fn one_ciphertext_serves_two_programs_and_rejects_mismatched_stages() {
        let parameters = parameters(6);
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let setup = shared_setup(&parameters);
        let dimension = parameters.level(0).unwrap().ring_dimension() as usize;
        let random = |bound: usize| rand::random::<u32>() as usize % bound;
        let (u, y, z) = (random(2), random(2), [0; 3].map(|_| random(2)));
        let lhs = BTreeMap::from([(WireId(0), InputBinding::Scalar(monomial(&parameters, u)))]);
        let rhs = BTreeMap::from([
            (0, InputBinding::Family(z.iter().map(|z| monomial(&parameters, *z)).collect())),
            (1, InputBinding::Scalar(monomial(&parameters, y))),
        ]);
        let (encrypted, secret) =
            setup.encrypt(&parameters, &lhs, &rhs, &mut execution, &mut store);
        assert!(encrypted.descriptor().binding().program().is_none());
        let artifacts = encrypted.manifest().artifacts.keys().cloned().collect::<Vec<_>>();
        let count = |pattern: &str| artifacts.iter().filter(|name| name.contains(pattern)).count();
        // Two row ports per companion family: one family per coordinate,
        // one helper family and one companion family for the automorphisms.
        assert_eq!(count("dynamic/rhs/0/digit-encoding."), 2, "{artifacts:?}");
        assert_eq!(count("dynamic/rhs/1/digit-encoding."), 2, "{artifacts:?}");
        assert_eq!(count(AUTOMORPHISM_HELPER_ROLE), 2, "{artifacts:?}");
        assert_eq!(count(AUTOMORPHISM_COMPANION_ROLE), 2, "{artifacts:?}");

        let values = [0; 4].map(|_| random(4));
        let coefficients = [0; 4].map(|_| random(8));
        let ((a, a_binding), (b, b_binding)) = two_programs(&values, &coefficients);
        let (_, a_online) = setup
            .preprocess(
                &parameters,
                &a,
                &a_binding,
                RnsOutputUse::DirectDecode,
                &mut execution,
                &mut store,
            )
            .unwrap();
        let (_, b_online) = setup
            .preprocess(
                &parameters,
                &b,
                &b_binding,
                RnsOutputUse::DirectDecode,
                &mut execution,
                &mut store,
            )
            .unwrap();
        let result = setup
            .evaluate(
                &parameters,
                &a,
                &a_binding,
                RnsOutputUse::DirectDecode,
                &a_online,
                &encrypted,
                &mut execution,
                &mut store,
            )
            .unwrap();
        // X^(L(u + y) + y + z_2), all exponents below n.
        let exponent = values[(u + y) % 4] + y + z[2];
        assert!(exponent < dimension);
        verify_output(
            &a,
            &result,
            0,
            &monomial(&parameters, exponent),
            &parameters,
            &secret,
            None,
            RnsOutputUse::DirectDecode,
            &mut execution,
            &mut store,
        )
        .unwrap();
        let result = setup
            .evaluate(
                &parameters,
                &b,
                &b_binding,
                RnsOutputUse::DirectDecode,
                &b_online,
                &encrypted,
                &mut execution,
                &mut store,
            )
            .unwrap();
        let decoded = decode_output(
            &b,
            &result,
            0,
            &parameters,
            &secret,
            None,
            RnsOutputUse::DirectDecode,
            &[0],
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(decoded, vec![BigInt::from(coefficients[u + z[1]])]);

        // Program A with Program B's compiled artifacts.
        assert!(
            setup
                .evaluate(
                    &parameters,
                    &a,
                    &a_binding,
                    RnsOutputUse::DirectDecode,
                    &b_online,
                    &encrypted,
                    &mut execution,
                    &mut store
                )
                .is_err()
        );
        // Another output use than the compiled one.
        assert!(
            setup
                .evaluate(
                    &parameters,
                    &a,
                    &a_binding,
                    RnsOutputUse::Gadget,
                    &a_online,
                    &encrypted,
                    &mut execution,
                    &mut store
                )
                .is_err()
        );
        // Another mapping than the compiled one: coordinates are not
        // Program ids.
        let swapped = RnsBggDynamicBinding::new(
            BTreeMap::from([(WireId(0), 0)]),
            BTreeMap::from([(RhsId(0), 1), (RhsId(1), 0)]),
        );
        assert!(
            setup
                .evaluate(
                    &parameters,
                    &a,
                    &swapped,
                    RnsOutputUse::DirectDecode,
                    &a_online,
                    &encrypted,
                    &mut execution,
                    &mut store
                )
                .is_err()
        );
        // A Program compiled against another setup over the same parameters.
        let other = shared_setup(&parameters);
        let (_, other_online) = other
            .preprocess(
                &parameters,
                &a,
                &a_binding,
                RnsOutputUse::DirectDecode,
                &mut execution,
                &mut store,
            )
            .unwrap();
        assert!(
            other
                .evaluate(
                    &parameters,
                    &a,
                    &a_binding,
                    RnsOutputUse::DirectDecode,
                    &other_online,
                    &encrypted,
                    &mut execution,
                    &mut store
                )
                .is_err()
        );
    }

    /// A coefficient output under the gadget use keeps its `2 x m` key and
    /// `1 x m` row, verified as a full-gadget relation of scale one; the same
    /// Program under direct decoding returns the table value modulo `t`.
    #[test]
    fn dynamic_coefficient_output_has_gadget_and_direct_carriers() {
        let parameters = parameters(4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let values = vec![3, 5];
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(2, layout, 8, values.clone(), LutOutputRepresentation::Coefficient)
                    .unwrap(),
            )
            .unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let output = builder.lut(Operand::Scalar(product), table).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let (setup, binding) = Setup::for_program(&parameters, &program, layout);
        for negative in [false, true] {
            let (bit, y) = (rand::random::<u32>() as usize % 2, 0);
            let x = monomial(&parameters, bit);
            let lhs =
                BTreeMap::from([(input, InputBinding::Scalar(if negative { -x } else { x }))]);
            let rhs = BTreeMap::from([(rhs.0, InputBinding::Scalar(monomial(&parameters, y)))]);
            let (encrypted, secret) =
                setup.encrypt(&parameters, &lhs, &rhs, &mut execution, &mut store);
            let value = if negative { -(values[bit] as i64) } else { values[bit] as i64 };
            let plaintext = |value: i64| {
                let magnitude = DCRTPoly::from_usize_to_constant(
                    &parameters.levels()[0],
                    value.unsigned_abs() as usize,
                );
                if value < 0 { -magnitude } else { magnitude }
            };
            for output_use in [RnsOutputUse::Gadget, RnsOutputUse::DirectDecode] {
                let (_, online) = setup
                    .preprocess(
                        &parameters,
                        &program,
                        &binding,
                        output_use,
                        &mut execution,
                        &mut store,
                    )
                    .unwrap();
                let result = setup
                    .evaluate(
                        &parameters,
                        &program,
                        &binding,
                        output_use,
                        &online,
                        &encrypted,
                        &mut execution,
                        &mut store,
                    )
                    .unwrap();
                let columns = mxx_backends::matrix::PolyMatrix::col_size(&download(
                    &execution,
                    &result,
                    "output-0.5",
                ));
                let expected_columns = match output_use {
                    RnsOutputUse::Gadget => parameters.level(0).unwrap().modulus_digits(),
                    RnsOutputUse::DirectDecode => 1,
                };
                assert_eq!(columns, expected_columns, "{output_use:?}");
                let verify = |value: i64, execution: &mut _, store: &mut _| {
                    verify_output(
                        &program,
                        &result,
                        0,
                        &plaintext(value),
                        &parameters,
                        &secret,
                        None,
                        output_use,
                        execution,
                        store,
                    )
                };
                verify(value, &mut execution, &mut store).unwrap();
                assert!(verify(value + 1, &mut execution, &mut store).is_err(), "{output_use:?}");
                let decoded = decode_output(
                    &program,
                    &result,
                    0,
                    &parameters,
                    &secret,
                    None,
                    output_use,
                    &[0],
                    &mut execution,
                    &mut store,
                    |_| Ok(()),
                );
                match output_use {
                    RnsOutputUse::Gadget => {
                        assert!(decoded.is_err(), "a gadget output is not decodable")
                    }
                    RnsOutputUse::DirectDecode => {
                        let t = RnsBggParameters::coefficient_decoding_modulus(5).unwrap() as i64;
                        assert_eq!(decoded.unwrap(), vec![BigInt::from(value.rem_euclid(t))]);
                    }
                }
            }
        }
    }

    /// Every Dynamic evaluation level is the one physical ring: the online
    /// graph has no modulus conversion, and a chain without a gadget
    /// successor after an evaluated level is rejected before execution.
    #[test]
    fn dynamic_evaluation_converts_no_modulus_and_needs_a_successor() {
        let parameters = parameters(3);
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(product)]).unwrap();
        let mut execution = execution(&parameters);
        let mut store = MemoryArtifactStore::default();
        let layout = ExpLayout { stride_log2: 0, width_log2: 0 };
        let (setup, binding) = Setup::for_program(&parameters, &program, layout);
        let (_, online) = setup
            .preprocess(
                &parameters,
                &program,
                &binding,
                RnsOutputUse::Gadget,
                &mut execution,
                &mut store,
            )
            .unwrap();
        let lhs = BTreeMap::from([(input, InputBinding::Scalar(monomial(&parameters, 1)))]);
        let rhs_values = BTreeMap::from([(rhs.0, InputBinding::Scalar(monomial(&parameters, 2)))]);
        let (encrypted, _) =
            setup.encrypt(&parameters, &lhs, &rhs_values, &mut execution, &mut store);
        crate::api::online_eval(
            &program,
            &ParamEnv::default(),
            &online,
            &encrypted,
            &setup.hash_key,
            RnsOnlineEvalBuild::new(&parameters, setup.context)
                .with_output_use(RnsOutputUse::Gadget)
                .with_dynamic_setup(&setup.setup, binding.clone())
                .unwrap(),
            &mut execution,
            &mut store,
            |_, outputs| Ok(outputs),
            |validated| {
                for (_, scope) in validated.source.scopes() {
                    for node in scope.nodes() {
                        assert!(!matches!(
                            node.kind(),
                            mxx_ir_core::node::NodeKind::BlockModSwitch { .. } |
                                mxx_ir_core::node::NodeKind::ModulusSwitch { .. } |
                                mxx_ir_core::node::NodeKind::ModulusReduce { .. }
                        ));
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        // Two levels: the product's output would be the terminal level.
        let short = super::test_support::parameters(2);
        let (setup, binding) = Setup::for_program(&short, &program, layout);
        assert!(
            setup
                .preprocess(
                    &short,
                    &program,
                    &binding,
                    RnsOutputUse::DirectDecode,
                    &mut execution,
                    &mut store
                )
                .is_err()
        );
    }
}
