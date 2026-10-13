//! The setup schema of Dynamic RNS-BGG evaluation.
//!
//! A Dynamic setup fixes everything encryption depends on before any Program
//! is chosen: the physical ring and gadget (one ordered CRT basis shared by
//! every logical level), the canonical input and RHS coordinates with their
//! shapes, the automorphism group `H` of the lookup tables a Program may
//! evaluate, the sampling contracts of the encodings encryption samples, and
//! the public matrices, which are expanded from the shared hash key under
//! tags that name only setup coordinates. It names no Program, LUT table,
//! read map, gate, or logical level.
//!
//! Encryption under a setup produces, per ciphertext, one *package* per RHS
//! coordinate (member) and per nonidentity automorphism: a gadget ciphertext
//! `C` with `s' C = z G + e_C` and the digit companions
//! `z_{a,j} = s' (B_{a,j} - d_{a,j} G) + eps_{a,j}` of its gadget digits
//! `d = Dec_G(C)`. A Program then reads a package at any logical level: the
//! multiplication at a gate whose key is `A` and whose decomposition is
//! `U = Dec_G(A)` uses `T_j = sum_a B_{a,j} U_{:,a}` and
//! `c'_j = sum_a z_{a,j} U_{:,a} + c d_{:,j}`, which cancel exactly for every
//! `A` (see [`super::companion_columns`]). The packages therefore do not
//! depend on the consuming gate, level, or Program.
//!
//! The setup matrices are expanded from the hash key, a public seed: the
//! manuscript's setup samples them uniformly, and the seeded expansion is
//! assumed to be indistinguishable from that distribution.
use super::*;
use crate::program::{ExpLayout, ExponentLutProgram, ProgramInputDeclaration};

/// A discrete Gaussian sampling contract: the width `sigma` of the existing
/// sampler (a standard-deviation parameter) and its hard coefficient cutoff.
/// A zero sigma samples nothing (noise-free encodings).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggGaussian {
    #[serde(with = "crate::noise_utils::bigdecimal_string")]
    sigma: BigDecimal,
    cutoff: BigUint,
}

impl RnsBggGaussian {
    /// A contract with an explicit cutoff, which must be positive for a
    /// positive sigma.
    pub fn new(sigma: BigDecimal, cutoff: BigUint) -> Result<Self, RnsBggError> {
        let sigma = canonical_encoding_sigma(sigma)?;
        if !sigma.is_zero() && cutoff.is_zero() {
            return Err(RnsBggError::Parameters("a positive sigma needs a positive cutoff"));
        }
        Ok(Self { cutoff: if sigma.is_zero() { BigUint::zero() } else { cutoff }, sigma })
    }

    /// A contract with the default cutoff `floor(6.5 sigma)` of
    /// [`mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound`].
    pub fn with_default_cutoff(sigma: BigDecimal) -> Result<Self, RnsBggError> {
        let sigma = canonical_encoding_sigma(sigma)?;
        let cutoff = encoding_error_bound(&sigma)?;
        Ok(Self { sigma, cutoff })
    }

    pub fn sigma(&self) -> &BigDecimal {
        &self.sigma
    }

    pub fn cutoff(&self) -> &BigUint {
        &self.cutoff
    }

    /// One `1 x columns` sample over `ring`; zero when sigma is zero.
    pub(crate) fn sample(&self, ring: &Ring, columns: usize) -> Result<Mat, RnsBggError> {
        if self.sigma.is_zero() {
            return Ok(ring.zero((1, columns)));
        }
        let sigma = crate::utils::decimal_to_real_expr(&self.sigma)
            .map_err(|_| RnsBggError::Parameters("invalid Gaussian sigma"))?;
        Ok(ring.gaussian((1, columns), sigma, self.cutoff.clone()))
    }

    /// One row of two `1 x columns` samples.
    pub(crate) fn sample_row(&self, ring: &Ring, columns: usize) -> Result<RnsBggRow, RnsBggError> {
        Ok((self.sample(ring, columns)?, self.sample(ring, columns)?))
    }
}

/// The sampling contracts of the encodings Dynamic encryption samples
/// itself. The noise simulation reads the same contracts from the setup, so
/// a sampled width and an analysed bound cannot differ.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDynamicDistributions {
    /// Errors of the input encodings.
    pub input: RnsBggGaussian,
    /// Errors of the digit companion encodings of every package.
    pub companion: RnsBggGaussian,
    /// Errors of the automorphism helpers `h_k = s H_k - s_k G + e`.
    pub helper: RnsBggGaussian,
    /// Errors `e_C` of the bottom rows of the RHS gadget ciphertexts
    /// ([`RnsBggDynamicSetup::rhs_bottom`]).
    pub ciphertext: RnsBggGaussian,
}

impl RnsBggDynamicDistributions {
    /// Every role with the same contract.
    pub fn uniform(contract: RnsBggGaussian) -> Self {
        Self {
            input: contract.clone(),
            companion: contract.clone(),
            helper: contract.clone(),
            ciphertext: contract,
        }
    }
}

/// The kind of a package, which names its public companion keys.
#[derive(Clone, Debug)]
pub(crate) enum RnsBggPackage {
    /// Member `member` (if a family) of RHS coordinate `coordinate`.
    Rhs { coordinate: u32, member: Option<IntExpr> },
    /// The switching package of automorphism index `k`.
    Automorphism(IntExpr),
}

/// A validated Dynamic setup (see the module documentation).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDynamicSetup {
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    parameters: [u8; 32],
    ring_dimension: u32,
    inputs: BTreeMap<u32, ProgramInputDeclaration>,
    /// The shape of every RHS coordinate.
    rhs: BTreeMap<u32, InputShape>,
    automorphism_layout: ExpLayout,
    automorphisms: Vec<usize>,
    /// The radix `B` of the LUT automorphism decomposition (see
    /// [`Self::with_lut_radix`]); `None` switches every automorphism directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lut_radix: Option<usize>,
    /// The CRT primes of the auxiliary modulus `p` of compact encodings (see
    /// [`Self::with_compact_encodings`]); `None` publishes every encoding at `q`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compact_extension: Option<Vec<u64>>,
    distributions: RnsBggDynamicDistributions,
    /// The digest of the setup without its hash key: what the hash tags of
    /// its public matrices name. The hash key, which keys every expansion,
    /// separates two setups of one schema.
    schema: [u8; 32],
    identity: [u8; 32],
}

impl RnsBggDynamicSetup {
    /// Validate a setup over `parameters`, which must be Dynamic with every
    /// level on one ordered CRT basis (see
    /// [`RnsBggParameters::validate_equal_basis`]). `layout` fixes the
    /// automorphism group `H`: its indices, in ascending order (which starts
    /// at the identity `1`), are the canonical automorphism coordinates; a
    /// Program LUT may use any input layout whose automorphisms lie in `H`.
    /// Family counts must be concrete and positive.
    pub fn new(
        parameters: &RnsBggParameters,
        context: [u8; 32],
        hash_key: &[u8],
        inputs: BTreeMap<u32, ProgramInputDeclaration>,
        rhs: BTreeMap<u32, InputShape>,
        layout: ExpLayout,
        distributions: RnsBggDynamicDistributions,
    ) -> Result<Self, RnsBggError> {
        if parameters.rhs_mode() != RhsMode::Dynamic {
            return Err(RnsBggError::Parameters("a Dynamic setup needs Dynamic parameters"));
        }
        parameters.validate_equal_basis()?;
        let ring_dimension = parameters.level(0)?.ring_dimension();
        let shapes = inputs.values().map(ProgramInputDeclaration::shape).chain(rhs.values());
        for shape in shapes {
            if let InputBinding::Family(count) = shape {
                count
                    .evaluate(&mxx_ir_core::ParamEnv::default())
                    .ok()
                    .filter(|count| count.sign() == num_bigint::Sign::Plus)
                    .ok_or(RnsBggError::Parameters("a setup family count must be concrete"))?;
            }
        }
        let automorphisms = layout
            .automorphism_indices(ring_dimension as usize)
            .map_err(|_| RnsBggError::Parameters("invalid setup automorphism layout"))?;
        let order = 2 * ring_dimension as usize;
        if automorphisms.first() != Some(&1) ||
            automorphisms.iter().any(|k| {
                automorphism_inverse(*k, ring_dimension as usize)
                    .is_none_or(|inverse| automorphisms.binary_search(&inverse).is_err()) ||
                    *k >= order
            })
        {
            return Err(RnsBggError::Parameters("setup automorphisms are not a group"));
        }
        let mut setup = Self {
            context,
            hash_key_commitment: hash_key_commitment(hash_key),
            parameters: parameters.parameter_identity(),
            ring_dimension,
            inputs,
            rhs,
            automorphism_layout: layout,
            automorphisms,
            lut_radix: None,
            compact_extension: None,
            distributions,
            schema: [0; 32],
            identity: [0; 32],
        };
        setup.seal();
        Ok(setup)
    }

    /// Fix the schema and identity digests of the current fields.
    fn seal(&mut self) {
        self.schema = stable_digest(
            b"mxx-rns-bgg/dynamic-schema/v1",
            &Self {
                hash_key_commitment: [0; 32],
                schema: [0; 32],
                identity: [0; 32],
                ..self.clone()
            },
        );
        self.identity = stable_digest(
            b"mxx-rns-bgg/dynamic-setup/v1",
            &Self { identity: [0; 32], ..self.clone() },
        );
    }

    /// Decompose the cyclic group `H = <g>` (`g = 1 + 2n/|H|`) in radix `B`:
    /// with `r = ceil(log_B |H|)` rounds, the exponent `e` of a branch `k = g^e`
    /// has digits `e_j < B` for `j < r - 1` and a last digit below
    /// `B' = |H| / B^(r - 1)` (`B' = B` when `|H| = B^r`). Encryption publishes
    /// the switching packages of `K_B = {g^(a B^j)}` for the nonzero digits `a`
    /// of every round only (`(B - 1)(r - 1) + B' - 1` instead of `|H| - 1`) and
    /// the helpers of `J_B = K_B u K_B^-1`, and a branch is switched once per
    /// nonzero digit of `e`: the branch without the top digit, `parent(k)`, is
    /// switched by `g^(a_top B^top)` ([`Self::radix_step`]). Each switch adds one
    /// decomposition gain to the noise of the branches below it. `B` is a power
    /// of two below or equal to `|H|`; `B = |H|` is the direct switching of
    /// every automorphism.
    pub fn with_lut_radix(mut self, radix: usize) -> Result<Self, RnsBggError> {
        let order = self.automorphisms.len();
        if radix < 2 || !radix.is_power_of_two() || radix > order {
            return Err(RnsBggError::Parameters(
                "the LUT radix must be a power of two up to the group order",
            ));
        }
        self.lut_radix = (radix < order).then_some(radix);
        if self.lut_radix.is_some() {
            let mut powers = self.generator_powers();
            powers.sort_unstable();
            if powers != self.automorphisms {
                return Err(RnsBggError::Parameters("a radix decomposition needs a cyclic group"));
            }
        }
        self.seal();
        Ok(self)
    }

    /// Publish the digit companions of encryption in two columns over `pq`, for
    /// the auxiliary modulus `p` the product of `extension` (CRT primes
    /// `1 mod 2n`, distinct and disjoint from the setup ring's): a companion
    /// `s (A - d G) + e` mod `q` of width `m` becomes
    /// `s (A~ - p d I_2) + e` mod `pq` of width two, for a `2 x 2` key `A~`
    /// over `R_pq`, and every reader expands it to
    /// `ModSwitch_(pq -> q)(c G~) = s (A' - d G) + e'` mod `q` with
    /// `A' = ModSwitch_(pq -> q)(A~ G~)` and the integer lift `G~` of the gadget
    /// (see [`crate::rns_bgg::dynamic::expand_compact_row`]): the manuscript's
    /// compressed digit encodings and their decompression `Dc`. The helpers,
    /// the inputs, and the GSW ciphertexts stay at `q`.
    pub fn with_compact_encodings(
        mut self,
        parameters: &RnsBggParameters,
        extension: Vec<u64>,
    ) -> Result<Self, RnsBggError> {
        let base = parameters.level(0)?.to_crt().0;
        let order = 2 * u64::from(self.ring_dimension);
        if extension.is_empty() ||
            extension.iter().enumerate().any(|(index, prime)| {
                *prime < 3 ||
                    (prime - 1) % order != 0 ||
                    base.contains(prime) ||
                    extension[..index].contains(prime)
            })
        {
            return Err(RnsBggError::Parameters(
                "compact primes must be distinct, 1 mod 2n, and disjoint from the setup ring",
            ));
        }
        self.compact_extension = Some(extension);
        self.seal();
        Ok(self)
    }

    /// The CRT primes of `p`, if encodings are compact.
    pub fn compact_extension(&self) -> Option<&[u64]> {
        self.compact_extension.as_deref()
    }

    /// The ring over `q p` at `level` (the level's basis followed by `p`'s), if
    /// encodings are compact.
    pub fn compact_ring(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
    ) -> Result<Option<Ring>, RnsBggError> {
        let Some(extension) = &self.compact_extension else { return Ok(None) };
        let moduli =
            parameters.level(level)?.to_crt().0.into_iter().chain(extension.iter().copied());
        Ok(Some(Ring::from_crt_moduli(moduli.map(IntExpr::from).collect(), self.ring_dimension)))
    }

    /// The radix of the LUT automorphism decomposition, `None` if every
    /// automorphism is switched directly.
    pub fn lut_radix(&self) -> Option<usize> {
        self.lut_radix
    }

    /// `g^e mod 2n` for `e < |H|`, with `g = 1 + 2n/|H|`.
    fn generator_powers(&self) -> Vec<usize> {
        generator_powers(self.ring_dimension as usize, self.automorphisms.len())
    }

    /// The parent and the switch `(parent(k), g^(a_top B^top))` of a
    /// nonidentity branch `k = g^e`, for the top nonzero radix-`B` digit
    /// `a_top` of `e` (see [`Self::with_lut_radix`]); `(1, k)` without a
    /// radix.
    pub fn radix_step(&self, k: usize) -> Result<(usize, usize), RnsBggError> {
        radix_step(&self.automorphisms, self.ring_dimension as usize, self.lut_radix, k)
    }

    /// The number of switches of every nonidentity automorphism (its nonzero
    /// radix-`B` digits; one without a radix).
    pub fn radix_depths(&self) -> BTreeMap<usize, usize> {
        radix_depths(&self.automorphisms, self.ring_dimension as usize, self.lut_radix)
    }

    /// The automorphisms with a switching package, ascending: `H \ {1}`, or
    /// `K_B` with a radix. The member order of the companion family.
    pub fn switching_automorphisms(&self) -> Vec<usize> {
        switching_automorphisms(&self.automorphisms, self.ring_dimension as usize, self.lut_radix)
    }

    /// The automorphisms with a published helper, ascending: the switching
    /// automorphisms and their inverses (`C_k` reads the helper of `k^-1`).
    /// The member order of the helper family.
    pub fn helper_automorphisms(&self) -> Vec<usize> {
        helper_automorphisms(&self.automorphisms, self.ring_dimension as usize, self.lut_radix)
    }

    /// The member of `k` in [`Self::switching_automorphisms`].
    pub fn switching_member(&self, k: usize) -> Result<usize, RnsBggError> {
        self.switching_automorphisms()
            .iter()
            .position(|candidate| *candidate == k)
            .ok_or(RnsBggError::Parameters("automorphism has no switching package"))
    }

    /// The member of `k` in [`Self::helper_automorphisms`].
    pub fn helper_member(&self, k: usize) -> Result<usize, RnsBggError> {
        self.helper_automorphisms()
            .iter()
            .position(|candidate| *candidate == k)
            .ok_or(RnsBggError::Parameters("automorphism has no helper"))
    }

    /// The setup of one Program whose inputs and RHS are its own
    /// coordinates (input `WireId(c)` and RHS `RhsId(c)` at coordinate `c`),
    /// every RHS sampled by encryption, with the identity mapping. Generic
    /// Dynamic callers with a fixed Program use it; a scheme serving several
    /// Programs from one setup declares its coordinates itself.
    pub fn for_program(
        parameters: &RnsBggParameters,
        context: [u8; 32],
        hash_key: &[u8],
        program: &ExponentLutProgram,
        layout: ExpLayout,
        distributions: RnsBggDynamicDistributions,
    ) -> Result<(Self, RnsBggDynamicBinding), RnsBggError> {
        let inputs = program
            .inputs()
            .iter()
            .map(|(wire, declaration)| (wire.0, declaration.clone()))
            .collect();
        let rhs = program.rhs_inputs().iter().map(|(id, shape)| (id.0, shape.clone())).collect();
        let setup = Self::new(parameters, context, hash_key, inputs, rhs, layout, distributions)?;
        let binding = RnsBggDynamicBinding::new(
            program.inputs().keys().map(|wire| (*wire, wire.0)).collect(),
            program.rhs_inputs().keys().map(|id| (*id, id.0)).collect(),
        );
        binding.validate(&setup, program)?;
        Ok((setup, binding))
    }

    /// The setup identity: a digest of every field.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn context(&self) -> &[u8; 32] {
        &self.context
    }

    pub fn hash_key_commitment(&self) -> &[u8; 32] {
        &self.hash_key_commitment
    }

    pub fn inputs(&self) -> &BTreeMap<u32, ProgramInputDeclaration> {
        &self.inputs
    }

    pub fn rhs(&self) -> &BTreeMap<u32, InputShape> {
        &self.rhs
    }

    pub fn distributions(&self) -> &RnsBggDynamicDistributions {
        &self.distributions
    }

    pub fn automorphism_layout(&self) -> ExpLayout {
        self.automorphism_layout
    }

    /// The automorphism group `H` in canonical (ascending) order, starting
    /// with the identity.
    pub fn automorphisms(&self) -> &[usize] {
        &self.automorphisms
    }

    /// `H \ {1}` in canonical order: the member order of the automorphism
    /// helper and companion families.
    pub fn nonidentity_automorphisms(&self) -> &[usize] {
        &self.automorphisms[1..]
    }

    /// The member of `k` in [`Self::nonidentity_automorphisms`].
    pub fn automorphism_member(&self, k: usize) -> Result<usize, RnsBggError> {
        self.nonidentity_automorphisms()
            .iter()
            .position(|candidate| *candidate == k)
            .ok_or(RnsBggError::Parameters("automorphism is not a setup coordinate"))
    }

    /// The inverse `k^-1 mod 2n` of a setup automorphism.
    pub fn automorphism_inverse(&self, k: usize) -> Result<usize, RnsBggError> {
        automorphism_inverse(k, self.ring_dimension as usize)
            .filter(|inverse| self.automorphisms.binary_search(inverse).is_ok())
            .ok_or(RnsBggError::Parameters("automorphism is not a setup coordinate"))
    }

    /// The encryption projection of the setup: its input coordinates, as
    /// input declarations keyed by `WireId(coordinate)`, and no LUT.
    pub fn encryption_inputs(&self) -> crate::program::EncryptionInputs {
        crate::program::EncryptionInputs::from_setup(
            self.inputs
                .iter()
                .map(|(coordinate, declaration)| (WireId(*coordinate), declaration.clone())),
        )
    }

    /// Artifact role of the bottom rows of the gadget ciphertexts of RHS
    /// coordinate `coordinate` ([`Self::rhs_bottom`]): two ports `.0` and `.1`
    /// holding `(C_10, C_11)`, as a family for a family coordinate. The
    /// encryption's prelude production exports them: its own, from the bound
    /// plaintexts, or one an enclosing scheme binds.
    pub fn rhs_bottom_role(coordinate: u32, family: bool) -> String {
        if family {
            format!("dynamic/rhs/{coordinate}/family/bottom")
        } else {
            format!("dynamic/rhs/{coordinate}/bottom")
        }
    }

    /// Artifact role of the automorphism helpers `h_k` an encryption
    /// publishes: a family of rows `(1 x c, 1 x c)`, member `i` for the
    /// `i`-th of [`Self::nonidentity_automorphisms`], ports `.0` and `.1`.
    pub fn automorphism_helper_role() -> &'static str {
        super::encoding::AUTOMORPHISM_HELPER_ROLE
    }

    /// Reject parameters whose identity is not the setup's.
    pub fn validate_parameters(&self, parameters: &RnsBggParameters) -> Result<(), RnsBggError> {
        if parameters.parameter_identity() != self.parameters {
            return Err(RnsBggError::Binding("parameters do not match the Dynamic setup"));
        }
        Ok(())
    }

    /// Reject a hash key or context that is not the setup's.
    pub(crate) fn validate_public_seed(
        &self,
        context: &[u8; 32],
        hash_key_commitment: &[u8; 32],
    ) -> Result<(), RnsBggError> {
        if context != &self.context || hash_key_commitment != &self.hash_key_commitment {
            return Err(RnsBggError::Binding("context or hash key differs from the Dynamic setup"));
        }
        Ok(())
    }

    fn tag(&self, domain: &str) -> mxx_dsl::HashTag {
        let mut tag = mxx_dsl::HashTag::from(b"rns-bgg/dynamic".to_vec());
        tag.push(hex(&self.schema));
        tag.push(domain.to_owned());
        tag
    }

    /// The hash tag of the companion keys of `package`. It names the setup
    /// and the package coordinate only: no logical level, Program, LUT, or
    /// encryption session.
    pub(crate) fn package_tag(&self, package: &RnsBggPackage) -> mxx_dsl::HashTag {
        let mut tag = self.tag("companion-key");
        match package {
            RnsBggPackage::Rhs { coordinate, member } => {
                tag.push("rhs");
                tag.push(IntExpr::constant(*coordinate));
                if let Some(member) = member {
                    tag.push("member");
                    tag.push(member.clone());
                }
            }
            RnsBggPackage::Automorphism(k) => {
                tag.push("automorphism");
                tag.push(k.clone());
            }
        }
        tag
    }

    /// The public key of input coordinate `coordinate` (member `member` of a
    /// family), at the setup ring: a `2 x 2c` key with `c` the key columns of
    /// its knowledge.
    pub fn input_key(
        &self,
        parameters: &RnsBggParameters,
        hash_key: mxx_dsl::Bytes,
        coordinate: u32,
        member: Option<IntExpr>,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let declaration = self
            .inputs
            .get(&coordinate)
            .ok_or(RnsBggError::Binding("input coordinate is not in the Dynamic setup"))?;
        let mut tag = self.tag("input-key");
        tag.push(IntExpr::constant(coordinate));
        if let Some(member) = member {
            tag.push("member");
            tag.push(member);
        }
        Ok(public_key::matrix_from_hash(
            &parameters.ring(0)?,
            hash_key,
            tag,
            parameters.key_component_columns(0, declaration.knowledge())?,
        ))
    }

    /// A `2 x 2` matrix over the compact ring at `level`, hashed under `tag`.
    pub(crate) fn compact_matrix(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
        hash_key: mxx_dsl::Bytes,
        tag: mxx_dsl::HashTag,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let ring = self
            .compact_ring(parameters, level)?
            .ok_or(RnsBggError::Parameters("the setup's encodings are not compact"))?;
        Ok(public_key::matrix_from_hash(&ring, hash_key, tag, 1))
    }

    /// The public top row `a` of the gadget ciphertext `C = [a; c]` of RHS
    /// coordinate `coordinate` (member `member` of a family), a `1 x 2c` row at
    /// the setup ring hashed under the setup's tag. Every encryption reuses it
    /// under a fresh secret; its digit rows need no companion, as an
    /// automorphism package's (see [`super::automorphism_public_rows`]).
    pub fn rhs_public_top(
        &self,
        parameters: &RnsBggParameters,
        hash_key: mxx_dsl::Bytes,
        coordinate: u32,
        member: Option<IntExpr>,
    ) -> Result<RnsBggRow, RnsBggError> {
        let mut tag = self.tag("rhs-top");
        tag.push(IntExpr::constant(coordinate));
        if let Some(member) = member {
            tag.push("member");
            tag.push(member);
        }
        let ring = parameters.ring(0)?;
        let columns = parameters.component_columns(0)?;
        let half = |index: u64| {
            let mut tag = tag.clone();
            tag.push("component");
            tag.push(IntExpr::constant(index));
            ring.hash_matrix(hash_key.clone(), tag, (1, columns))
        };
        Ok((half(0), half(1)))
    }

    /// The bottom row `c = nu s G - s_0 a + e_C` of the gadget ciphertext
    /// `C = [a; c]` of member `member` of RHS coordinate `coordinate` for the
    /// `1 x 1` plaintext `nu` under the payload secret row `s = (s_0, 1)` at the
    /// setup ring, with the public top row `a` ([`Self::rhs_public_top`]) and
    /// `e_C` sampled under the setup's ciphertext contract: `s C = nu s G +
    /// e_C`. Encryption, or an enclosing scheme's prelude, exports it under
    /// [`Self::rhs_bottom_role`].
    pub fn rhs_bottom(
        &self,
        parameters: &RnsBggParameters,
        hash_key: mxx_dsl::Bytes,
        coordinate: u32,
        member: Option<IntExpr>,
        secret: &RnsBggRow,
        plaintext: mxx_dsl::Mat,
    ) -> Result<RnsBggRow, RnsBggError> {
        let gadget = parameters.public_gadget(0)?;
        let target = (
            secret.0.clone() * gadget.clone() * plaintext.clone(),
            secret.1.clone() * gadget * plaintext,
        );
        let top = self.rhs_public_top(parameters, hash_key, coordinate, member)?;
        Ok(super::ciphertext_matrix(
            parameters,
            0,
            parameters.component_columns(0)?,
            secret.0.clone(),
            top,
            target,
            &self.distributions.ciphertext,
        )?
        .1)
    }

    /// The public helper matrix `H_k` of automorphism `k`, a `2 x m` matrix
    /// at the setup ring.
    pub fn helper_key(
        &self,
        parameters: &RnsBggParameters,
        hash_key: mxx_dsl::Bytes,
        k: IntExpr,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        let mut tag = self.tag("helper-key");
        tag.push(k);
        Ok(public_key::matrix_from_hash(
            &parameters.ring(0)?,
            hash_key,
            tag,
            parameters.component_columns(0)?,
        ))
    }
}

/// The mapping of a Program's input wires and RHS ids to the coordinates of
/// a Dynamic setup. Program-local ids are not global: preprocessing, online
/// evaluation, and noise simulation validate this mapping against the
/// setup and the Program before they read any package.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDynamicBinding {
    inputs: BTreeMap<WireId, u32>,
    rhs: BTreeMap<RhsId, u32>,
}

impl RnsBggDynamicBinding {
    pub fn new(inputs: BTreeMap<WireId, u32>, rhs: BTreeMap<RhsId, u32>) -> Self {
        Self { inputs, rhs }
    }

    /// The setup coordinate of input `wire`.
    pub fn input(&self, wire: WireId) -> Result<u32, RnsBggError> {
        self.inputs.get(&wire).copied().ok_or(RnsBggError::Binding("input has no setup coordinate"))
    }

    /// The setup coordinate of RHS `rhs`.
    pub fn rhs(&self, rhs: RhsId) -> Result<u32, RnsBggError> {
        self.rhs.get(&rhs).copied().ok_or(RnsBggError::Binding("RHS has no setup coordinate"))
    }

    /// Check that this mapping covers exactly the inputs and RHS of
    /// `program`, with matching shapes and knowledge, and that every LUT of
    /// `program` uses automorphisms of the setup.
    pub fn validate(
        &self,
        setup: &RnsBggDynamicSetup,
        program: &ExponentLutProgram,
    ) -> Result<(), RnsBggError> {
        let empty = mxx_ir_core::ParamEnv::default();
        let same_shape = |left: &InputShape, right: &InputShape| match (left, right) {
            (InputBinding::Scalar(()), InputBinding::Scalar(())) => true,
            (InputBinding::Family(left), InputBinding::Family(right)) => {
                matches!((left.evaluate(&empty), right.evaluate(&empty)), (Ok(l), Ok(r)) if l == r)
            }
            _ => false,
        };
        if self.inputs.len() != program.inputs().len() ||
            self.rhs.len() != program.rhs_inputs().len()
        {
            return Err(RnsBggError::Binding("the setup mapping does not cover the Program"));
        }
        for (wire, declaration) in program.inputs() {
            let coordinate = setup
                .inputs
                .get(&self.input(*wire)?)
                .ok_or(RnsBggError::Binding("input coordinate is not in the Dynamic setup"))?;
            if coordinate.knowledge() != declaration.knowledge() ||
                !same_shape(coordinate.shape(), declaration.shape())
            {
                return Err(RnsBggError::Binding("Program input does not match its coordinate"));
            }
        }
        for (rhs, shape) in program.rhs_inputs() {
            let coordinate = setup
                .rhs
                .get(&self.rhs(*rhs)?)
                .ok_or(RnsBggError::Binding("RHS coordinate is not in the Dynamic setup"))?;
            if !same_shape(coordinate, shape) {
                return Err(RnsBggError::Binding("Program RHS does not match its coordinate"));
            }
        }
        for table in program.encryption_inputs().luts() {
            let indices = table
                .input_layout()
                .automorphism_indices(setup.ring_dimension as usize)
                .map_err(|_| RnsBggError::Lut)?;
            if indices.iter().any(|k| setup.automorphisms.binary_search(k).is_err()) {
                return Err(RnsBggError::Binding("a Program LUT uses an automorphism outside H"));
            }
        }
        Ok(())
    }
}

/// The smallest automorphism layout whose group `H` contains the
/// automorphisms of every LUT of `program`: stride zero and the largest
/// `stride_log2 + width_log2` of its LUT layouts, since a layout's indices
/// are `1 mod 2n / 2^(stride + width)`.
pub fn covering_automorphism_layout(program: &ExponentLutProgram) -> ExpLayout {
    let width_log2 = program
        .encryption_inputs()
        .luts()
        .iter()
        .map(|table| table.input_layout().stride_log2 + table.input_layout().width_log2)
        .max()
        .unwrap_or(0);
    ExpLayout { stride_log2: 0, width_log2 }
}

/// `g^e mod 2n` for `e < order`, with `g = 1 + 2n/order`: the cyclic group
/// `H` of `order` automorphisms of rings of dimension `ring_dimension`, in
/// exponent order.
fn generator_powers(ring_dimension: usize, order: usize) -> Vec<usize> {
    let modulus = 2 * ring_dimension;
    let generator = (1 + modulus / order) % modulus;
    std::iter::successors(Some(1usize), |power| Some(power * generator % modulus))
        .take(order)
        .collect()
}

/// The exponent `e` of `k = g^e` in the cyclic group `automorphisms` of rings
/// of dimension `ring_dimension` (see [`generator_powers`]).
fn generator_exponent(
    automorphisms: &[usize],
    ring_dimension: usize,
    k: usize,
) -> Result<usize, RnsBggError> {
    generator_powers(ring_dimension, automorphisms.len())
        .iter()
        .position(|power| *power == k)
        .ok_or(RnsBggError::Parameters("automorphism is not a setup coordinate"))
}

/// The radix-`radix` decomposition step `(parent(k), switch)` of a
/// nonidentity `k` (see [`RnsBggDynamicSetup::with_lut_radix`]): `(1, k)`
/// without a radix.
pub fn radix_step(
    automorphisms: &[usize],
    ring_dimension: usize,
    radix: Option<usize>,
    k: usize,
) -> Result<(usize, usize), RnsBggError> {
    if k == 1 || automorphisms.binary_search(&k).is_err() {
        return Err(RnsBggError::Parameters("a radix step needs a nonidentity setup automorphism"));
    }
    let Some(radix) = radix.filter(|radix| *radix < automorphisms.len()) else {
        return Ok((1, k));
    };
    let exponent = generator_exponent(automorphisms, ring_dimension, k)?;
    let mut top = 1;
    while top * radix <= exponent {
        top *= radix;
    }
    let switch = exponent - exponent % top;
    let powers = generator_powers(ring_dimension, automorphisms.len());
    Ok((powers[exponent - switch], powers[switch]))
}

/// The number of switches of every nonidentity automorphism of the group
/// `automorphisms` under the radix `radix`: its exponent's nonzero digits,
/// or one without a radix.
pub fn radix_depths(
    automorphisms: &[usize],
    ring_dimension: usize,
    radix: Option<usize>,
) -> BTreeMap<usize, usize> {
    let Some(radix) = radix.filter(|radix| *radix < automorphisms.len()) else {
        return automorphisms.iter().filter(|k| **k != 1).map(|k| (*k, 1)).collect();
    };
    generator_powers(ring_dimension, automorphisms.len())
        .into_iter()
        .enumerate()
        .skip(1)
        .map(|(mut exponent, k)| {
            let mut depth = 0;
            while exponent > 0 {
                depth += usize::from(exponent % radix != 0);
                exponent /= radix;
            }
            (k, depth)
        })
        .collect()
}

/// The automorphisms of the group `automorphisms` (ascending, of rings of
/// dimension `ring_dimension`) with a switching package under the radix
/// `radix` (see [`RnsBggDynamicSetup::with_lut_radix`]), ascending: `H \ {1}`
/// without a radix, else `K_B = {g^(a B^j)}`.
pub fn switching_automorphisms(
    automorphisms: &[usize],
    ring_dimension: usize,
    radix: Option<usize>,
) -> Vec<usize> {
    let mut switching = radix_depths(automorphisms, ring_dimension, radix)
        .into_iter()
        .filter(|(_, depth)| *depth == 1)
        .map(|(k, _)| k)
        .collect::<Vec<_>>();
    switching.sort_unstable();
    switching
}

/// The automorphisms with a published helper: the switching automorphisms
/// ([`switching_automorphisms`]) and their inverses, ascending.
pub fn helper_automorphisms(
    automorphisms: &[usize],
    ring_dimension: usize,
    radix: Option<usize>,
) -> Vec<usize> {
    let mut helpers = std::collections::BTreeSet::new();
    for k in switching_automorphisms(automorphisms, ring_dimension, radix) {
        helpers.insert(k);
        helpers.extend(automorphism_inverse(k, ring_dimension));
    }
    helpers.into_iter().collect()
}

/// The automorphism index `k^-1 mod 2n` inverting `X -> X^k`.
pub(crate) fn automorphism_inverse(k: usize, ring_dimension: usize) -> Option<usize> {
    let order = BigUint::from(2 * ring_dimension);
    mxx_backends::utils::mod_inverse_biguints(&(BigUint::from(k) % &order), &order)
        .and_then(|inverse| inverse.to_usize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn stable_digest<T: Serialize>(domain: &[u8], value: &T) -> [u8; 32] {
    let encoded = serde_json::to_vec(value).expect("setup schema encoding");
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((encoded.len() as u64).to_le_bytes());
    digest.update(encoded);
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cyclic group of 16 automorphisms of a ring of dimension 64 in radix 2 publishes the
    /// 4 switching packages `g^(2^j)`, in radix 4 the 6 packages `g^(a 4^j)`, and in radix 8 the
    /// 7 packages `g^a` and, in a last round of radix 2, `g^8`; every branch is
    /// its parent times its switch, its parent has one switch fewer, and its depth is its
    /// exponent's nonzero digits. The helpers are the switching automorphisms closed under
    /// inversion. Radix 16 (the group order) switches every automorphism directly.
    #[test]
    fn a_radix_decomposition_switches_each_nonzero_digit() {
        let n = 64;
        let group = ExpLayout { stride_log2: 0, width_log2: 4 }.automorphism_indices(n).unwrap();
        let powers = generator_powers(n, group.len());
        let mut sorted = powers.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, group, "the group is cyclic");
        for radix in [None, Some(16)] {
            assert_eq!(switching_automorphisms(&group, n, radix), group[1..].to_vec());
            assert_eq!(helper_automorphisms(&group, n, radix), group[1..].to_vec());
            assert!(radix_depths(&group, n, radix).values().all(|depth| *depth == 1));
        }
        for (radix, digits, packages) in [(2, 4, 4), (4, 2, 6), (8, 2, 8)] {
            let switching = switching_automorphisms(&group, n, Some(radix));
            let helpers = helper_automorphisms(&group, n, Some(radix));
            let depths = radix_depths(&group, n, Some(radix));
            assert_eq!(switching.len(), packages);
            assert!(helpers.len() <= 2 * switching.len());
            assert_eq!(depths.values().max(), Some(&digits));
            for k in &group[1..] {
                let (parent, switch) = radix_step(&group, n, Some(radix), *k).unwrap();
                assert_eq!(parent * switch % (2 * n), *k);
                assert!(switching.contains(&switch), "{switch} of {k} has no package");
                let parent_depth = if parent == 1 { 0 } else { depths[&parent] };
                assert_eq!(parent_depth + 1, depths[k]);
                let mut exponent = powers.iter().position(|power| power == k).unwrap();
                let mut nonzero = 0;
                while exponent > 0 {
                    nonzero += usize::from(exponent % radix != 0);
                    exponent /= radix;
                }
                assert_eq!(depths[k], nonzero);
            }
            for k in &switching {
                assert!(helpers.contains(k));
                assert!(helpers.contains(&automorphism_inverse(*k, n).unwrap()));
            }
        }
    }
}
