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
    /// Errors of the RHS gadget ciphertexts encryption samples itself
    /// ([`RnsBggPackageSource::Sampled`]).
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

/// How the gadget ciphertext of an RHS package is produced.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RnsBggPackageSource {
    /// Encryption samples `C = [a; t s_bar ... ]` from the RHS plaintext
    /// with a uniform top row and the setup's ciphertext error contract:
    /// `s C = nu s G + e_C`.
    Sampled,
    /// The enclosing scheme supplies `C` with `s C = nu s G + e_C` under its
    /// own construction; `error_bound` is its declared coefficient bound on
    /// `e_C`, which the noise simulation uses.
    Supplied { error_bound: BigUint },
}

/// One canonical RHS coordinate of a setup.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDynamicRhs {
    pub shape: InputShape,
    pub source: RnsBggPackageSource,
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
    rhs: BTreeMap<u32, RnsBggDynamicRhs>,
    automorphism_layout: ExpLayout,
    automorphisms: Vec<usize>,
    /// The baby-step count `n_1` of a baby-step giant-step split of `H` (see
    /// [`Self::with_baby_steps`]); `None` switches every automorphism directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    baby_steps: Option<usize>,
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
        rhs: BTreeMap<u32, RnsBggDynamicRhs>,
        layout: ExpLayout,
        distributions: RnsBggDynamicDistributions,
    ) -> Result<Self, RnsBggError> {
        if parameters.rhs_mode() != RhsMode::Dynamic {
            return Err(RnsBggError::Parameters("a Dynamic setup needs Dynamic parameters"));
        }
        parameters.validate_equal_basis()?;
        let ring_dimension = parameters.level(0)?.ring_dimension();
        let shapes = inputs
            .values()
            .map(ProgramInputDeclaration::shape)
            .chain(rhs.values().map(|rhs| &rhs.shape));
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
            baby_steps: None,
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

    /// Split the cyclic group `H = <g>` (`g = 1 + 2n/|H|`) into baby steps
    /// `B = {g^i : i < n_1}` and giant steps `G = {g^(n_1 t) : t < |H| / n_1}`:
    /// every `k = g^(n_1 t + i)` is then switched as `Aut_(g^(n_1 t))` after
    /// `Aut_(g^i)`, so encryption publishes the switching packages of
    /// `(B u G) \ {1}` only (about `2 sqrt |H|` instead of `|H| - 1`), and the
    /// helpers of those automorphisms and their inverses. A composed branch
    /// adds one decomposition gain to its noise. `n_1 = |H|` (or `1`) is the
    /// direct switching of every automorphism.
    pub fn with_baby_steps(mut self, baby_steps: usize) -> Result<Self, RnsBggError> {
        let order = self.automorphisms.len();
        if baby_steps == 0 || !order.is_multiple_of(baby_steps) {
            return Err(RnsBggError::Parameters("baby steps must divide the group order"));
        }
        self.baby_steps = (baby_steps != 1 && baby_steps != order).then_some(baby_steps);
        if self.baby_steps.is_some() {
            let mut powers = self.generator_powers();
            powers.sort_unstable();
            if powers != self.automorphisms {
                return Err(RnsBggError::Parameters("a baby-step split needs a cyclic group"));
            }
        }
        self.seal();
        Ok(self)
    }

    /// The baby-step count of the split, `None` if every automorphism is
    /// switched directly.
    pub fn baby_steps(&self) -> Option<usize> {
        self.baby_steps
    }

    /// `g^e mod 2n` for `e < |H|`, with `g = 1 + 2n/|H|`.
    fn generator_powers(&self) -> Vec<usize> {
        generator_powers(self.ring_dimension as usize, self.automorphisms.len())
    }

    /// The branches switched as a composition `Aut_g Aut_b` (both steps not
    /// the identity) under the split; empty without one.
    pub fn composed_automorphisms(&self) -> std::collections::BTreeSet<usize> {
        composed_automorphisms(
            self.ring_dimension as usize,
            self.automorphisms.len(),
            self.baby_steps,
        )
    }

    /// The giant and baby steps `(g^(n_1 t), g^i)` of `k = g^(n_1 t + i)`;
    /// `(1, k)` without a split.
    pub fn decomposition(&self, k: usize) -> Result<(usize, usize), RnsBggError> {
        let Some(baby_steps) = self.baby_steps else {
            self.automorphisms
                .binary_search(&k)
                .map_err(|_| RnsBggError::Parameters("automorphism is not a setup coordinate"))?;
            return Ok((1, k));
        };
        let powers = self.generator_powers();
        let exponent = powers
            .iter()
            .position(|power| *power == k)
            .ok_or(RnsBggError::Parameters("automorphism is not a setup coordinate"))?;
        Ok((powers[exponent - exponent % baby_steps], powers[exponent % baby_steps]))
    }

    /// The automorphisms with a switching package, ascending: `H \ {1}`, or
    /// `(B u G) \ {1}` with a split. The member order of the companion family.
    pub fn switching_automorphisms(&self) -> Vec<usize> {
        switching_automorphisms(&self.automorphisms, self.ring_dimension as usize, self.baby_steps)
    }

    /// The automorphisms with a published helper, ascending: the switching
    /// automorphisms and their inverses (`C_k` reads the helper of `k^-1`).
    /// The member order of the helper family.
    pub fn helper_automorphisms(&self) -> Vec<usize> {
        helper_automorphisms(&self.automorphisms, self.ring_dimension as usize, self.baby_steps)
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
        let rhs = program
            .rhs_inputs()
            .iter()
            .map(|(id, shape)| {
                (
                    id.0,
                    RnsBggDynamicRhs { shape: shape.clone(), source: RnsBggPackageSource::Sampled },
                )
            })
            .collect();
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

    pub fn rhs(&self) -> &BTreeMap<u32, RnsBggDynamicRhs> {
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

    /// Artifact role of the gadget ciphertexts of RHS coordinate
    /// `coordinate`: four ports `.0` to `.3` holding the blocks
    /// `((C_00, C_01), (C_10, C_11))`, as a family for a family coordinate.
    /// A scheme supplying the ciphertexts of a
    /// [`RnsBggPackageSource::Supplied`] coordinate exports them under this
    /// role from the prelude production it binds to the encryption.
    pub fn rhs_ciphertext_role(coordinate: u32, family: bool) -> String {
        if family {
            format!("dynamic/rhs/{coordinate}/family/ciphertext")
        } else {
            format!("dynamic/rhs/{coordinate}/ciphertext")
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
            if !same_shape(&coordinate.shape, shape) {
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

/// The automorphisms `g^(n_1 t + i)` with `t, i > 0` of the cyclic group of
/// `order` automorphisms split with `baby_steps = n_1` (see
/// [`RnsBggDynamicSetup::with_baby_steps`]): the branches switched as a
/// composition. Empty without a split.
pub fn composed_automorphisms(
    ring_dimension: usize,
    order: usize,
    baby_steps: Option<usize>,
) -> std::collections::BTreeSet<usize> {
    let Some(baby_steps) = baby_steps.filter(|steps| *steps > 1 && *steps < order) else {
        return std::collections::BTreeSet::new();
    };
    generator_powers(ring_dimension, order)
        .into_iter()
        .enumerate()
        .filter(|(exponent, _)| *exponent >= baby_steps && exponent % baby_steps != 0)
        .map(|(_, k)| k)
        .collect()
}

/// The automorphisms of the group `automorphisms` (ascending, of rings of
/// dimension `ring_dimension`) with a switching package under the split
/// `baby_steps` (see [`RnsBggDynamicSetup::with_baby_steps`]), ascending:
/// `H \ {1}` without a split, else `(B u G) \ {1}`.
pub fn switching_automorphisms(
    automorphisms: &[usize],
    ring_dimension: usize,
    baby_steps: Option<usize>,
) -> Vec<usize> {
    let order = automorphisms.len();
    let Some(baby_steps) = baby_steps.filter(|steps| *steps > 1 && *steps < order) else {
        return automorphisms.iter().copied().filter(|k| *k != 1).collect();
    };
    let mut switching = generator_powers(ring_dimension, order)
        .into_iter()
        .enumerate()
        .filter(|(exponent, _)| {
            *exponent != 0 && (*exponent < baby_steps || exponent % baby_steps == 0)
        })
        .map(|(_, k)| k)
        .collect::<Vec<_>>();
    switching.sort_unstable();
    switching
}

/// The automorphisms with a published helper: the switching automorphisms
/// ([`switching_automorphisms`]) and their inverses, ascending.
pub fn helper_automorphisms(
    automorphisms: &[usize],
    ring_dimension: usize,
    baby_steps: Option<usize>,
) -> Vec<usize> {
    let mut helpers = std::collections::BTreeSet::new();
    for k in switching_automorphisms(automorphisms, ring_dimension, baby_steps) {
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

    /// A split of the cyclic group of 16 automorphisms of a ring of dimension 64 into 4 baby
    /// and 4 giant steps publishes 6 switching packages instead of 15; the composed branches are
    /// the 9 with both steps nonidentity, each the product of a baby and a giant switch; and the
    /// helpers are the switching automorphisms closed under inversion.
    #[test]
    fn a_baby_step_split_switches_two_square_roots_of_the_group() {
        let n = 64;
        let group = ExpLayout { stride_log2: 0, width_log2: 4 }.automorphism_indices(n).unwrap();
        assert_eq!(switching_automorphisms(&group, n, None), group[1..].to_vec());
        assert_eq!(helper_automorphisms(&group, n, None), group[1..].to_vec());
        assert!(composed_automorphisms(n, group.len(), None).is_empty());
        let switching = switching_automorphisms(&group, n, Some(4));
        let helpers = helper_automorphisms(&group, n, Some(4));
        let composed = composed_automorphisms(n, group.len(), Some(4));
        assert_eq!(switching.len(), 6);
        assert_eq!(composed.len(), 9);
        let powers = generator_powers(n, group.len());
        let mut sorted = powers.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, group, "the group is cyclic");
        for k in &group[1..] {
            let exponent = powers.iter().position(|power| power == k).unwrap();
            let (giant, baby) = (powers[exponent - exponent % 4], powers[exponent % 4]);
            assert_eq!(giant * baby % (2 * n), *k);
            assert_eq!(composed.contains(k), giant != 1 && baby != 1);
            for step in [giant, baby].into_iter().filter(|step| *step != 1) {
                assert!(switching.contains(&step), "{step} of {k} has no package");
            }
        }
        for k in &switching {
            assert!(helpers.contains(k));
            assert!(helpers.contains(&automorphism_inverse(*k, n).unwrap()));
        }
    }
}
