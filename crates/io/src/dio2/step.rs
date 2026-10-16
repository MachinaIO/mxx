//! The Program of one step (spec §6), its refresh targets, and the root objects its inputs read.
//! The chain never drops a modulus, so one Program serves the decoders of every prime.
//!
//! The step input `S_i` is a set of root objects (rows at `Q_0`, see
//! [`mxx_exponent_lut::rns_bgg::RnsBggRoot`]): `ONEPRIV` ([`PRIVATE_TAG`]), the state
//! ([`STATE_TAG`]), the label column `0` with the one-encoding `ONEPUB` as its last member
//! ([`PUBLIC_TAG`], the PRF's input family of `v + 1` members), the other label columns
//! ([`LABEL_TAG`], `v m_v` members, entry `(r, 1 + k)` at `r m_v + k`), the LUT helpers of every
//! automorphism of [`StepPlan::automorphisms`], and the anchor hint. Every Program input reads
//! one of them under its root tag, so the samplers' inputs (each PRF builds its own accumulator
//! and input family) all read the same rows. The evaluator-bit family ([`XBITS_TAG`]) is
//! derived from `ONEPUB` outside the common body using [`xbits_rows`].
//!
//! The Program of a step that is followed by another one ([`ChainProgram::refreshing`]) computes,
//! besides the common body `Pi` with its selected data row, the σ sampler (uniform signs)
//! and its automorphisms, the CBD fresh noise of
//! every refresh part (one PRF, one row of every part's blocks), the smudges of every part and
//! prime (one PRF, one row of every smudge's columns per digit), and the label products; the last
//! step's ([`ChainProgram::last`]) computes `Pi` and the smudge of the output decoder. The
//! Program does not grow with the number of parts. Production label products are described by
//! [`LabelProducts`] and lowered directly into DSL families by the refresh composer; they are
//! not scalar Program output ports. Noise-only Programs retain the scalar balanced-tree model.

use super::{Dio2Error, Dio2Parameters, StreamingFunction, function::StepWires};
use mxx_exponent_lut::{
    program::{
        AttributeKnowledge, DataId, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
        LutOutputRepresentation, LutTable, Operand, RhsId, WireId,
    },
    program_gadgets::{
        prf::{PrfKeyShape, PrfKeys, PrfOutput, SparseLwrPrf},
        prf_sampler::{PrfDistribution, PrfSampler},
    },
    rns_bgg::{RnsBggRoot, root_helper_input_name, root_row_input_name},
};
use mxx_ir_core::IntExpr;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Root tag of `ONEPRIV`, the private encoding of one under the payload secret.
pub const PRIVATE_TAG: &str = "dio2/one-private";
/// Root tag of the state.
pub const STATE_TAG: &str = "dio2/state";
/// Root tag of the PRF's input family: label column `0`, then `ONEPUB`.
pub const PUBLIC_TAG: &str = "dio2/label-0";
/// Root tag of the label columns `1..=m_v`.
pub const LABEL_TAG: &str = "dio2/label";
/// Clear row-major labels used by the symbolic public-product family.
pub const LABEL_VALUES_INPUT: &str = "dio2/current-labels";
/// Root tag of the public evaluator-bit monomial family, formed outside the common body.
pub const XBITS_TAG: &str = "dio2/xbits";
/// Runtime little-endian chunk value selecting the common label matrix's column block.
pub const CHUNK_INPUT: &str = "dio2/chunk";
/// Fixed declaration domain shared by every step and chunk.
pub const UNIFORM_SEED: u64 = 0;

/// Apply a public monomial identically to a root key block or an encoded root row.
pub fn public_monomial(value: mxx_dsl::Mat, exponent: mxx_dsl::Int) -> mxx_dsl::Mat {
    value.multiply_monomial(exponent)
}

/// Form the evaluator-bit family from the ONEPUB row or one of its key blocks. Both roles
/// use this same helper; the common Program never fixes a chunk's bits in its instructions.
pub fn xbits_rows(
    function: &StreamingFunction,
    one: mxx_dsl::Mat,
    chunk: mxx_dsl::Int,
) -> Result<mxx_dsl::Family<mxx_dsl::Mat>, mxx_dsl::DslError> {
    mxx_dsl::parallel(function.eval_chunk_bits, |bit| {
        let set = chunk.clone().bit(bit.clone())?.to_int();
        Ok(public_monomial(one.clone(), (function.x_monomial)(bit, set)?))
    })
}

/// A root-row group, without enumerating the members of either label family.
#[derive(Clone, Debug)]
pub enum RowGroup {
    Scalar(Target),
    Public,
    Labels,
    Helpers(Vec<usize>),
}

impl RowGroup {
    /// Family cardinality, or `None` for a scalar root row.
    pub fn count(&self, plan: &StepPlan) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Public => Some(plan.label_rows + 1),
            Self::Labels => Some(plan.label_rows * plan.label_width),
            Self::Helpers(sigmas) => Some(sigmas.len()),
        }
    }

    /// The refresh target index of a member, in the original part ordering.
    pub fn target_index(&self, plan: &StepPlan, member: mxx_dsl::Int) -> mxx_dsl::Int {
        use mxx_dsl::Int;
        match self {
            Self::Scalar(target) => Int::constant(plan.target_index(target) as i64),
            Self::Public => mxx_dsl::select(
                member.clone().equal(plan.label_rows).to_int(),
                vec![member.mul(plan.label_width + 1).add(3usize), Int::constant(1)],
            )
            .expect("two integer candidates"),
            Self::Labels => member
                .clone()
                .div(plan.label_width)
                .mul(plan.label_width + 1)
                .add(member.rem(plan.label_width))
                .add(4usize),
            Self::Helpers(sigmas) => Int::evaluate(IntExpr::Select {
                selector: Box::new(member.expression().expect("a structural member")),
                branches: sigmas
                    .iter()
                    .map(|sigma| IntExpr::constant(plan.target_index(&Target::Helper(*sigma))))
                    .collect(),
            }),
        }
    }

    /// The root key tag, using symbolic family indices for labels.
    pub fn key_tag(&self, plan: &StepPlan, member: mxx_dsl::Int) -> mxx_dsl::HashTag {
        match self {
            Self::Scalar(target) => target.key_tag(plan).expect("a scalar root tag"),
            Self::Public => RnsBggRoot::input_tag(PUBLIC_TAG, Some(member)),
            Self::Labels => RnsBggRoot::input_tag(LABEL_TAG, Some(member)),
            Self::Helpers(sigmas) => {
                RnsBggRoot::helper_tag(mxx_dsl::Int::evaluate(IntExpr::Select {
                    selector: Box::new(member.expression().expect("a structural member")),
                    branches: sigmas.iter().copied().map(IntExpr::constant).collect(),
                }))
            }
        }
    }
}

/// Which refresh targets a noise-only Program materializes. Production and claim graphs always
/// use [`StepPlan::new`], which includes every target, with labels represented symbolically.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoiseTargets {
    /// Materialize and simulate every target.
    Full,
    /// Keep every non-label target, but only one maximal-leaf label product per chunk.
    ///
    /// Label inputs are identically bounded public monomials of CBD(1) error. The public-product
    /// rule is `E(left * right) = A E(left) + ||mu_left||_1 E(right)`, with `A >= 1` the
    /// public gadget-action gain and `||mu_left||_1 = 1`. Balanced products padded to two leaves
    /// have nondecreasing error envelopes and moment bounds as their leaf count grows: adding
    /// a leaf adds a nonnegative error term and never decreases an existing leaf's action
    /// depth. All products stay at level zero on the equal-modulus chain. Thus a column with
    /// maximal leaf count in any row attains the maximum label noise, including after the
    /// decoder's common re-targeting and mask action. This is only a noise-model reduction,
    /// not an executable replacement for the label matrix. Full sampler member counts, decoder
    /// event counts, and the evaluation audit retain the full matrix's multiplicities. Identical
    /// weight-one smudge digits share one representative row, re-targeted once per actual digit
    /// and summed with unknown covariance. Their audit events retain the exact digit count;
    /// this also avoids overflowing a PRF family's `members * bucket_width` at large `d`.
    Representative,
}

/// A refresh target of a step (spec §6.6): what it becomes in the next step's input.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum Target {
    /// `ONEPRIV`.
    Private,
    /// `ONEPUB`, the encoding of one under the next mask.
    Public,
    /// The state.
    State,
    /// Entry `(row, column)` of the next label matrix.
    Label { row: usize, column: usize },
    /// The LUT helper of automorphism `sigma`.
    Helper(usize),
    /// Block `index` of the next anchor hint.
    Anchor(usize),
}

impl Target {
    /// The root tag of the root object this target becomes, with its member; `None` for an
    /// anchor block.
    pub fn root(&self, plan: &StepPlan) -> Option<(&'static str, Option<usize>)> {
        Some(match *self {
            Target::Private => (PRIVATE_TAG, None),
            Target::Public => (PUBLIC_TAG, Some(plan.label_rows)),
            Target::State => (STATE_TAG, None),
            Target::Label { row, column: 0 } => (PUBLIC_TAG, Some(row)),
            Target::Label { row, column } => (LABEL_TAG, Some(row * plan.label_width + column - 1)),
            Target::Helper(_) | Target::Anchor(_) => return None,
        })
    }

    /// The hash tag of this target's root key (`A'` of its refresh, spec §5).
    pub fn key_tag(&self, plan: &StepPlan) -> Option<mxx_dsl::HashTag> {
        match self {
            Target::Helper(sigma) => {
                Some(RnsBggRoot::helper_tag(mxx_dsl::Int::constant(*sigma as i64)))
            }
            Target::Anchor(_) => None,
            _ => self.root(plan).map(|(tag, member)| {
                RnsBggRoot::input_tag(
                    tag,
                    member.map(|member| mxx_dsl::Int::constant(member as i64)),
                )
            }),
        }
    }
}

/// The shape of every step: the automorphisms whose helpers a step reads, the refresh targets,
/// and the sizes of the samplers.
#[derive(Clone, Debug)]
pub struct StepPlan {
    /// Production plans lower label products and targets directly as DSL families. Noise-only
    /// plans retain scalar products so the existing exact per-column model remains available.
    pub symbolic_labels: bool,
    /// Concrete selector retained only by the Rust noise model, never production Programs.
    noise_transition: Vec<Vec<bool>>,
    /// The automorphisms `K` of every LUT of every step (spec §5), ascending; `1` among them.
    pub automorphisms: Vec<usize>,
    /// The materialized refresh targets; production plans omit the symbolic label range. In
    /// full noise mode these are in part order. In representative mode
    /// `Label { row: 0, column: 0 }` denotes the maximal-leaf label class, not a fixed entry.
    pub targets: Vec<Target>,
    /// Noise-only target selection; full for every production/claim plan.
    pub noise_targets: NoiseTargets,
    /// The number of targets in the complete plan, including unmaterialized labels.
    pub full_target_count: usize,
    /// `w_0`.
    pub columns: usize,
    pub label_rows: usize,
    pub label_width: usize,
    pub ring_dimension: usize,
    pub smudge_digits: usize,
    /// The primes of `Q_0`: a refresh part has one decoder per prime.
    pub primes: usize,
    /// The LUT helper families of every step's Program: the branch automorphisms of each
    /// [`mxx_exponent_lut::program::LutTable::helper_family_key`], which a step input provides as
    /// one family of helper rows per port.
    pub helper_families: BTreeMap<String, Vec<usize>>,
}

impl StepPlan {
    /// The production plan of `function`, with a symbolic range for all label targets.
    pub fn new(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, NoiseTargets::Full, true)
    }

    /// Build a noise-only plan with explicit target selection. Unlike `targets.len()`,
    /// [`Self::full_target_count`] and the full sampler-size methods describe the full plan.
    /// Do not pass a representative plan to obfuscation, evaluation, or claim export.
    pub fn for_noise(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        noise_targets: NoiseTargets,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, noise_targets, false)
    }

    fn build(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        noise_targets: NoiseTargets,
        symbolic_labels: bool,
    ) -> Result<Self, Dio2Error> {
        let columns = parameters.columns()?;
        let mut plan = Self {
            symbolic_labels,
            noise_transition: label_transition(parameters, &vec![false; function.eval_chunk_bits]),
            automorphisms: Vec::new(),
            targets: Vec::new(),
            noise_targets,
            full_target_count: 0,
            columns,
            label_rows: parameters.label_rows(),
            label_width: parameters.label_width(),
            ring_dimension: parameters.ring_dimension as usize,
            smudge_digits: parameters.smudge_digits,
            primes: parameters.primes.len(),
            helper_families: BTreeMap::new(),
        };
        // The common body has just two kinds: refreshing and final.
        let mut automorphisms = BTreeSet::new();
        let probe_shape = |role: &str, members| {
            parameters
                .prf
                .declared_shape(0, role, members)
                .map_err(|_| Dio2Error::Parameters("the PRF keys do not fit"))
        };
        let (n, d) = (plan.ring_dimension, plan.smudge_digits);
        for blocks in [1, 0] {
            let smudge_columns = if blocks == 0 { 1 } else { blocks * plan.primes * columns };
            let digits = match noise_targets {
                NoiseTargets::Full => d,
                NoiseTargets::Representative => 1,
            };
            let keys = ChainKeys {
                sigma: (blocks > 0).then(|| probe_shape("sigma", n)).transpose()?,
                cbd: (blocks > 0).then(|| probe_shape("cbd", blocks * columns * n)).transpose()?,
                smudge: probe_shape("smudge", smudge_columns * n * digits)?,
            };
            let probe = ChainProgram::build(parameters, function, &plan, &keys, blocks)?;
            for table in probe.program.encryption_inputs().luts() {
                let branches = table
                    .branch_automorphisms(plan.ring_dimension)
                    .map_err(|_| Dio2Error::Parameters("a LUT does not fit the ring"))?;
                if table.automorphism_index().is_none() {
                    automorphisms.extend(branches.iter().copied());
                }
                plan.helper_families.insert(table.helper_family_key(), branches);
            }
        }
        plan.automorphisms = automorphisms.into_iter().collect();
        // The σ sampler's automorphism tables, which the probes (built before the automorphisms
        // were known) do not have.
        for sigma in plan.automorphisms.iter().copied().filter(|sigma| *sigma != 1) {
            let table = LutTable::automorphism(sigma, LutOutputRepresentation::Coefficient)?;
            let branches = table
                .branch_automorphisms(plan.ring_dimension)
                .map_err(|_| Dio2Error::Parameters("a LUT does not fit the ring"))?;
            plan.helper_families.insert(table.helper_family_key(), branches);
        }
        let anchors = super::anchor_columns(columns).div_ceil(2 * columns);
        plan.full_target_count =
            3 + plan.label_rows * (plan.label_width + 1) + plan.automorphisms.len() + anchors;
        let labels = if symbolic_labels {
            0
        } else {
            match noise_targets {
                NoiseTargets::Full => plan.label_rows * (plan.label_width + 1),
                NoiseTargets::Representative => 1,
            }
        };
        plan.targets = [Target::Private, Target::Public, Target::State]
            .into_iter()
            .chain((0..labels).map(|index| Target::Label {
                row: index / (plan.label_width + 1),
                column: index % (plan.label_width + 1),
            }))
            .chain(plan.automorphisms.iter().map(|sigma| Target::Helper(*sigma)))
            .chain((0..anchors).map(Target::Anchor))
            .collect();
        Ok(plan)
    }

    /// Select a concrete label transition for the noise-only scalar product model.
    pub(crate) fn noise_for_chunk(&self, parameters: &Dio2Parameters, x: &[bool]) -> Self {
        assert!(!self.symbolic_labels, "production Programs never specialize on chunks");
        Self { noise_transition: label_transition(parameters, x), ..self.clone() }
    }

    /// Original target ordering: private, public, state, all row-major labels, helpers, anchor.
    pub fn target_index(&self, target: &Target) -> usize {
        match target {
            Target::Private => 0,
            Target::Public => 1,
            Target::State => 2,
            Target::Label { row, column } => 3 + row * (self.label_width + 1) + column,
            Target::Helper(sigma) => {
                3 + self.label_rows * (self.label_width + 1) +
                    self.automorphisms
                        .iter()
                        .position(|other| other == sigma)
                        .expect("a planned helper")
            }
            Target::Anchor(block) => {
                3 + self.label_rows * (self.label_width + 1) + self.automorphisms.len() + block
            }
        }
    }

    /// Root inputs represented by fixed-size group descriptors, not one host value per label.
    pub fn input_groups(&self) -> Vec<(String, usize, RowGroup)> {
        let mut groups = Vec::new();
        for port in 0..2 {
            for (tag, group) in [
                (PRIVATE_TAG, RowGroup::Scalar(Target::Private)),
                (STATE_TAG, RowGroup::Scalar(Target::State)),
                (PUBLIC_TAG, RowGroup::Public),
                (LABEL_TAG, RowGroup::Labels),
            ] {
                groups.push((root_row_input_name(tag, port), port, group));
            }
            for (key, sigmas) in &self.helper_families {
                groups.push((
                    root_helper_input_name(key, port),
                    port,
                    RowGroup::Helpers(sigmas.clone()),
                ));
            }
        }
        groups
    }

    /// The full number of refresh parts' column blocks: two per target, even in representative
    /// mode. PRF families must retain their exact structural multiplicities for noise bounds.
    pub fn blocks(&self) -> usize {
        2 * self.full_target_count
    }

    /// The members of the CBD sampler: `w_0 n` per refresh part.
    pub fn cbd_members(&self) -> usize {
        self.blocks() * self.columns * self.ring_dimension
    }

    /// The smudges of a refreshing step: one per part and prime, part `p` of prime `c` at
    /// `p |P| + c`.
    pub fn smudges(&self) -> usize {
        self.blocks() * self.primes
    }

    /// The members of the smudge PRF: `w_0 n d` per smudge.
    pub fn smudge_members(&self) -> usize {
        self.smudges()
            .checked_mul(self.columns)
            .and_then(|members| members.checked_mul(self.ring_dimension))
            .and_then(|members| members.checked_mul(self.smudge_digits))
            .expect("the full smudge family exceeds addressable member counts")
    }

    /// The members of the output decoder's smudge PRF: `n d` for its one column.
    pub fn output_smudge_members(&self) -> usize {
        self.ring_dimension * self.smudge_digits
    }

    /// The materialized smudge PRF's members, for a refreshing or last step. Representative
    /// mode models one weight-one digit, not all `d` identically bounded digits; its noise is
    /// re-targeted with every actual digit weight. Use [`Self::smudge_members`] and
    /// [`Self::output_smudge_members`] for full physical counts, never this reduced shape.
    pub fn noise_smudge_members(&self, last: bool) -> usize {
        match self.noise_targets {
            NoiseTargets::Full => {
                if last {
                    self.output_smudge_members()
                } else {
                    self.smudge_members()
                }
            }
            NoiseTargets::Representative => {
                if last {
                    self.ring_dimension
                } else {
                    self.smudges() * self.columns * self.ring_dimension
                }
            }
        }
    }
}

/// The uniform declared PRF key shapes, shared by every step and chunk of one Program kind.
/// Coordinate tables and selectors are data; the declaration's domain is not a key-set digest.
pub struct ChainKeys {
    /// The σ sampler's `n` keys; `None` for the last step.
    pub sigma: Option<PrfKeyShape>,
    /// The CBD sampler's keys ([`StepPlan::cbd_members`]); `None` for the last step.
    pub cbd: Option<PrfKeyShape>,
    /// The smudge PRF's keys ([`StepPlan::smudge_members`]); of the output decoder (`n d` keys)
    /// for the last step.
    pub smudge: PrfKeyShape,
}

/// The output positions of a [`ChainProgram`].
#[derive(Clone, Debug)]
pub struct ChainOutputs {
    pub private: Option<usize>,
    pub public: Option<usize>,
    pub state: usize,
    pub sigma: Option<usize>,
    /// The σ sampler's automorphisms `Aut_k`, `k` in `K` but `1`.
    pub automorphed: BTreeMap<usize, usize>,
    /// The CBD noise of every refresh part: a row of `w_0` blocks per part, part `p` at blocks
    /// `p w_0 ..` (none for the last step).
    pub cbd: Option<usize>,
    /// The weight-one digit rows of every refresh part and prime ([`StepPlan::smudges`]), in
    /// digit order: a row of `w_0` columns per smudge; of the output decoder: one column.
    /// The refresh re-targets digit `r` to the scalar form `P^r / 2` before adding the rows.
    /// All digits share one output row, with digit `r` occupying columns
    /// `r * smudge_columns .. (r + 1) * smudge_columns`. Noise-representative mode
    /// stores just one digit, whose bound is used with every actual digit weight.
    pub smudge: usize,
    /// Scalar noise-model label products: entry `(r, c)` at `r (m_v + 1) + c` in full noise
    /// mode, only the maximal-leaf product at index zero in representative mode. Production
    /// Programs leave this empty and use [`ChainProgram::label_products`] instead.
    pub labels: Vec<usize>,
}

/// The uniform Program of every step/chunk of one kind, with fixed root input tags.
pub struct ChainProgram {
    /// Fixed public transition for DSL label families; absent in noise-only and final Programs.
    pub label_products: Option<LabelProducts>,
    pub program: ExponentLutProgram,
    pub root_inputs: BTreeMap<WireId, String>,
    pub outputs: ChainOutputs,
    /// Scalar public products omitted by noise-representative construction. Each omitted
    /// product contributes exactly `2 n w_0` physical events to the evaluation audit; all
    /// public label products are level-zero, non-coefficient outputs on this chain.
    pub omitted_label_products: usize,
    /// Builder-owned wire intervals of the single representative smudge PRF and its placed
    /// row. Noise auditing scales their operation counts by `d`, excluding input bindings.
    pub(crate) representative_smudge_ranges: Vec<std::ops::RangeInclusive<WireId>>,
    pub sigma: Option<PrfSampler>,
    /// The CBD sampler.
    pub cbd: Option<PrfSampler>,
    /// The smudge PRF.
    pub smudge: SparseLwrPrf,
    /// The obfuscator's bits.
    pub y: RhsId,
    /// Integer data family bound to the selected function row.
    pub data: DataId,
}

impl ChainProgram {
    /// The Program of a step followed by a refresh, whose `plan.blocks()` part blocks each
    /// decode a row of `w_0` columns.
    pub fn refreshing(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        keys: &ChainKeys,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, plan, keys, plan.blocks())
    }

    /// The Program of the last step, whose state the output decoder reads.
    pub fn last(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        keys: &ChainKeys,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, plan, keys, 0)
    }

    /// The Program with `blocks` refresh part blocks; none for the last step, whose output
    /// decoder decodes one column.
    fn build(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        keys: &ChainKeys,
        blocks: usize,
    ) -> Result<Self, Dio2Error> {
        let last = blocks == 0;
        let profile = parameters.profile()?;
        let mut builder = ExponentLutProgramBuilder::new();
        let mut root_inputs = BTreeMap::new();
        let label_products = if !last && plan.symbolic_labels && plan.full_target_count != 0 {
            Some(LabelProducts::new(parameters, function.eval_chunk_bits))
        } else {
            None
        };
        let root_prf = |prf: &SparseLwrPrf, root_inputs: &mut BTreeMap<WireId, String>| {
            root_inputs.insert(prf.initial_input(), PRIVATE_TAG.to_owned());
            root_inputs.insert(prf.input_vector(), PUBLIC_TAG.to_owned());
        };
        // The smudge PRF reads the label column 0, so it also provides the one-encoding.
        let smudge_columns = if last { 1 } else { blocks * plan.primes * plan.columns };
        if keys.cbd.is_some() == last {
            return Err(Dio2Error::Parameters("a refreshing step, and only it, has CBD keys"));
        }
        let smudge = SparseLwrPrf::build(
            &mut builder,
            profile,
            PrfKeys::Shape(keys.smudge.clone().with_data_tables()),
            LutOutputRepresentation::Coefficient,
        )?;
        root_prf(&smudge, &mut root_inputs);
        let public = Operand::Member {
            family: smudge.input_vector(),
            index: IntExpr::constant(plan.label_rows),
        };
        let state = builder.input(AttributeKnowledge::PrivatePayload)?;
        root_inputs.insert(state, STATE_TAG.to_owned());
        let x = builder.input_family(
            IntExpr::constant(function.eval_chunk_bits),
            AttributeKnowledge::PublicPayload,
        )?;
        root_inputs.insert(x, XBITS_TAG.to_owned());
        let y = builder.rhs_family(IntExpr::constant(function.obf_chunk_bits))?;
        let data = builder.data_family(function.data.width, Some(function.data.bound))?;
        let new_state = function.body.build(&mut builder, &StepWires { state, x, y, data })?;
        let mut outputs = ChainOutputs {
            private: None,
            public: None,
            state: 0,
            sigma: None,
            automorphed: BTreeMap::new(),
            cbd: None,
            smudge: 0,
            labels: Vec::new(),
        };
        let mut wires = vec![Operand::Scalar(new_state)];
        let mut omitted_label_products = 0;
        let mut representative_smudge_ranges = Vec::new();
        match plan.noise_targets {
            NoiseTargets::Full => {
                let (row, _) = placed_row(
                    &mut builder,
                    &smudge,
                    plan,
                    smudge_columns,
                    Some(plan.smudge_digits),
                )?;
                outputs.smudge = wires.len();
                wires.push(Operand::Scalar(row));
            }
            NoiseTargets::Representative => {
                let (row, family) = placed_row(&mut builder, &smudge, plan, smudge_columns, None)?;
                representative_smudge_ranges.extend([smudge.initial_input()..=state, family..=row]);
                outputs.smudge = wires.len();
                wires.push(Operand::Scalar(row));
            }
        }
        let (mut sigma_sampler, mut cbd_sampler) = (None, None);
        if !last {
            let sigma_shape = keys
                .sigma
                .clone()
                .ok_or(Dio2Error::Parameters("a refreshing step needs σ keys"))?;
            let (sampler, packed) = PrfSampler::sign_polynomial(
                &mut builder,
                profile,
                PrfKeys::Shape(sigma_shape.with_data_tables()),
                None,
            )?;
            root_prf(sampler.prf(), &mut root_inputs);
            outputs.private = Some(wires.len());
            wires.push(Operand::Scalar(sampler.prf().initial_input()));
            outputs.public = Some(wires.len());
            wires.push(Operand::Member {
                family: sampler.prf().input_vector(),
                index: IntExpr::constant(plan.label_rows),
            });
            outputs.sigma = Some(wires.len());
            wires.push(Operand::Scalar(packed));
            for sigma in plan.automorphisms.iter().copied().filter(|sigma| *sigma != 1) {
                let table = LutTable::automorphism(sigma, LutOutputRepresentation::Coefficient)?;
                let lut = builder.register_lut(table)?;
                let automorphed = builder.lut(Operand::Scalar(packed), lut)?;
                outputs.automorphed.insert(sigma, wires.len());
                wires.push(Operand::Scalar(automorphed));
            }
            sigma_sampler = Some(sampler);
            if let Some(shape) = &keys.cbd {
                let cbd = PrfSampler::build(
                    &mut builder,
                    profile,
                    PrfKeys::Shape(shape.clone().with_data_tables()),
                    PrfDistribution::Cbd { eta: 1 },
                )?;
                root_prf(cbd.prf(), &mut root_inputs);
                let (row, _) =
                    placed_row(&mut builder, cbd.prf(), plan, blocks * plan.columns, None)?;
                outputs.cbd = Some(wires.len());
                wires.push(Operand::Scalar(row));
                cbd_sampler = Some(cbd);
            }
            // The label columns `1..=m_v` and the products of the next label matrix.
            if !plan.symbolic_labels {
                let labels = builder.input_family(
                    IntExpr::constant(plan.label_rows * plan.label_width),
                    AttributeKnowledge::PublicPayload,
                )?;
                root_inputs.insert(labels, LABEL_TAG.to_owned());
                let transition = &plan.noise_transition;
                let (rows, selected_columns) = match plan.noise_targets {
                    NoiseTargets::Full => {
                        (plan.label_rows, (0..=plan.label_width).collect::<Vec<_>>())
                    }
                    NoiseTargets::Representative => {
                        let leaves = (0..=plan.label_width)
                            .map(|column| transition.iter().filter(|row| row[column]).count())
                            .collect::<Vec<_>>();
                        let column =
                            (0..leaves.len()).max_by_key(|column| leaves[*column]).unwrap();
                        omitted_label_products = plan.label_rows *
                            leaves.iter().map(|count| (*count).max(2) - 1).sum::<usize>() -
                            (leaves[column].max(2) - 1);
                        (1, vec![column])
                    }
                };
                for row in 0..rows {
                    for column in selected_columns.iter().copied() {
                        let leaves = (0..plan.label_width)
                            .filter(|k| transition[*k][column])
                            .map(|k| Operand::Member {
                                family: labels,
                                index: IntExpr::constant(row * plan.label_width + k),
                            })
                            .collect::<Vec<_>>();
                        let product = public_product(&mut builder, leaves, &public)?;
                        outputs.labels.push(wires.len());
                        wires.push(Operand::Scalar(product));
                    }
                }
            }
        }
        let program = builder.finish(wires)?;
        Ok(Self {
            label_products,
            program,
            root_inputs,
            outputs,
            omitted_label_products,
            representative_smudge_ranges,
            sigma: sigma_sampler,
            cbd: cbd_sampler,
            smudge,
            y,
            data,
        })
    }
}

/// The row of `columns` columns (a block per column) of the samples of `prf`'s member family,
/// and its first placement wire, as `FreshNoiseProgram` and the
/// decoder's smudge Program place them. The family-to-row interval identifies the operation
/// event counts that must be repeated for identical noise-representative smudge digits.
///
/// - CBD noise (`digits` is `None`): member `column n + k` lands at `X^k` of block `column`.
/// - All smudge digits (`digits` is `Some(d)`): member `(column n + k) d + r` lands at `X^k` of
///   column `r * columns + column` with weight one. A single indexed body covers all digits. The
///   refresh applies its full weight `P^r / 2` as a BigInt scalar re-targeting (see
///   `mxx_exponent_lut::decoder::decoding`), so no power of `P` must fit a small-constant
///   instruction or its coefficient payload bound.
fn placed_row(
    builder: &mut ExponentLutProgramBuilder,
    prf: &SparseLwrPrf,
    plan: &StepPlan,
    columns: usize,
    digits: Option<usize>,
) -> Result<(WireId, WireId), Dio2Error> {
    let PrfOutput::Family(family) = prf.output() else {
        return Err(Dio2Error::Parameters("a sampler needs several members"));
    };
    let n = plan.ring_dimension;
    let digits = digits.unwrap_or(1);
    let range = FamilyRange::full(IntExpr::constant(columns * n * digits))?;
    let mut first = None;
    let placed = builder.parallel(range, |builder, index| {
        let member = IntExpr::Add(
            Box::new(IntExpr::Mul(
                Box::new(IntExpr::Rem(
                    Box::new(index.clone()),
                    Box::new(IntExpr::constant(columns * n)),
                )),
                Box::new(IntExpr::constant(digits)),
            )),
            Box::new(IntExpr::FloorDiv(
                Box::new(index.clone()),
                Box::new(IntExpr::constant(columns * n)),
            )),
        )
        .canonicalize();
        let coefficient =
            IntExpr::Rem(Box::new(index), Box::new(IntExpr::constant(n))).canonicalize();
        let wire = builder.mul_small_constant(
            Operand::Member { family, index: member },
            vec![1],
            coefficient,
        )?;
        first = Some(wire);
        Ok(Operand::Scalar(wire))
    })?;
    Ok((
        builder.sum_columns(placed, columns * digits)?,
        first.expect("a placement emits one body wire"),
    ))
}

/// The product of the public encodings `leaves` by a balanced tree of public products, padded
/// with the one-encoding `one` to at least two leaves.
fn public_product(
    builder: &mut ExponentLutProgramBuilder,
    mut leaves: Vec<Operand<WireId>>,
    one: &Operand<WireId>,
) -> Result<WireId, Dio2Error> {
    while leaves.len() < 2 {
        leaves.push(one.clone());
    }
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
        for pair in leaves.chunks(2) {
            next.push(match pair {
                [left, right] => Operand::Scalar(builder.mul_public(left.clone(), right.clone())?),
                [single] => single.clone(),
                _ => unreachable!("chunks of two"),
            });
        }
        leaves = next;
    }
    match leaves.pop() {
        Some(Operand::Scalar(wire)) => Ok(wire),
        _ => Err(Dio2Error::Parameters("a label product needs a product")),
    }
}

/// A fixed-seed selector recipe for the irregular leaf sets of `D_x`. Each computed column
/// packs its selected row indices in fixed-width binary fields. Reading one field uses
/// `index_bits` bit tests; the DSL records one recipe, not an expanded table or a matrix node
/// per selected leaf or label.
pub struct LabelProducts {
    seed: [u8; 32],
    width: usize,
    chunks: usize,
    /// The leaf slots of every column: the label rows `m_v` (at least two), a bound on every
    /// column's selected-row count that holds by construction, so the forest multiplies exactly
    /// the selected rows without a bound that depends on the hashed selector table.
    pub max_leaves: usize,
    /// Binary field width of an input row index.
    pub index_bits: usize,
}

impl LabelProducts {
    /// Read a selected input row from its packed column, using structural bit positions.
    pub fn index(
        &self,
        columns: &mxx_dsl::Family<mxx_dsl::Int>,
        chunk: mxx_dsl::Int,
        column: mxx_dsl::Int,
        leaf: mxx_dsl::Int,
    ) -> Result<mxx_dsl::Int, mxx_dsl::DslError> {
        let packed = columns.at(chunk.mul(self.width + 1).add(column));
        let mut index = mxx_dsl::Int::constant(0);
        for bit in 0..self.index_bits {
            index = index.add(
                packed
                    .clone()
                    .bit(leaf.clone().mul(self.index_bits).add(bit))?
                    .to_int()
                    .mul(1usize << bit),
            );
        }
        Ok(index)
    }

    /// The clear transition over a row-major integer family. This uses the same leaf selector
    /// table as the encoding forest, but sums exponents modulo the PRF input modulus.
    pub fn next(
        &self,
        parameters: &Dio2Parameters,
        labels: &mxx_dsl::Family<mxx_dsl::Int>,
        chunk: mxx_dsl::Int,
    ) -> Result<mxx_dsl::Family<mxx_dsl::Int>, mxx_dsl::DslError> {
        let width = parameters.label_width();
        let (packed_columns, counts) = self.columns();
        mxx_dsl::parallel(parameters.label_rows() * (width + 1), |member| {
            let row = member.clone().div(width + 1);
            let column = member.rem(width + 1);
            let count = counts.at(chunk.clone().mul(width + 1).add(column.clone()));
            let total =
                mxx_dsl::iterate(self.max_leaves, mxx_dsl::Int::constant(0), |leaf, total| {
                    let index =
                        self.index(&packed_columns, chunk.clone(), column.clone(), leaf.clone())?;
                    let value = labels.at(row.clone().mul(width + 1).add(index).add(1usize));
                    Ok(total.add(mxx_dsl::select(
                        leaf.less(count.clone()).to_int(),
                        vec![mxx_dsl::Int::constant(0), value],
                    )?))
                })?;
            Ok(total.rem(parameters.label_modulus()))
        })
    }

    /// Retain the fixed seed, without embedding the selector table.
    pub fn new(parameters: &Dio2Parameters, eval_chunk_bits: usize) -> Self {
        let width = parameters.label_width();
        let index_bits = (usize::BITS - (width - 1).leading_zeros()) as usize;
        Self {
            seed: label_transition_seed(),
            width,
            chunks: 1usize.checked_shl(eval_chunk_bits as u32).expect("addressable chunks"),
            max_leaves: width.max(2),
            index_bits,
        }
    }

    /// One shared public computation of all packed columns and their unpadded leaf counts.
    pub fn columns(&self) -> (mxx_dsl::Family<mxx_dsl::Int>, mxx_dsl::Family<mxx_dsl::Int>) {
        mxx_dsl::DslContext::new("dio2/label-columns").sha256_bit_columns(
            self.seed,
            self.width,
            self.chunks * (self.width + 1),
            self.index_bits,
        )
    }
}

/// The label transition `D_x` (spec §3): column block `val(x)` of the common row-major
/// `m_v` by `2^c (m_v + 1)` bit matrix, shared by every step and selected little-endian.
pub fn label_transition(parameters: &Dio2Parameters, x: &[bool]) -> Vec<Vec<bool>> {
    let width = parameters.label_width();
    let seed = label_transition_seed();
    let columns = (1usize << x.len()) * (width + 1);
    let base = super::function::chunk_value(x) * (width + 1);
    let mut cached_index = usize::MAX;
    let mut block = [0u8; 32];
    let mut bit = |index: usize| {
        let block_index = index / 256;
        if block_index != cached_index {
            let mut digest = Sha256::new();
            digest.update(seed);
            digest.update((block_index as u64).to_le_bytes());
            block = digest.finalize().into();
            cached_index = block_index;
        }
        let offset = index % 256;
        block[offset / 8] >> (offset % 8) & 1 == 1
    };
    (0..width).map(|k| (0..=width).map(|c| bit(k * columns + base + c)).collect()).collect()
}

/// Fixed domain of the single row-major label bit matrix; chunks select column blocks.
fn label_transition_seed() -> [u8; 32] {
    Sha256::digest(b"mxx-io/dio2/label-transition/column-blocks/v1").into()
}

/// The next label matrix `A_(i+1) = A_i[:, 1..] D_x mod Q` in the clear.
pub fn next_labels(
    parameters: &Dio2Parameters,
    labels: &[Vec<usize>],
    x: &[bool],
) -> Vec<Vec<usize>> {
    let q = parameters.label_modulus();
    let transition = label_transition(parameters, x);
    labels
        .iter()
        .map(|row| {
            (0..transition[0].len())
                .map(|column| {
                    (0..transition.len())
                        .filter(|k| transition[*k][column])
                        .map(|k| row[1 + k])
                        .sum::<usize>() %
                        q
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;
    use mxx_exponent_lut::rns_bgg::RnsBggDerivedWorkload;

    /// All step rows and evaluator chunks share one Program identity per terminal kind.
    #[test]
    fn chain_program_id_is_independent_of_step_and_chunk() {
        let (parameters, function) = toy();
        let plan = StepPlan::new(&parameters, &function).unwrap();
        let shape = |role: &str, members| {
            parameters.prf.declared_shape(UNIFORM_SEED, role, members).unwrap()
        };
        for last in [false, true] {
            let keys = ChainKeys {
                sigma: (!last).then(|| shape("sigma", plan.ring_dimension)),
                cbd: (!last).then(|| shape("cbd", plan.cbd_members())),
                smudge: shape("smudge", plan.noise_smudge_members(last)),
            };
            let mut identity = None;
            for step in 0..function.stream_length() {
                let mut row_function = function.clone();
                row_function.data = super::super::StepData::new(
                    function.data.width,
                    function.data.bound,
                    vec![function.step_row(step).to_vec()],
                )
                .unwrap();
                for chunk in 0..1usize << function.eval_chunk_bits {
                    let program = if last {
                        ChainProgram::last(&parameters, &row_function, &plan, &keys)
                    } else {
                        ChainProgram::refreshing(&parameters, &row_function, &plan, &keys)
                    }
                    .unwrap();
                    let data = mxx_exponent_lut::program::ProgramData::new()
                        .with(program.data, function.step_row(step).to_vec());
                    assert_eq!(data.values(program.data).unwrap(), function.step_row(step));
                    assert!(chunk < 1usize << function.eval_chunk_bits);
                    if let Some(expected) = &identity {
                        assert_eq!(program.program.id(), *expected);
                    } else {
                        identity = Some(program.program.id());
                    }
                    let xbits = program
                        .root_inputs
                        .iter()
                        .find_map(|(wire, tag)| (tag == XBITS_TAG).then_some(*wire))
                        .unwrap();
                    assert!(matches!(
                        program.program.inputs()[&xbits].shape(),
                        mxx_exponent_lut::program::InputBinding::Family(_)
                    ));
                }
            }
        }
    }

    /// The shared root-row/key operation produces the clear monomials for random chunks.
    #[test]
    fn xbits_rows_decode_to_monomials() {
        use mxx_backends::{
            MemoryArtifactStore,
            authority::CpuExecution,
            backend::poly::cpu_backend,
            poly::{
                Poly, PolyParams,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use rand::Rng;
        let (_, function) = toy();
        let bits =
            std::env::var("MXX_IO_DIO2_XBITS").map_or(3, |value| value.parse::<usize>().unwrap());
        assert!((1..usize::BITS as usize).contains(&bits));
        let function = StreamingFunction::additive(1, bits, 1, function.state_layout);
        let ring_dimension =
            1u32.checked_shl(bits as u32 + function.state_layout.stride_log2).unwrap();
        let crt_bits = std::env::var("MXX_IO_DIO2_XBITS_CRT_BITS")
            .map_or(17, |value| value.parse::<usize>().unwrap());
        let level = DCRTPolyParams::new(ring_dimension, 2, crt_bits, 4, None, None);
        let ring = mxx_dsl::Ring::from_crt_moduli(
            level.to_crt().0.into_iter().map(Into::into).collect(),
            ring_dimension,
        );
        let context = mxx_dsl::DslContext::new("xbits-test");
        let chunk = context.input::<mxx_dsl::Int>(CHUNK_INPUT, mxx_dsl::IntType).unwrap();
        let rows = xbits_rows(&function, ring.identity(1), chunk).unwrap();
        let graph = context.output("rows", rows).unwrap().build().unwrap();
        let mut execution = CpuExecution::new(cpu_backend([level.clone()]));
        let maximum = (1usize << bits) - 1;
        for chunk in [0, maximum, rand::rng().random_range(0..=maximum)] {
            let result = execute_graph(
                StageGraph::new(
                    mxx_dsl::BuiltGraph { graph: graph.graph.clone() },
                    BTreeMap::from([(
                        CHUNK_INPUT.to_owned(),
                        mxx_backends::RuntimeValue::Int(chunk.into()),
                    )]),
                    BTreeMap::new(),
                ),
                &mxx_ir_core::ParamEnv::default(),
                &mut execution,
                &mut MemoryArtifactStore::default(),
                |_| Ok(()),
            )
            .unwrap();
            let mxx_backends::RuntimeValue::IndexedFamily { values, .. } = &result.outputs["rows"]
            else {
                panic!("matrix family")
            };
            assert_eq!(values.len(), bits);
            for (bit, value) in values.iter().enumerate() {
                let row = <CpuExecution as StageExecution<MemoryArtifactStore>>::download_matrix(
                    &execution, value,
                )
                .unwrap();
                let expected = DCRTPoly::const_rotate_poly(
                    &level,
                    ((chunk >> bit) & 1) * (1usize << bit) * function.state_layout.stride(),
                );
                assert_eq!(mxx_backends::matrix::PolyMatrix::entry(&row, 0, 0), expected);
            }
        }
    }

    /// All digit counts use the same packed Program output and output identifiers.
    #[test]
    fn test_smudge_digits_share_one_program_output() {
        let mut expected = None;
        for digits in [1, 3, 60, 306] {
            let (mut parameters, function) = toy();
            parameters.smudge_digits = digits;
            let plan = StepPlan::new(&parameters, &function).unwrap();
            let keys = ChainKeys {
                sigma: None,
                cbd: None,
                smudge: parameters
                    .prf
                    .declared_shape(0, "smudge", plan.output_smudge_members())
                    .unwrap(),
            };
            let program = ChainProgram::last(&parameters, &function, &plan, &keys).unwrap();
            let outputs = (program.outputs.smudge, program.program.outputs().to_vec());
            if let Some(expected) = &expected {
                assert_eq!(&outputs, expected);
            } else {
                expected = Some(outputs);
            }
        }
    }

    #[test]
    fn oversized_prf_selector_family_is_symbolic() {
        let (parameters, function) = toy();
        let plan = StepPlan::new(&parameters, &function).unwrap();
        let keys = ChainKeys {
            sigma: None,
            cbd: None,
            smudge: parameters.prf.declared_shape(0, "oversized", usize::MAX).unwrap(),
        };
        let program = ChainProgram::last(&parameters, &function, &plan, &keys)
            .expect("symbolic PRF selector table");
        assert!(
            program
                .program
                .data_families()
                .values()
                .any(|family| family.count() > usize::MAX as u128)
        );
    }

    /// The symbolic selector and integer transition agree with the existing clear evaluator,
    /// including arbitrary column leaf counts and multiple rows. Node cardinality is bounded
    /// independently of the number of label entries (only index bit width grows).
    #[test]
    fn symbolic_label_transition_matches_clear_labels() {
        use mxx_backends::{
            MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_dsl::DslContext;
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use num_bigint::BigInt;
        use rand::Rng;
        let mut node_counts = Vec::new();
        for height in [1, 2, 8] {
            let (mut parameters, _) = toy();
            parameters.prf.lwr_dimension = height;
            let mut execution = CpuExecution::new(cpu_backend(parameters.rings().unwrap()));
            for chunk in [false, true] {
                let labels = (0..height)
                    .map(|_| {
                        (0..=parameters.label_width())
                            .map(|_| rand::rng().random_range(0..parameters.label_modulus()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let products = LabelProducts::new(&parameters, 1);
                let graph = DslContext::new("dio2-symbolic-labels");
                let input =
                    graph.int_family_input("labels", height * (parameters.label_width() + 1));
                let graph = graph
                    .output(
                        "next",
                        products
                            .next(&parameters, &input, mxx_dsl::Int::constant(usize::from(chunk)))
                            .unwrap(),
                    )
                    .unwrap()
                    .build()
                    .unwrap();
                node_counts.push(
                    graph.graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>(),
                );
                let runtime = BTreeMap::from([(
                    "labels".to_owned(),
                    RuntimeValue::integer_values(
                        labels.iter().flatten().copied().map(BigInt::from).collect(),
                    ),
                )]);
                let output = execute_graph(
                    StageGraph::new(graph, runtime, BTreeMap::new()),
                    &mxx_ir_core::ParamEnv::default(),
                    &mut execution,
                    &mut MemoryArtifactStore::default(),
                    |_| Ok(()),
                )
                .unwrap();
                let actual =
                    <CpuExecution as StageExecution<MemoryArtifactStore>>::download_integers(
                        &execution,
                        &output.outputs["next"],
                    )
                    .unwrap();
                let expected = next_labels(&parameters, &labels, &[chunk])
                    .into_iter()
                    .flatten()
                    .map(BigInt::from)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
        }
        assert!(node_counts.iter().max().unwrap() < &(node_counts[0] * 2));
    }

    /// One unchanged graph selects random multi-bit chunks from one SHA-256 column recipe.
    #[test]
    fn chunk_label_columns_match_clear_transition() {
        use mxx_backends::{
            MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_dsl::DslContext;
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use rand::Rng;
        let bits = std::env::var("MXX_IO_DIO2_LABEL_CHUNK_BITS")
            .map_or(3usize, |value| value.parse().unwrap());
        let (parameters, _) = toy();
        let labels = (0..parameters.label_rows())
            .map(|_| {
                (0..=parameters.label_width())
                    .map(|_| rand::rng().random_range(0..parameters.label_modulus()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let graph = DslContext::new("chunk-label-columns");
        let chunk = graph.input::<mxx_dsl::Int>(CHUNK_INPUT, mxx_dsl::IntType).unwrap();
        let input = graph
            .int_family_input("labels", parameters.label_rows() * (parameters.label_width() + 1));
        let products = LabelProducts::new(&parameters, bits);
        let graph = graph
            .output("next", products.next(&parameters, &input, chunk).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let recipes = graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::Sha256BitColumns { .. })
            })
            .count();
        assert_eq!(recipes, 1);
        let mut execution = CpuExecution::new(cpu_backend(parameters.rings().unwrap()));
        for chunk in [0, (1usize << bits) - 1, rand::rng().random_range(0..1usize << bits)] {
            let runtime = BTreeMap::from([
                (
                    "labels".to_owned(),
                    RuntimeValue::integer_values(
                        labels.iter().flatten().copied().map(num_bigint::BigInt::from).collect(),
                    ),
                ),
                (CHUNK_INPUT.to_owned(), RuntimeValue::Int(chunk.into())),
            ]);
            let output = execute_graph(
                StageGraph::new(
                    mxx_dsl::BuiltGraph { graph: graph.graph.clone() },
                    runtime,
                    BTreeMap::new(),
                ),
                &mxx_ir_core::ParamEnv::default(),
                &mut execution,
                &mut MemoryArtifactStore::default(),
                |_| Ok(()),
            )
            .unwrap();
            let actual = <CpuExecution as StageExecution<MemoryArtifactStore>>::download_integers(
                &execution,
                &output.outputs["next"],
            )
            .unwrap();
            let x = (0..bits).map(|bit| (chunk >> bit) & 1 == 1).collect::<Vec<_>>();
            let expected = next_labels(&parameters, &labels, &x)
                .into_iter()
                .flatten()
                .map(num_bigint::BigInt::from)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    /// Column blocks share a single row-major hash expansion, including block boundaries.
    #[test]
    fn label_transition_preserves_hash_expansion() {
        let (mut parameters, _) = toy();
        parameters.prf.lwr_dimension = 8;
        let x = [true];
        let width = parameters.label_width();
        let seed = label_transition_seed();
        for (row, values) in label_transition(&parameters, &x).iter().enumerate() {
            for (column, value) in values.iter().enumerate() {
                let index = row * (2 * (width + 1)) + width + 1 + column;
                let mut digest = Sha256::new();
                digest.update(seed);
                digest.update(((index / 256) as u64).to_le_bytes());
                let block: [u8; 32] = digest.finalize().into();
                let offset = index % 256;
                assert_eq!(*value, block[offset / 8] >> (offset % 8) & 1 == 1);
            }
        }
    }

    /// The step Programs of the toy function build on declared keys, and every output's gadget
    /// level (the level after a private output's, a public output's own) is at most the
    /// decoding level, so the refresh can re-target every output to it.
    #[test]
    fn step_programs_fit_the_toy_chain_plan() {
        let (parameters, function) = toy();
        let plan = StepPlan::new(&parameters, &function).unwrap();
        let shape = |role: &str, members| parameters.prf.declared_shape(1, role, members).unwrap();
        let keys = ChainKeys {
            sigma: Some(shape("sigma", plan.ring_dimension)),
            cbd: Some(shape("cbd", plan.cbd_members())),
            smudge: shape("smudge", plan.smudge_members()),
        };
        let program = ChainProgram::refreshing(&parameters, &function, &plan, &keys).unwrap();
        assert!(program.outputs.labels.is_empty());
        assert!(program.label_products.is_some());
        assert!(plan.targets.iter().all(|target| !matches!(target, Target::Label { .. })));
        let workload = RnsBggDerivedWorkload::from_program(&program.program).unwrap();
        for output in program.program.outputs() {
            let wire = match output {
                Operand::Scalar(wire) | Operand::Member { family: wire, .. } => *wire,
                Operand::Selected { .. } => unreachable!("a step outputs fixed members"),
            };
            let level = workload.level(wire).unwrap();
            let gadget_level = match program.program.wire_knowledge(wire).unwrap() {
                AttributeKnowledge::PublicPayload => level,
                AttributeKnowledge::PrivatePayload => level + 1,
            };
            assert!(gadget_level <= parameters.decoding_level(), "{output:?} at level {level}");
        }
    }
}
