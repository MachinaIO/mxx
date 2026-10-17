//! The Program of one step (spec §6), its refresh targets, and the root objects its inputs read.
//! The chain never drops a modulus, so one Program serves the decoders of every prime.
//!
//! The step input `S_i` ([`StepRows`]) is a set of root objects (rows at `Q_0`, see
//! [`mxx_exponent_lut::rns_bgg::RnsBggRoot`]): `ONEPRIV` ([`PRIVATE_TAG`]), the state
//! ([`STATE_TAG`]), the label column `0` with the one-encoding `ONEPUB` as its last member
//! ([`PUBLIC_TAG`], the PRF's input family of `v + 1` members), the other label columns
//! ([`LABEL_TAG`], `v m_v` members, entry `(r, 1 + k)` at `r m_v + k`), the LUT helpers `HLP[k]`
//! of every automorphism of [`StepPlan::automorphisms`], and the anchor hint. Every Program input
//! reads one of them under its root tag, so the samplers' inputs (each PRF builds its own
//! accumulator and input family) all read the same rows. The evaluator-bit family
//! ([`XBITS_TAG`]) is derived from `ONEPUB` outside the common body: its rows by [`xbits_row`]
//! and its keys by the same public monomial ([`xbits_key`]).
//!
//! The Program of a step that is followed by another one ([`ChainProgram::refreshing`]) computes,
//! besides the common body `Pi` with its selected data row, the σ sampler (uniform signs)
//! and its automorphisms, the CBD fresh noise of
//! every refresh part (one PRF, one row of every part's blocks), the smudges of every part and
//! prime (one PRF, one row of every smudge's columns per digit), and the label products; the last
//! step's ([`ChainProgram::last`]) computes `Pi` and the smudge of the output decoder. The
//! Program does not grow with the number of parts. Production label products are described by
//! [`LabelProducts`] and lowered by the refresh composition ([`super::compose`]), each in the
//! part that decodes it; they are not scalar Program output ports. Noise-only Programs retain
//! scalar product chains of the same multiplication order. The production graphs lower the
//! Program with its CBD and smudge rows deferred ([`ChainProgram::deferred_outputs`]): those
//! rows are read one column block at a time and every Program family is lowered where its
//! members are read, while the noise model simulates the same Program eagerly.

use super::{
    Dio2Error, Dio2Parameters, StreamingFunction,
    function::StepWires,
    labels::{LabelProducts, label_transition},
};
use mxx_dsl::{Bytes, Family, Int, Mat};
use mxx_exponent_lut::{
    program::{
        AttributeKnowledge, DataId, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
        LutOutputRepresentation, LutTable, Operand, RhsId, WireId,
    },
    program_gadgets::{
        prf::{PrfKeyShape, PrfKeys, PrfOutput, SparseLwrPrf},
        prf_sampler::{PrfDistribution, PrfSampler},
    },
    rns_bgg::{RnsBggError, RnsBggMatrix, RnsBggRoot, RnsBggRow},
};
use mxx_ir_core::IntExpr;
use std::collections::{BTreeMap, BTreeSet};

/// Root tag of `ONEPRIV`, the private encoding of one under the payload secret.
pub const PRIVATE_TAG: &str = "dio2/one-private";
/// Root tag of the state.
pub const STATE_TAG: &str = "dio2/state";
/// Root tag of the PRF's input family: label column `0`, then `ONEPUB`.
pub const PUBLIC_TAG: &str = "dio2/label-0";
/// Root tag of the label columns `1..=m_v`.
pub const LABEL_TAG: &str = "dio2/label";
/// Root tag of the public evaluator-bit monomial family, formed outside the common body.
pub const XBITS_TAG: &str = "dio2/xbits";
/// Fixed declaration domain shared by every step and chunk.
pub const UNIFORM_SEED: u64 = 0;

/// Apply a public monomial identically to a root key block or an encoded root row.
pub fn public_monomial(value: Mat, exponent: Int) -> Mat {
    value.multiply_monomial(exponent)
}

/// The exponent `x_monomial(bit, x_bit)` of evaluator bit `bit` of the chunk `chunk`.
pub fn xbits_exponent(
    function: &StreamingFunction,
    bit: Int,
    chunk: Int,
) -> Result<Int, mxx_dsl::DslError> {
    let set = chunk.bit(bit.clone())?.to_int();
    (function.x_monomial)(bit, set)
}

/// Member `bit` of the evaluator-bit family of chunk `chunk`: the public monomial of the bit
/// times the `ONEPUB` row `one` (spec §6.1).
pub fn xbits_row(
    function: &StreamingFunction,
    one: &RnsBggRow,
    bit: Int,
    chunk: Int,
) -> Result<RnsBggRow, mxx_dsl::DslError> {
    let exponent = xbits_exponent(function, bit, chunk)?;
    Ok((public_monomial(one.0.clone(), exponent.clone()), public_monomial(one.1.clone(), exponent)))
}

/// The key of [`xbits_row`]: the same public monomial times every block of the `ONEPUB` key
/// `one`.
pub fn xbits_key(
    function: &StreamingFunction,
    one: &RnsBggMatrix,
    bit: Int,
    chunk: Int,
) -> Result<RnsBggMatrix, mxx_dsl::DslError> {
    let exponent = xbits_exponent(function, bit, chunk)?;
    let block = |matrix: &Mat| public_monomial(matrix.clone(), exponent.clone());
    Ok(((block(&one.0.0), block(&one.0.1)), (block(&one.1.0), block(&one.1.1))))
}

/// Form the evaluator-bit family from the `ONEPUB` row component or key block `one`.
pub fn xbits_rows(
    function: &StreamingFunction,
    one: Mat,
    chunk: Int,
) -> Result<Family<Mat>, mxx_dsl::DslError> {
    mxx_dsl::parallel(function.eval_chunk_bits, |bit| {
        Ok(public_monomial(one.clone(), xbits_exponent(function, bit, chunk.clone())?))
    })
}

/// The step input `S_i` (spec §5): every root object as a row of two `1 x w_0` components at
/// `Q_0`, and the anchor hint `b_i` (`1 x 2 (w_0 + 2)`).
#[derive(Clone)]
pub struct StepRows {
    /// `ONEPRIV`.
    pub private: RnsBggRow,
    /// The state.
    pub state: RnsBggRow,
    /// Label column `0` (member `r`), then `ONEPUB` (member `v`): `v + 1` members.
    pub public: Family<RnsBggRow>,
    /// Label columns `1..=m_v`: entry `(r, 1 + k)` at member `r m_v + k`.
    pub labels: Family<RnsBggRow>,
    /// `HLP[k]` of every automorphism `k` of [`StepPlan::automorphisms`], in its order.
    pub helpers: Family<RnsBggRow>,
    /// The anchor hint.
    pub hint: Mat,
}

/// [`StepRows`] as one graph value (a loop state or a published record).
pub type StepRowsValue =
    (RnsBggRow, RnsBggRow, Family<RnsBggRow>, Family<RnsBggRow>, Family<RnsBggRow>, Mat);

/// Initial material uses singleton families for private/state rows and the hint,
/// so the ordinary backend can publish each member and release its GPU storage
/// before generating refresh keys. Evaluation unwraps those singleton members;
/// the carried `StepRowsValue` and its cryptographic meaning are unchanged.
pub(crate) type InitialRowsValue = (
    Family<RnsBggRow>,
    Family<RnsBggRow>,
    Family<RnsBggRow>,
    Family<RnsBggRow>,
    Family<RnsBggRow>,
    Family<Mat>,
);

impl StepRows {
    pub fn into_value(self) -> StepRowsValue {
        (self.private, self.state, self.public, self.labels, self.helpers, self.hint)
    }

    pub fn from_value((private, state, public, labels, helpers, hint): StepRowsValue) -> Self {
        Self { private, state, public, labels, helpers, hint }
    }

    pub(crate) fn from_initial(
        (private, state, public, labels, helpers, hint): InitialRowsValue,
    ) -> Self {
        Self {
            private: private.at(0usize),
            state: state.at(0usize),
            public,
            labels,
            helpers,
            hint: hint.at(0usize),
        }
    }

    pub(crate) fn initial_schema(
        plan: &StepPlan,
        ring: &mxx_dsl::Ring,
    ) -> <InitialRowsValue as mxx_dsl::GraphValue>::Schema {
        let (private, state, public, labels, helpers, hint) = Self::schema(plan, ring);
        (
            mxx_dsl::FamilyType { element: private, count: 1usize.into(), member_slot: None },
            mxx_dsl::FamilyType { element: state, count: 1usize.into(), member_slot: None },
            public,
            labels,
            helpers,
            mxx_dsl::FamilyType { element: hint, count: 1usize.into(), member_slot: None },
        )
    }

    /// The schema of a step input of `plan` on the root ring `ring`.
    pub fn schema(
        plan: &StepPlan,
        ring: &mxx_dsl::Ring,
    ) -> <StepRowsValue as mxx_dsl::GraphValue>::Schema {
        let component = mxx_dsl::MatType(ring.matrix_type((1, plan.columns)));
        let row = (component.clone(), component);
        let family = |count: usize| mxx_dsl::FamilyType {
            element: row.clone(),
            count: IntExpr::constant(count),
            member_slot: None,
        };
        (
            row.clone(),
            row.clone(),
            family(plan.label_rows + 1),
            family(plan.label_rows * plan.label_width),
            family(plan.automorphisms.len()),
            mxx_dsl::MatType(ring.matrix_type((1, super::anchor_columns(plan.columns)))),
        )
    }
}

/// A root object of a step input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootObject {
    Private,
    State,
    /// Member `r < v` of label column 0, or `ONEPUB` (member `v`).
    Public,
    /// Member `r m_v + k` of label column `1 + k`.
    Label,
    /// The helper of automorphism `automorphisms[member]`.
    Helper,
}

impl StepPlan {
    /// The root key of member `member` (`None` for a scalar object) of the root object
    /// `object` under the root `root` and its hash key handle `hash_key`: the key a step input
    /// row of the object is encoded under, which every Program reading the object derives.
    pub fn root_key(
        &self,
        root: &RnsBggRoot,
        hash_key: Bytes,
        object: RootObject,
        member: Option<Int>,
    ) -> Result<RnsBggMatrix, RnsBggError> {
        match object {
            RootObject::Private => root.input_key(hash_key, PRIVATE_TAG, None),
            RootObject::State => root.input_key(hash_key, STATE_TAG, None),
            RootObject::Public => root.input_key(hash_key, PUBLIC_TAG, member),
            RootObject::Label => root.input_key(hash_key, LABEL_TAG, member),
            RootObject::Helper => {
                let member = member
                    .ok_or(RnsBggError::Binding("a helper key names its member"))?
                    .expression()
                    .map_err(|_| RnsBggError::Binding("a helper member is an index expression"))?;
                Ok(root.key(hash_key, RnsBggRoot::helper_tag(self.automorphism(member))))
            }
        }
    }

    /// The automorphism `automorphisms[position]` of a position expression.
    pub fn automorphism(&self, position: IntExpr) -> Int {
        Int::evaluate(
            IntExpr::Select {
                selector: Box::new(position),
                branches: self.automorphisms.iter().copied().map(IntExpr::constant).collect(),
            }
            .canonicalize(),
        )
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
    /// public gadget-action gain and `||mu_left||_1 = 1`. Product chains padded to two leaves
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
        if symbolic_labels && plan.automorphisms.len() > 1 {
            let table = LutTable::automorphism_family(
                plan.automorphisms.iter().copied().filter(|k| *k != 1).collect(),
            )?;
            plan.helper_families.insert(
                table.helper_family_key(),
                table
                    .branch_automorphisms(plan.ring_dimension)
                    .map_err(|_| Dio2Error::Parameters("a helper does not fit the ring"))?,
            );
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
    /// Deferred nonidentity helper automorphisms, in plan order (production only).
    pub helper_blocks: Option<usize>,
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
    /// The outputs a composition reads by column blocks where it decodes them: the smudge row
    /// and, for a refreshing step, the CBD row. Their PRF members are lowered in the loops that
    /// read the blocks (`mxx_exponent_lut::api::ProgramBuild::lower_deferred`).
    pub fn deferred_outputs(&self) -> BTreeSet<usize> {
        std::iter::once(self.outputs.smudge)
            .chain(self.outputs.cbd)
            .chain(self.outputs.helper_blocks)
            .collect()
    }

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
            helper_blocks: None,
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
            if plan.symbolic_labels && plan.automorphisms.len() > 1 {
                let indices =
                    plan.automorphisms.iter().copied().filter(|k| *k != 1).collect::<Vec<_>>();
                let count = indices.len();
                let lut = builder.register_lut(LutTable::automorphism_family(indices)?)?;
                let family = builder
                    .parallel(FamilyRange::full(IntExpr::constant(count))?, |body, _| {
                        Ok(Operand::Scalar(body.lut(Operand::Scalar(packed), lut)?))
                    })?;
                let row = builder.sum_columns(family, count)?;
                outputs.helper_blocks = Some(wires.len());
                wires.push(Operand::Scalar(row));
            } else {
                for sigma in plan.automorphisms.iter().copied().filter(|sigma| *sigma != 1) {
                    let table =
                        LutTable::automorphism(sigma, LutOutputRepresentation::Coefficient)?;
                    let lut = builder.register_lut(table)?;
                    let automorphed = builder.lut(Operand::Scalar(packed), lut)?;
                    outputs.automorphed.insert(sigma, wires.len());
                    wires.push(Operand::Scalar(automorphed));
                }
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

/// The product of the public encodings `leaves` (in ascending order) as the right-deep chain
/// `l_0 (l_1 (... l_(p-1)))`, padded with the one-encoding `one` to at least two leaves: the
/// multiplication order of the production label products ([`super::labels`]).
fn public_product(
    builder: &mut ExponentLutProgramBuilder,
    mut leaves: Vec<Operand<WireId>>,
    one: &Operand<WireId>,
) -> Result<WireId, Dio2Error> {
    while leaves.len() < 2 {
        leaves.push(one.clone());
    }
    let mut product = leaves.pop().expect("two leaves");
    while let Some(leaf) = leaves.pop() {
        product = Operand::Scalar(builder.mul_public(leaf, product)?);
    }
    match product {
        Operand::Scalar(wire) => Ok(wire),
        _ => Err(Dio2Error::Parameters("a label product needs a product")),
    }
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
        let chunk = context.input::<mxx_dsl::Int>("chunk", mxx_dsl::IntType).unwrap();
        let rows = xbits_rows(&function, ring.identity(1), chunk).unwrap();
        let graph = context.output("rows", rows).unwrap().build().unwrap();
        let mut execution = CpuExecution::new(cpu_backend([level.clone()]));
        let maximum = (1usize << bits) - 1;
        for chunk in [0, maximum, rand::rng().random_range(0..=maximum)] {
            let result = execute_graph(
                StageGraph::new(
                    mxx_dsl::BuiltGraph { graph: graph.graph.clone() },
                    BTreeMap::from([(
                        "chunk".to_owned(),
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
