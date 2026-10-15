//! The Program of one step (spec §6), its refresh targets, and the root objects its inputs read.
//! The chain never drops a modulus, so one Program serves the decoders of every prime.
//!
//! The step input `S_i` is a set of root objects (rows at `Q_0`, see
//! [`mxx_exponent_lut::rns_bgg::RnsBggRoot`]): `ONEPRIV` ([`PRIVATE_TAG`]), the state
//! ([`STATE_TAG`]), the label column `0` with the one-encoding `ONEPUB` as its zero member
//! ([`PUBLIC_TAG`], the PRF's input family of `v + 1` members), the other label columns
//! ([`LABEL_TAG`], `v m_v` members, entry `(r, 1 + k)` at `r m_v + k`), the LUT helpers of every
//! automorphism of [`StepPlan::automorphisms`], and the anchor hint. Every Program input reads
//! one of them under its root tag, so the samplers' inputs (each PRF builds its own accumulator
//! and input family) all read the same rows.
//!
//! The Program of a step that is followed by another one ([`ChainProgram::refreshing`]) computes,
//! besides `Pi_i`, the σ sampler (uniform signs) and its automorphisms, the CBD fresh noise of
//! every refresh part (one PRF, one row of every part's blocks), the smudges of every part and
//! prime (one PRF, one row of every smudge's columns), and the label products; the last step's
//! ([`ChainProgram::last`]) computes `Pi_i` and the smudge of the output decoder. The Program
//! does not grow with the number of parts.

use super::{Dio2Error, Dio2Parameters, StreamingFunction, function::StepWires};
use mxx_exponent_lut::{
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, FamilyRange,
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
    /// The automorphisms `K` of every LUT of every step (spec §5), ascending; `1` among them.
    pub automorphisms: Vec<usize>,
    /// The refresh targets of a step, in part order: part `(t, j)` is column block `2 t + j`.
    pub targets: Vec<Target>,
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
    /// The plan of `function` on `parameters`.
    pub fn new(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
    ) -> Result<Self, Dio2Error> {
        let columns = parameters.columns()?;
        let mut plan = Self {
            automorphisms: Vec::new(),
            targets: Vec::new(),
            columns,
            label_rows: parameters.label_rows(),
            label_width: parameters.label_width(),
            ring_dimension: parameters.ring_dimension as usize,
            smudge_digits: parameters.smudge_digits,
            primes: parameters.primes.len(),
            helper_families: BTreeMap::new(),
        };
        // Probe Programs of every step on declared key shapes give every LUT a step reads: the
        // automorphisms of the steps' own LUTs (the automorphism tables of the σ sampler only add
        // members of this set) and every helper family.
        let mut automorphisms = BTreeSet::new();
        let probe_shape = |role: &str, members| {
            parameters
                .prf
                .declared_shape(0, role, members)
                .map_err(|_| Dio2Error::Parameters("the PRF keys do not fit"))
        };
        let (n, d) = (plan.ring_dimension, plan.smudge_digits);
        for step in 0..function.stream_length() {
            let blocks = usize::from(step + 1 < function.stream_length());
            let smudge_columns = if blocks == 0 { 1 } else { blocks * plan.primes * columns };
            let keys = ChainKeys {
                sigma: (blocks > 0).then(|| probe_shape("sigma", n)).transpose()?,
                cbd: (blocks > 0).then(|| probe_shape("cbd", blocks * columns * n)).transpose()?,
                smudge: probe_shape("smudge", smudge_columns * n * d)?,
            };
            let probe = ChainProgram::build(
                parameters,
                function,
                &plan,
                step,
                &vec![false; function.eval_chunk_bits],
                &keys,
                blocks,
            )?;
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
        plan.targets = [Target::Private, Target::Public, Target::State]
            .into_iter()
            .chain((0..plan.label_rows).flat_map(|row| {
                (0..=plan.label_width).map(move |column| Target::Label { row, column })
            }))
            .chain(plan.automorphisms.iter().map(|sigma| Target::Helper(*sigma)))
            .chain((0..anchors).map(Target::Anchor))
            .collect();
        Ok(plan)
    }

    /// The rows of a step input as its consumers read them: per graph input name (an imported
    /// encryption's [`root_row_input_name`] or [`root_helper_input_name`]), its port and the
    /// targets whose rows it holds, and whether it is a family of them (else one row). Every
    /// stage that produces a step input outputs these names, so a protocol links each input to
    /// its producer by name.
    pub fn input_rows(&self) -> Vec<(String, usize, Vec<Target>, bool)> {
        let public = (0..self.label_rows)
            .map(|row| Target::Label { row, column: 0 })
            .chain([Target::Public])
            .collect::<Vec<_>>();
        let labels = (0..self.label_rows)
            .flat_map(|row| (1..=self.label_width).map(move |column| Target::Label { row, column }))
            .collect::<Vec<_>>();
        let mut rows = Vec::new();
        for port in 0..2 {
            for (tag, targets, family) in [
                (PRIVATE_TAG, vec![Target::Private], false),
                (STATE_TAG, vec![Target::State], false),
                (PUBLIC_TAG, public.clone(), true),
                (LABEL_TAG, labels.clone(), true),
            ] {
                rows.push((root_row_input_name(tag, port), port, targets, family));
            }
            for (key, sigmas) in &self.helper_families {
                let helpers = sigmas.iter().map(|sigma| Target::Helper(*sigma)).collect();
                rows.push((root_helper_input_name(key, port), port, helpers, true));
            }
        }
        rows
    }

    /// The number of refresh parts' column blocks: two per target.
    pub fn blocks(&self) -> usize {
        2 * self.targets.len()
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
        self.smudges() * self.columns * self.ring_dimension * self.smudge_digits
    }

    /// The members of the output decoder's smudge PRF: `n d` for its one column.
    pub fn output_smudge_members(&self) -> usize {
        self.ring_dimension * self.smudge_digits
    }
}

/// The key shapes of the PRFs of one Program.
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
    /// The smudges of every refresh part and prime ([`StepPlan::smudges`]): a row of `w_0`
    /// columns per smudge; of the output decoder for the last step: one column.
    pub smudge: usize,
    /// The label products, entry `(r, c)` at `r (m_v + 1) + c`.
    pub labels: Vec<usize>,
}

/// The Program of step `i` on input chunk `x`, with the root tag of every input.
pub struct ChainProgram {
    pub program: ExponentLutProgram,
    pub root_inputs: BTreeMap<WireId, String>,
    pub outputs: ChainOutputs,
    pub sigma: Option<PrfSampler>,
    /// The CBD sampler.
    pub cbd: Option<PrfSampler>,
    /// The smudge PRF.
    pub smudge: SparseLwrPrf,
    /// The obfuscator's bits.
    pub y: RhsId,
}

impl ChainProgram {
    /// The Program of a step followed by a refresh, whose `plan.blocks()` part blocks each
    /// decode a row of `w_0` columns.
    pub fn refreshing(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        step: usize,
        x: &[bool],
        keys: &ChainKeys,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, plan, step, x, keys, plan.blocks())
    }

    /// The Program of the last step, whose state the output decoder reads.
    pub fn last(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        step: usize,
        x: &[bool],
        keys: &ChainKeys,
    ) -> Result<Self, Dio2Error> {
        Self::build(parameters, function, plan, step, x, keys, 0)
    }

    /// The Program with `blocks` refresh part blocks; none for the last step, whose output
    /// decoder decodes one column.
    fn build(
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
        step: usize,
        x: &[bool],
        keys: &ChainKeys,
        blocks: usize,
    ) -> Result<Self, Dio2Error> {
        let last = blocks == 0;
        let profile = parameters.profile()?;
        let mut builder = ExponentLutProgramBuilder::new();
        let mut root_inputs = BTreeMap::new();
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
            PrfKeys::Shape(keys.smudge.clone()),
            LutOutputRepresentation::Coefficient,
        )?;
        root_prf(&smudge, &mut root_inputs);
        let public = Operand::Member {
            family: smudge.input_vector(),
            index: IntExpr::constant(plan.label_rows),
        };
        let state = builder.input(AttributeKnowledge::PrivatePayload)?;
        root_inputs.insert(state, STATE_TAG.to_owned());
        let x_wires = x
            .iter()
            .enumerate()
            .map(|(bit, set)| {
                builder
                    .mul_small_constant(
                        public.clone(),
                        vec![1],
                        IntExpr::constant((function.x_monomial)(bit, *set)),
                    )
                    .map(Operand::Scalar)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let y = builder.rhs_family(IntExpr::constant(function.obf_chunk_bits))?;
        let new_state =
            function.step(step).build(&mut builder, &StepWires { state, x: x_wires, y })?;
        let mut outputs = ChainOutputs {
            private: None,
            public: None,
            state: 0,
            sigma: None,
            automorphed: BTreeMap::new(),
            cbd: None,
            smudge: 1,
            labels: Vec::new(),
        };
        let mut wires = vec![
            Operand::Scalar(new_state),
            Operand::Scalar(placed_row(&mut builder, &smudge, plan, smudge_columns, true)?),
        ];
        let (mut sigma_sampler, mut cbd_sampler) = (None, None);
        if !last {
            let sigma_shape = keys
                .sigma
                .clone()
                .ok_or(Dio2Error::Parameters("a refreshing step needs σ keys"))?;
            let (sampler, packed) = PrfSampler::sign_polynomial(
                &mut builder,
                profile,
                PrfKeys::Shape(sigma_shape),
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
                    PrfKeys::Shape(shape.clone()),
                    PrfDistribution::Cbd { eta: 1 },
                )?;
                root_prf(cbd.prf(), &mut root_inputs);
                let row = placed_row(&mut builder, cbd.prf(), plan, blocks * plan.columns, false)?;
                outputs.cbd = Some(wires.len());
                wires.push(Operand::Scalar(row));
                cbd_sampler = Some(cbd);
            }
            // The label columns `1..=m_v` and the products of the next label matrix.
            let labels = builder.input_family(
                IntExpr::constant(plan.label_rows * plan.label_width),
                AttributeKnowledge::PublicPayload,
            )?;
            root_inputs.insert(labels, LABEL_TAG.to_owned());
            let transition = label_transition(parameters, x);
            for row in 0..plan.label_rows {
                for column in 0..=plan.label_width {
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
        let program = builder.finish(wires)?;
        Ok(Self {
            program,
            root_inputs,
            outputs,
            sigma: sigma_sampler,
            cbd: cbd_sampler,
            smudge,
            y,
        })
    }
}

/// The row of `columns` columns (a block per column) of the samples of `prf`'s member family, as
/// `FreshNoiseProgram` and the decoder's smudge Program place them.
///
/// - CBD noise (`smudge` false): member `column n + k` lands at `X^k` of block `column`.
/// - A smudge (`smudge` true): member `(column n + k) d + digit` lands at `X^k` of column `column`
///   with weight `P^digit`. The remaining weight `2^-1` of every member (see
///   `mxx_exponent_lut::decoder::decoding`) acts on the summed row in the refresh, by a
///   re-targeting to the scalar form `2^-1`.
fn placed_row(
    builder: &mut ExponentLutProgramBuilder,
    prf: &SparseLwrPrf,
    plan: &StepPlan,
    columns: usize,
    smudge: bool,
) -> Result<WireId, Dio2Error> {
    let PrfOutput::Family(family) = prf.output() else {
        return Err(Dio2Error::Parameters("a sampler needs several members"));
    };
    let n = plan.ring_dimension;
    let digits = if smudge { plan.smudge_digits } else { 1 };
    let base = prf.profile().p() as i64;
    let mut sum = None;
    for digit in 0..digits {
        let range = FamilyRange::full(IntExpr::constant(columns * n))?;
        let placed = builder.parallel(range, |builder, index| {
            let member = IntExpr::Add(
                Box::new(IntExpr::Mul(
                    Box::new(index.clone()),
                    Box::new(IntExpr::constant(digits)),
                )),
                Box::new(IntExpr::constant(digit)),
            )
            .canonicalize();
            let coefficient =
                IntExpr::Rem(Box::new(index), Box::new(IntExpr::constant(n))).canonicalize();
            builder
                .mul_small_constant(
                    Operand::Member { family, index: member },
                    vec![base.pow(digit as u32)],
                    coefficient,
                )
                .map(Operand::Scalar)
        })?;
        let row = builder.sum_columns(placed, columns)?;
        sum = Some(match sum {
            None => row,
            Some(sum) => builder.add(Operand::Scalar(sum), Operand::Scalar(row))?,
        });
    }
    sum.ok_or(Dio2Error::Parameters("a smudge has at least one digit"))
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

/// The label transition `D_x in {0,1}^(m_v x (m_v + 1))` of chunk `x` (spec §3): hashed from
/// the chunk alone, shared by every step.
pub fn label_transition(parameters: &Dio2Parameters, x: &[bool]) -> Vec<Vec<bool>> {
    let width = parameters.label_width();
    let mut digest = Sha256::new();
    digest.update(b"mxx-io/dio2/label-transition/v1");
    digest.update((x.len() as u64).to_le_bytes());
    digest.update(x.iter().map(|bit| u8::from(*bit)).collect::<Vec<_>>());
    let seed: [u8; 32] = digest.finalize().into();
    let bit = |index: usize| {
        let mut digest = Sha256::new();
        digest.update(seed);
        digest.update(((index / 256) as u64).to_le_bytes());
        let block: [u8; 32] = digest.finalize().into();
        let offset = index % 256;
        block[offset / 8] >> (offset % 8) & 1 == 1
    };
    (0..width).map(|k| (0..=width).map(|c| bit(k * (width + 1) + c)).collect()).collect()
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
        let program =
            ChainProgram::refreshing(&parameters, &function, &plan, 0, &[true], &keys).unwrap();
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
