//! The refresh of a step's targets (spec §6.3–§6.8) and the output decoder of the last step
//! (spec §8), composed into the step's online evaluation graph.
//!
//! The compose reads the lowered (unfinished) outputs of a [`ChainProgram`]: a private wire in
//! the state form of its level (a raw coefficient wire too), a public wire in the public form of
//! its level. On the decoding level `L` (of modulus `Q_0`, like every level) it forms
//!
//! - the new-mask rows `Sigma = (g, omega)` and `Sigma^(k) = (g^(k), omega)` in the public form of
//!   `L` (spec §4.3, §4.4): `g^(k)` is component 1 of the σ sampler's `Aut_k`, `omega` component 1
//!   of the one-encoding, each re-targeted by `Gamma`;
//! - every target of [`StepPlan::targets`] in the state form of `L` (spec §6.6), the labels by the
//!   public multiplication of the label products by `E^sigma` (spec §6.5);
//! - for every part and every prime `q` of `Q_0`, the decoded row `d = Sigma X - c_j Y + f Z` and
//!   its key (spec §6.7), plus the smudge of the part and prime (the smudge row re-targeted to the
//!   scalar form `2^-1`, see [`super::step::ChainProgram`]). The CBD noise `f` of a part is shared
//!   by its primes, so the residues recombine to one row.
//!
//! In [`ComposeMode::Keys`] (the obfuscator's run on zero rows) it outputs the decoded key
//! `[K_0; K_1]` of every part and prime, whose preimage under the anchor the obfuscator samples.
//! In [`ComposeMode::Decode`] (the evaluator's run) it unmasks every part and prime with the
//! anchor hint and its preimage, rounds it to `y_j mod q`, and sums the residues' CRT terms to the
//! part's row of the next step input modulo `Q_0`; it outputs the next step input by the names
//! its consumers read ([`StepPlan::input_rows`] and [`HINT_INPUT`], the anchor blocks side by
//! side cut to the hint's width).
//!
//! The output decoder re-targets the state to the scalar form `Q_q` of the first prime `q` and
//! rounds the unmasked row to the state monomial modulo `q`.

use super::{
    Dio2Error, Dio2Parameters,
    step::{ChainProgram, StepPlan, Target},
};
use mxx_backends::poly::PolyParams;
use mxx_dsl::{HashTag, Int, Mat, Ring};
use mxx_exponent_lut::{
    ExponentLutProgram,
    program::{AttributeKnowledge, Operand, WireId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggEncoding, RnsBggForm, RnsBggMatrix, RnsBggParameters,
        RnsBggRoot, RnsBggRow, RnsOnlineEvalBuild, refresh_combination, residue_term,
    },
};
use mxx_ir_core::{IntExpr, node::ConcatAxis};
use num_bigint::{BigInt, BigUint};

/// The graph input of the anchor hint `b_i` (`1 x 2 (w_0 + 2)` at `Q_0`).
pub const HINT_INPUT: &str = "dio2/hint";
/// The graph input of the next anchor `B_(i+1)` (`2 x 2 (w_0 + 2)` at `Q_0`). The refresh pads it
/// to whole refresh blocks with columns hashed under the next root (spec §4.7).
pub const NEXT_ANCHOR_INPUT: &str = "dio2/next-anchor";

/// The graph input of the preimages of every part for the prime `primes[prime]`: part
/// `(t, j)` (target `t` of [`StepPlan::targets`], component `j`) at member `2 t + j`.
pub fn preimage_input(prime: usize) -> String {
    format!("dio2/preimage/{prime}")
}

/// The graph output of the decoded keys of every part for the prime `primes[prime]`, one member
/// per part as [`preimage_input`] orders them.
pub fn key_output(prime: usize) -> String {
    format!("dio2/key/{prime}")
}

/// The graph output of the decoded key of the output decoder.
pub const OUTPUT_KEY_OUTPUT: &str = "dio2/output-key";
/// The graph input of the output decoder's preimage.
pub const OUTPUT_PREIMAGE_INPUT: &str = "dio2/output-preimage";
/// The graph output of the decoded plaintext of the output decoder.
pub const OUTPUT_PLAINTEXT_OUTPUT: &str = "dio2/output";

/// What a composed online evaluation outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComposeMode {
    /// The decoded keys, for the preimages.
    Keys,
    /// The decoded residues, unmasked with the hint and the preimages.
    Decode,
}

/// An encoding as a row and its key.
type Encoding = (RnsBggRow, RnsBggMatrix);

/// The refresh of a step.
pub struct StepCompose<'a> {
    pub parameters: &'a Dio2Parameters,
    pub chain_parameters: &'a RnsBggParameters,
    pub plan: &'a StepPlan,
    pub program: &'a ChainProgram,
    /// The next step's root, whose keys are the targets' `A'`.
    pub next_root: Option<&'a RnsBggRoot>,
    pub mode: ComposeMode,
}

impl StepCompose<'_> {
    /// The level and form of a lowered output wire.
    fn carrier(
        &self,
        workload: &RnsBggDerivedWorkload,
        output: usize,
    ) -> Result<(usize, RnsBggForm), Dio2Error> {
        carrier(&self.program.program, workload, output)
    }

    /// `Gamma` of output `output` to the form `target` (a row in the decoding ring).
    fn retarget(
        &self,
        workload: &RnsBggDerivedWorkload,
        outputs: &[RnsBggEncoding],
        output: usize,
        target: Mat,
    ) -> Result<Encoding, Dio2Error> {
        let (level, form) = self.carrier(workload, output)?;
        let encoding = &outputs[output];
        Ok(self.chain_parameters.retarget_encoding(
            level,
            form,
            self.parameters.decoding_level(),
            target,
            (encoding.row().clone(), encoding.public_key().matrix().clone()),
        )?)
    }

    /// Compose the refresh (or, for the last step, the output decoder) into `builder`'s graph
    /// from the lowered Program outputs `outputs`.
    ///
    /// The parts are decoded in one loop: the targets' rows and keys and the parts' `A'` are
    /// gathered into families (one member per part), and the body decodes part `p` for every
    /// prime from its members, its CBD blocks and its smudges (column ranges of the Program's
    /// rows at `p`), so the graph does not grow with the number of parts.
    pub fn compose(
        &self,
        builder: &mut RnsOnlineEvalBuild<'_>,
        outputs: &[RnsBggEncoding],
    ) -> Result<(), Dio2Error> {
        let workload = RnsBggDerivedWorkload::from_program(&self.program.program)?;
        let level = self.parameters.decoding_level();
        let ring = self.chain_parameters.ring(level)?;
        let modulus: BigUint = self.chain_parameters.level(level)?.modulus().as_ref().clone();
        // The smudge rows are re-targeted to the scalar `2^-1` (the weight of the odd digits), and
        // the halved weights leave `2^-1` in every coefficient of the decoded row; removing it
        // from every coefficient centers the smudge (`mxx_exponent_lut::decoder::decoding`).
        let half = self.chain_parameters.scalar(level, BigInt::from((&modulus + 1u8) / 2u8))?;
        let half_coefficients =
            ring.all_coefficients((1, 1), IntExpr::constant((&modulus + 1u8) / 2u8));
        let state_form = self.chain_parameters.form(level, RnsBggForm::State)?;
        let public_form = self.chain_parameters.form(level, RnsBggForm::Public)?;
        let outs = &self.program.outputs;
        let columns = self.plan.columns;
        // `Gamma` of the `blocks` carriers of output `output` from carrier `first` on (`first` a
        // loop expression) to `blocks` copies of `target`: the blocks' columns of the row and
        // the key, acted on by the block-diagonal digits of `target`.
        let blocks_of = |output: usize, first: IntExpr, blocks: usize, target: &Mat| {
            let (source, form) = self.carrier(&workload, output)?;
            let width = self.chain_parameters.component_columns(source)?;
            let range = || {
                let start =
                    IntExpr::Mul(Box::new(first.clone()), Box::new(IntExpr::constant(width)))
                        .canonicalize();
                let end = IntExpr::Add(
                    Box::new(start.clone()),
                    Box::new(IntExpr::constant(blocks * width)),
                )
                .canonicalize();
                Some(mxx_ir_core::node::IndexRange { start, end })
            };
            let encoding = &outputs[output];
            let slice = |mat: &Mat| mat.clone().slice(None, range());
            let row = encoding.row();
            let key = encoding.public_key().matrix();
            let sliced = (
                (slice(&row.0), slice(&row.1)),
                ((slice(&key.0.0), slice(&key.0.1)), (slice(&key.1.0), slice(&key.1.1))),
            );
            let diagonal = Mat::concat(ConcatAxis::Diagonal, vec![target.clone(); blocks]);
            Ok::<_, Dio2Error>(
                self.chain_parameters.retarget_encoding(source, form, level, diagonal, sliced)?,
            )
        };
        let component_one = |encoding: Encoding| (encoding.0.1, (encoding.1.0.1, encoding.1.1.1));
        if outs.private.is_none() {
            // The last step: the output decoder of the state modulo the first prime `q`, at the
            // decoding level, in the scalar form `Q_q`.
            let q = self.parameters.primes[0];
            let cofactor = self.chain_parameters.scalar(level, BigInt::from(&modulus / q))?;
            let selected = self.retarget(&workload, outputs, outs.state, cofactor)?;
            let smudge = blocks_of(outs.smudge, IntExpr::constant(0), 1, &half)?;
            let row = selected.0.1 + smudge.0.1;
            let key = (selected.1.0.1 + smudge.1.0.1, selected.1.1.1 + smudge.1.1.1);
            match self.mode {
                ComposeMode::Keys => builder.add_output(
                    OUTPUT_KEY_OUTPUT,
                    Mat::concat(ConcatAxis::Rows, vec![key.0, key.1]),
                ),
                ComposeMode::Decode => {
                    let preimage = self.root_ring().preimage_input(
                        OUTPUT_PREIMAGE_INPUT,
                        (super::anchor_columns(columns), 1),
                        BigInt::from(self.preimage_bound()?),
                    );
                    let unmasked = unmask(&ring, &self.hint(), preimage, row) - half_coefficients;
                    let plaintext = Ring::from_crt_moduli(
                        vec![IntExpr::from(q)],
                        self.parameters.ring_dimension,
                    );
                    builder
                        .add_output(OUTPUT_PLAINTEXT_OUTPUT, unmasked.modulus_switch(&plaintext));
                }
            }
            return Ok(());
        }
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        // `omega` and `g^(k)`: component 1 in the public form of `L`.
        let omega = component_one(self.retarget(
            &workload,
            outputs,
            outs.public.ok_or_else(missing)?,
            public_form.clone(),
        )?);
        let sigma = |k: usize| -> Result<(Mat, (Mat, Mat)), Dio2Error> {
            let output = match k {
                1 => outs.sigma.ok_or_else(missing)?,
                k => *outs.automorphed.get(&k).ok_or_else(missing)?,
            };
            Ok(component_one(self.retarget(&workload, outputs, output, public_form.clone())?))
        };
        let pair = |g: &(Mat, (Mat, Mat))| -> Encoding {
            (
                (g.0.clone(), omega.0.clone()),
                ((g.1.0.clone(), omega.1.0.clone()), (g.1.1.clone(), omega.1.1.clone())),
            )
        };
        let to_state = |encoding: Encoding| -> Result<Encoding, Dio2Error> {
            Ok(self.chain_parameters.retarget_encoding(
                level,
                RnsBggForm::Public,
                level,
                state_form.clone(),
                encoding,
            )?)
        };
        let g = sigma(1)?;
        let mask = to_state(pair(&g))?;

        let next_root = self.next_root.ok_or_else(missing)?;
        let next_hash_key = next_root.hash_key_input();
        let root_ring = next_root.ring();
        let anchor = super::anchor_columns(columns);
        let padding = 2 * columns * self.anchor_blocks() - anchor;
        let next_anchor = Mat::concat(
            ConcatAxis::Columns,
            vec![
                root_ring.input(NEXT_ANCHOR_INPUT, (2, anchor)),
                root_ring.hash_matrix(
                    next_hash_key.clone(),
                    HashTag::from(b"dio2/anchor-padding".to_vec()),
                    (2, padding),
                ),
            ],
        );
        let zero_row = ring.zero((1, self.chain_parameters.component_columns(level)?));
        // Every part's target row and key, and its `A'_j`, as one member per part.
        let mut rows = Vec::with_capacity(self.plan.blocks());
        let mut key_tops = Vec::with_capacity(self.plan.blocks());
        let mut key_bottoms = Vec::with_capacity(self.plan.blocks());
        let mut a_primes = Vec::with_capacity(self.plan.blocks());
        for target in &self.plan.targets {
            let encoding = match target {
                Target::Private => self.retarget(
                    &workload,
                    outputs,
                    outs.private.ok_or_else(missing)?,
                    state_form.clone(),
                )?,
                Target::Public => mask.clone(),
                Target::State => {
                    self.retarget(&workload, outputs, outs.state, state_form.clone())?
                }
                Target::Label { row, column } => {
                    let output = outs.labels[row * (self.plan.label_width + 1) + column];
                    let label = self.retarget(&workload, outputs, output, public_form.clone())?;
                    let alpha = outputs[output]
                        .public_payload()
                        .ok_or(Dio2Error::Parameters("a label product carries its payload"))?
                        .clone()
                        .reduce_modulus(&ring);
                    self.chain_parameters.mul_public_rows(
                        level,
                        (&label.0, &label.1, &alpha),
                        (&mask.0, &mask.1),
                    )?
                }
                Target::Helper(k) => to_state(pair(&sigma(*k)?))?,
                Target::Anchor(_) => (
                    (zero_row.clone(), zero_row.clone()),
                    ((zero_row.clone(), zero_row.clone()), (zero_row.clone(), zero_row.clone())),
                ),
            };
            for component in 0..2 {
                let a_prime = match (target, target.key_tag(self.plan)) {
                    (_, Some(tag)) => {
                        a_prime_block(next_root.key(next_hash_key.clone(), tag), component)
                    }
                    (Target::Anchor(block), None) => {
                        let start = (2 * block + component) * columns;
                        next_anchor.clone().slice(
                            None,
                            Some(mxx_ir_core::node::IndexRange {
                                start: IntExpr::constant(start),
                                end: IntExpr::constant(start + columns),
                            }),
                        )
                    }
                    _ => return Err(missing()),
                };
                a_primes.push(a_prime.reduce_modulus(&ring));
                let pick = |pair: &(Mat, Mat)| {
                    if component == 0 { pair.0.clone() } else { pair.1.clone() }
                };
                rows.push(pick(&encoding.0));
                key_tops.push(pick(&encoding.1.0));
                key_bottoms.push(pick(&encoding.1.1));
            }
        }
        let pack = |members: Vec<Mat>| mxx_dsl::Family::pack(members);
        let (rows, key_tops, key_bottoms, a_primes) =
            (pack(rows)?, pack(key_tops)?, pack(key_bottoms)?, pack(a_primes)?);
        let preimages = (0..self.plan.primes)
            .map(|prime| {
                Ok(self.root_ring().preimage_input_family(
                    preimage_input(prime),
                    self.plan.blocks(),
                    (anchor, columns),
                    BigInt::from(self.preimage_bound()?),
                ))
            })
            .collect::<Result<Vec<_>, Dio2Error>>()?;
        let reference = self.parameters.reference();
        let cbd = outs.cbd.ok_or_else(missing)?;
        let half_blocks =
            Mat::concat(ConcatAxis::Columns, vec![half_coefficients.clone(); columns]);
        let hint = self.hint();
        // Part `p`'s decoded row (or key) for the prime `primes[prime]`.
        let decode = |part: &Int, prime: usize, keys: Option<usize>| -> Result<Mat, Dio2Error> {
            let index = part.expression()?;
            let fresh = component_one(blocks_of(
                cbd,
                IntExpr::Mul(Box::new(index.clone()), Box::new(IntExpr::constant(columns)))
                    .canonicalize(),
                columns,
                &public_form,
            )?);
            let smudge = component_one(blocks_of(
                outs.smudge,
                IntExpr::Mul(
                    Box::new(IntExpr::Add(
                        Box::new(IntExpr::Mul(
                            Box::new(index),
                            Box::new(IntExpr::constant(self.plan.primes)),
                        )),
                        Box::new(IntExpr::constant(prime)),
                    )),
                    Box::new(IntExpr::constant(columns)),
                )
                .canonicalize(),
                columns,
                &half,
            )?);
            let actions = self.chain_parameters.refresh_actions(
                level,
                a_primes.at(part.clone()),
                self.parameters.prime_index(prime)?,
                &reference,
            )?;
            let combine = |one: (Mat, Mat), target: Mat, fresh: Mat, smudge: Mat| {
                Ok::<_, Dio2Error>(
                    refresh_combination(one, target, (fresh, columns), &actions)? + smudge,
                )
            };
            Ok(match keys {
                Some(k) => {
                    let pick =
                        |pair: &(Mat, Mat)| if k == 0 { pair.0.clone() } else { pair.1.clone() };
                    let target = if k == 0 {
                        key_tops.at(part.clone())
                    } else {
                        key_bottoms.at(part.clone())
                    };
                    combine((pick(&g.1), pick(&omega.1)), target, pick(&fresh.1), pick(&smudge.1))?
                }
                None => combine(
                    (g.0.clone(), omega.0.clone()),
                    rows.at(part.clone()),
                    fresh.0,
                    smudge.0,
                )?,
            })
        };
        let mut failure = None;
        let mut guard = |result: Result<Mat, Dio2Error>| {
            result.map_err(|error| {
                failure = Some(error);
                mxx_dsl::DslError::Schema
            })
        };
        match self.mode {
            ComposeMode::Keys => {
                for prime in 0..self.plan.primes {
                    let keys = mxx_dsl::parallel(self.plan.blocks(), |part| {
                        let top = guard(decode(&part, prime, Some(0)))?;
                        let bottom = guard(decode(&part, prime, Some(1)))?;
                        Ok(Mat::concat(ConcatAxis::Rows, vec![top, bottom]))
                    });
                    match keys {
                        Ok(keys) => builder.add_output(key_output(prime), keys),
                        Err(error) => return Err(failure.take().unwrap_or(error.into())),
                    }
                }
            }
            ComposeMode::Decode => {
                let refreshed = mxx_dsl::parallel(self.plan.blocks(), |part| {
                    let mut sum: Option<Mat> = None;
                    for (prime, q) in self.parameters.primes.iter().copied().enumerate() {
                        let row = guard(decode(&part, prime, None))?;
                        let unmasked = unmask(&ring, &hint, preimages[prime].at(part.clone()), row) -
                            half_blocks.clone();
                        let residue = unmasked.modulus_switch(&Ring::from_crt_moduli(
                            vec![IntExpr::from(q)],
                            self.parameters.ring_dimension,
                        ));
                        let term = guard(
                            residue_term(
                                &self.parameters.primes,
                                q,
                                self.parameters.ring_dimension,
                                residue,
                            )
                            .map_err(Dio2Error::from),
                        )?;
                        sum = Some(match sum {
                            None => term,
                            Some(sum) => sum + term,
                        });
                    }
                    sum.ok_or(mxx_dsl::DslError::Schema)
                });
                let refreshed = match refreshed {
                    Ok(refreshed) => refreshed,
                    Err(error) => return Err(failure.take().unwrap_or(error.into())),
                };
                let index = |target: &Target, port: usize| {
                    self.plan
                        .targets
                        .iter()
                        .position(|other| other == target)
                        .map(|position| Int::constant((2 * position + port) as i64))
                        .ok_or_else(missing)
                };
                for (name, port, targets, family) in self.plan.input_rows() {
                    let members = targets
                        .iter()
                        .map(|target| Ok(refreshed.at(index(target, port)?)))
                        .collect::<Result<Vec<_>, Dio2Error>>()?;
                    if family {
                        builder.add_output(name, mxx_dsl::Family::pack(members)?);
                    } else {
                        builder.add_output(name, members.into_iter().next().ok_or_else(missing)?);
                    }
                }
                let blocks = self
                    .plan
                    .targets
                    .iter()
                    .filter(|target| matches!(target, Target::Anchor(_)))
                    .flat_map(|target| [index(target, 0), index(target, 1)])
                    .map(|index| Ok(refreshed.at(index?)))
                    .collect::<Result<Vec<_>, Dio2Error>>()?;
                let hint = Mat::concat(ConcatAxis::Columns, blocks).slice(
                    None,
                    Some(mxx_ir_core::node::IndexRange {
                        start: IntExpr::constant(0),
                        end: IntExpr::constant(anchor),
                    }),
                );
                builder.add_output(HINT_INPUT, hint);
            }
        }
        Ok(())
    }

    /// The refresh blocks of the next anchor: `ceil(2 (w_0 + 2) / (2 w_0))`.
    fn anchor_blocks(&self) -> usize {
        self.plan.targets.iter().filter(|target| matches!(target, Target::Anchor(_))).count()
    }

    /// The ring of `Q_0`.
    fn root_ring(&self) -> Ring {
        Ring::from_crt_moduli(
            self.parameters.primes.iter().copied().map(IntExpr::from).collect(),
            self.parameters.ring_dimension,
        )
    }

    /// The anchor hint `b` (at `Q_0`), a graph input of the stage's root scope.
    fn hint(&self) -> Mat {
        self.root_ring().input(HINT_INPUT, (1, super::anchor_columns(self.plan.columns)))
    }

    /// The coefficient bound of the anchor's preimages.
    fn preimage_bound(&self) -> Result<BigUint, Dio2Error> {
        super::preimage_bound(self.parameters)
    }
}

/// The level and form of the lowered output `output` of `program`: a private wire in the state
/// form of its level, a public wire in the public form.
pub(crate) fn carrier(
    program: &ExponentLutProgram,
    workload: &RnsBggDerivedWorkload,
    output: usize,
) -> Result<(usize, RnsBggForm), Dio2Error> {
    let wire = output_wire(&program.outputs()[output]);
    let level = workload.level(wire)?;
    let form = match program.wire_knowledge(wire)? {
        AttributeKnowledge::PublicPayload => RnsBggForm::Public,
        AttributeKnowledge::PrivatePayload => RnsBggForm::State,
    };
    Ok((level, form))
}

/// The wire of an output operand.
fn output_wire(operand: &Operand<WireId>) -> WireId {
    match operand {
        Operand::Scalar(wire) | Operand::Member { family: wire, .. } => *wire,
        Operand::Selected { family, .. } => *family,
    }
}

/// `b K_pre - row` in the decoding ring `ring`: the hint `hint` (at `Q_0`) times the preimage
/// `preimage` (`2 (w_0 + 2) x columns` at `Q_0`), reduced to `ring`.
fn unmask(ring: &Ring, hint: &Mat, preimage: mxx_dsl::Preimage, row: Mat) -> Mat {
    hint.clone().mul_small_rhs(preimage).reduce_modulus(ring) - row
}

/// `A'_(j) = [K_(0j); K_(1j)]` of a root key, `2 x w_0`.
fn a_prime_block(key: RnsBggMatrix, component: usize) -> Mat {
    let (top, bottom) = match component {
        0 => (key.0.0, key.1.0),
        _ => (key.0.1, key.1.1),
    };
    Mat::concat(ConcatAxis::Rows, vec![top, bottom])
}
