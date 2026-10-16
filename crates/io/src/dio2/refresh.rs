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
//!   scalar form `P^r / 2` for each digit `r`, then summed, see [`super::step::ChainProgram`]). The
//!   CBD noise `f` of a part is shared by its primes, so the residues recombine to one row.
//!
//! In [`ComposeMode::Keys`] (the obfuscator's run on zero rows) it outputs the decoded key
//! `[K_0; K_1]` of every part and prime, whose preimage under the anchor the obfuscator samples.
//! In [`ComposeMode::Decode`] (the evaluator's run) it unmasks every part and prime with the
//! anchor hint and its preimage, rounds it to `y_j mod q`, and sums the residues' CRT terms to the
//! part's row of the next step input modulo `Q_0`; it outputs the next step input by the names
//! its consumers read ([`StepPlan::input_groups`] and [`HINT_INPUT`], the anchor blocks side by
//! side cut to the hint's width).
//!
//! The output decoder re-targets the state to the scalar form `Q_q` of the first prime `q` and
//! rounds the unmasked row to the state monomial modulo `q`.

use super::{
    Dio2Error, Dio2Parameters,
    step::{
        ChainProgram, LABEL_TAG, LABEL_VALUES_INPUT, NoiseTargets, PUBLIC_TAG, StepPlan, Target,
    },
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
    /// The exact balanced public-product forest of §6.4, lowered as nested DSL families.
    /// Odd nodes are carried, not multiplied by noisy padding. Only zero/one-leaf columns
    /// use ONEPUB to reach two leaves, exactly as the scalar noise-model Program does.
    /// Consequently every contributing leaf has the same action depth and error as before.
    fn labels(&self, mask: &Encoding) -> Result<mxx_dsl::Family<Encoding>, Dio2Error> {
        let products = self
            .program
            .label_products
            .as_ref()
            .ok_or(Dio2Error::Input("symbolic label products are missing"))?;
        let root =
            self.chain_parameters.root().ok_or(Dio2Error::Input("the chain root is missing"))?;
        let ring = root.ring();
        let width = self.plan.label_width;
        let rows = (0..2)
            .map(|port| {
                ring.input_family(
                    mxx_exponent_lut::rns_bgg::root_row_input_name(LABEL_TAG, port),
                    self.plan.label_rows * width,
                    (1, self.plan.columns),
                )
            })
            .collect::<Vec<_>>();
        let public = (0..2)
            .map(|port| {
                ring.input_family(
                    mxx_exponent_lut::rns_bgg::root_row_input_name(PUBLIC_TAG, port),
                    self.plan.label_rows + 1,
                    (1, self.plan.columns),
                )
            })
            .collect::<Vec<_>>();
        let values =
            mxx_dsl::int_family_input(LABEL_VALUES_INPUT, self.plan.label_rows * (width + 1));
        let key = root.hash_key_input();
        let one_index = Int::constant(self.plan.label_rows as i64);
        let one = (
            (public[0].at(one_index.clone()), public[1].at(one_index.clone())),
            root.key(key.clone(), RnsBggRoot::input_tag(PUBLIC_TAG, Some(one_index))),
            ring.identity(1),
        );
        let level = self.parameters.decoding_level();
        let form = self.chain_parameters.form(level, RnsBggForm::Public)?;
        let stride = self.parameters.profile()?.layout.stride();
        let (packed_columns, counts) = products.columns();
        let chunk = mxx_dsl::DslContext::new("dio2/labels")
            .input::<Int>(super::step::CHUNK_INPUT, mxx_dsl::IntType)?;
        let mut failure = None;
        let labels = mxx_dsl::parallel(self.plan.label_rows * (width + 1), |member| {
            let row = member.clone().div(width + 1);
            let column = member.rem(width + 1);
            let count = counts.at(chunk.clone().mul(width + 1).add(column.clone()));
            let padded = mxx_dsl::select(
                count.clone().less(2).to_int(),
                vec![count.clone(), Int::constant(2)],
            )?;
            let mut current = mxx_dsl::parallel(products.max_leaves, |leaf| {
                let index =
                    products.index(&packed_columns, chunk.clone(), column.clone(), leaf.clone())?;
                let input = row.clone().mul(width).add(index.clone());
                let value = values.at(row.clone().mul(width + 1).add(index).add(1usize));
                let encoding = (
                    (rows[0].at(input.clone()), rows[1].at(input.clone())),
                    root.key(key.clone(), RnsBggRoot::input_tag(LABEL_TAG, Some(input))),
                    ring.identity(1).multiply_monomial(value.mul(stride)),
                );
                mxx_dsl::select(leaf.less(count.clone()).to_int(), vec![one.clone(), encoding])
            })?;
            let mut slots = products.max_leaves;
            let mut active = padded;
            while slots > 1 {
                let previous = current;
                current = mxx_dsl::parallel(slots.div_ceil(2), |pair| {
                    let left = previous.at(pair.clone().mul(2usize));
                    let right_index = pair.clone().mul(2usize).add(1usize);
                    let right = previous.at(right_index.clone().rem(slots));
                    let product = self
                        .chain_parameters
                        .mul_public_rows(0, (&left.0, &left.1, &left.2), (&right.0, &right.1))
                        .map_err(|error| {
                            failure = Some(Dio2Error::from(error));
                            mxx_dsl::DslError::Schema
                        })?;
                    mxx_dsl::select(
                        right_index.less(active.clone()).to_int(),
                        vec![left.clone(), (product.0, product.1, left.2 * right.2)],
                    )
                })?;
                slots = slots.div_ceil(2);
                active = active.add(1usize).div(2usize);
            }
            let label = current.at(Int::constant(0));
            let encoding = self
                .chain_parameters
                .retarget_encoding(0, RnsBggForm::Public, level, form.clone(), (label.0, label.1))
                .and_then(|encoding| {
                    self.chain_parameters.mul_public_rows(
                        level,
                        (&encoding.0, &encoding.1, &label.2),
                        (&mask.0, &mask.1),
                    )
                })
                .map_err(|error| {
                    failure = Some(Dio2Error::from(error));
                    mxx_dsl::DslError::Schema
                })?;
            Ok(encoding)
        });
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(labels?)
    }
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
        // The smudge digit rows are re-targeted to `P^r / 2` (the weights of the odd digits), and
        // the halved weights leave `2^-1` in every coefficient of the decoded row; removing it
        // from every coefficient centers the smudge (`mxx_exponent_lut::decoder::decoding`).
        let weights = smudge_weight_family(self.parameters, &ring, &modulus)?;
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
            let diagonal = target.clone().block_diagonal(blocks);
            Ok::<_, Dio2Error>(
                self.chain_parameters.retarget_encoding(source, form, level, diagonal, sliced)?,
            )
        };
        let component_one = |encoding: Encoding| (encoding.0.1, (encoding.1.0.1, encoding.1.1.1));
        let smudge = |first: IntExpr, blocks: usize| -> Result<(Mat, (Mat, Mat)), Dio2Error> {
            let zero = ring.zero((1, blocks));
            let mut failure = None;
            let total = mxx_dsl::iterate(
                self.parameters.smudge_digits,
                (zero.clone(), (zero.clone(), zero)),
                |index, total| {
                    let offset = if self.plan.noise_targets == NoiseTargets::Full {
                        let columns = if outs.private.is_none() {
                            1
                        } else {
                            self.plan.smudges() * self.plan.columns
                        };
                        index.expression()? * IntExpr::constant(columns)
                    } else {
                        IntExpr::constant(0)
                    };
                    let weight = weights.at(index);
                    let digit = blocks_of(outs.smudge, first + offset, blocks, &weight)
                        .map(component_one)
                        .map_err(|error| {
                            failure = Some(error);
                            mxx_dsl::DslError::Schema
                        })?;
                    Ok((total.0 + digit.0, (total.1.0 + digit.1.0, total.1.1 + digit.1.1)))
                },
            );
            if let Some(error) = failure {
                return Err(error);
            }
            Ok(total?)
        };
        if outs.private.is_none() {
            // The last step: the output decoder of the state modulo the first prime `q`, at the
            // decoding level, in the scalar form `Q_q`.
            let q = self.parameters.primes[0];
            let cofactor = self.chain_parameters.scalar(level, BigInt::from(&modulus / q))?;
            let selected = self.retarget(&workload, outputs, outs.state, cofactor)?;
            let smudge = smudge(IntExpr::constant(0), 1)?;
            let row = selected.0.1 + smudge.0;
            let key = (selected.1.0.1 + smudge.1.0, selected.1.1.1 + smudge.1.1);
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
        let mut rows = Vec::with_capacity(2 * self.plan.targets.len());
        let mut key_tops = Vec::with_capacity(2 * self.plan.targets.len());
        let mut key_bottoms = Vec::with_capacity(2 * self.plan.targets.len());
        let mut a_primes = Vec::with_capacity(2 * self.plan.targets.len());
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
        let labels = self.labels(&mask)?;
        let label_count = self.plan.label_rows * (self.plan.label_width + 1);
        let parts = mxx_dsl::parallel(self.plan.blocks(), |part| {
            let target = part.clone().div(2usize);
            let component = part.rem(2usize);
            let label_index = target.clone().sub(3usize).rem(label_count);
            let label = labels.at(label_index.clone());
            let column = label_index.clone().rem(self.plan.label_width + 1);
            let row = label_index.clone().div(self.plan.label_width + 1);
            let label_key = mxx_dsl::select(
                column.clone().equal(0).to_int(),
                vec![
                    next_root.key(
                        next_hash_key.clone(),
                        RnsBggRoot::input_tag(
                            LABEL_TAG,
                            Some(row.clone().mul(self.plan.label_width).add(column).sub(1usize)),
                        ),
                    ),
                    next_root
                        .key(next_hash_key.clone(), RnsBggRoot::input_tag(PUBLIC_TAG, Some(row))),
                ],
            )?;
            let pick = |pair: &(Mat, Mat)| {
                mxx_dsl::select(component.clone(), vec![pair.0.clone(), pair.1.clone()])
            };
            let label_part = (
                pick(&label.0)?,
                pick(&label.1.0)?,
                pick(&label.1.1)?,
                mxx_dsl::select(
                    component.clone(),
                    vec![a_prime_block(label_key.clone(), 0), a_prime_block(label_key, 1)],
                )?
                .reduce_modulus(&ring),
            );
            let other = mxx_dsl::select(
                target.clone().less(3usize).to_int(),
                vec![target.clone().sub(label_count), target.clone()],
            )?
            .rem(self.plan.targets.len())
            .mul(2usize)
            .add(component);
            let scalar_part = (
                rows.at(other.clone()),
                key_tops.at(other.clone()),
                key_bottoms.at(other.clone()),
                a_primes.at(other),
            );
            let is_label = target.clone().less(3 + label_count) & Int::constant(2).less(target);
            mxx_dsl::select(is_label.to_int(), vec![scalar_part, label_part])
        })?;
        let rows = parts.field(|part| part.0)?;
        let key_tops = parts.field(|part| part.1)?;
        let key_bottoms = parts.field(|part| part.2)?;
        let a_primes = parts.field(|part| part.3)?;
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
            ring.all_coefficients((1, columns), IntExpr::constant((&modulus + 1u8) / 2u8));
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
            let smudge = smudge(
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
            )?;
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
                                prime,
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
                    Ok::<_, Dio2Error>(Int::constant(
                        (2 * self.plan.target_index(target) + port) as i64,
                    ))
                };
                for (name, port, group) in self.plan.input_groups() {
                    if let Some(count) = group.count(self.plan) {
                        builder.add_output(
                            name,
                            mxx_dsl::parallel(count, |member| {
                                Ok(refreshed.at(group
                                    .target_index(self.plan, member)
                                    .mul(2usize)
                                    .add(port)))
                            })?,
                        );
                    } else {
                        builder.add_output(
                            name,
                            refreshed.at(group
                                .target_index(self.plan, Int::constant(0))
                                .mul(2usize)
                                .add(port)),
                        );
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

/// A scalar-polynomial family of `P^r 2^-1 mod Q`, with one indexed body and no literal table.
fn smudge_weight_family(
    parameters: &Dio2Parameters,
    ring: &Ring,
    modulus: &BigUint,
) -> Result<mxx_dsl::Family<Mat>, mxx_dsl::DslError> {
    let half = ring.polynomial([IntExpr::constant((modulus + 1u8) / 2u8)]);
    mxx_dsl::parallel(parameters.smudge_digits, |index| {
        Ok(ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::PowerOfBase {
                base: IntExpr::constant(parameters.prf.output_modulus),
                exponent: index.expression()?,
            },
        ) * half)
    })
}

/// The decoder weights `P^r 2^-1 mod Q` for weight-one smudge digit rows, in digit order.
/// `Q` is the odd decoding modulus; both the composed rows/keys and their noise simulation
/// use these BigUint weights instead of putting powers of `P` in small-constant instructions.
pub(super) fn smudge_weights(parameters: &Dio2Parameters, modulus: &BigUint) -> Vec<BigUint> {
    let half = (modulus + 1u8) / 2u8;
    let base = BigUint::from(parameters.prf.output_modulus);
    let mut weight = half;
    (0..parameters.smudge_digits)
        .map(|_| {
            let current = weight.clone();
            weight = &weight * &base % modulus;
            current
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::{
        step::{ChainKeys, LabelProducts},
        test_support::toy,
    };
    use mxx_dsl::DslContext;

    #[test]
    fn test_symbolic_smudge_weights_match_modular_weights_without_node_growth() {
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend,
        };
        use mxx_exponent_lut::api::{StageExecution, StageGraph, execute_graph};
        use std::collections::BTreeMap;
        let (mut parameters, _) = toy();
        let root = parameters.root(rand::random()).unwrap();
        let chain = parameters.chain(&root).unwrap();
        let level = parameters.decoding_level();
        let ring = chain.ring(level).unwrap();
        let modulus = chain.level(level).unwrap().modulus().as_ref().clone();
        let mut execution = CpuExecution::new(cpu_backend(parameters.rings().unwrap()));
        let mut expected_nodes = None;
        for digits in [1, 3, 60, 306] {
            parameters.smudge_digits = digits;
            let weights = smudge_weight_family(&parameters, &ring, &modulus).unwrap();
            let coefficients =
                mxx_dsl::parallel(digits, |index| Ok(weights.at(index).coefficients().at(0)))
                    .unwrap();
            let graph = DslContext::new("dio2-smudge-weights")
                .output("weights", coefficients)
                .unwrap()
                .build()
                .unwrap();
            let nodes =
                graph.graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>();
            if let Some(expected) = expected_nodes {
                assert_eq!(nodes, expected);
            } else {
                expected_nodes = Some(nodes);
            }
            let output = execute_graph(
                StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &mxx_ir_core::ParamEnv::default(),
                &mut execution,
                &mut MemoryArtifactStore::default(),
                |_| Ok(()),
            )
            .unwrap();
            let actual = <CpuExecution as StageExecution<MemoryArtifactStore>>::download_integers(
                &execution,
                &output.outputs["weights"],
            )
            .unwrap();
            let expected = smudge_weights(&parameters, &modulus)
                .into_iter()
                .map(BigInt::from)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    /// Isolate the production label lowerer from the samplers: their flattened PRF selector
    /// counts can exceed `usize` even though the label forest itself is a bounded graph.
    fn label_forest(parameters: &Dio2Parameters) -> mxx_ir_core::Graph {
        let (small, function) = toy();
        let mut plan = StepPlan::new(&small, &function).unwrap();
        let keys = ChainKeys {
            sigma: None,
            cbd: None,
            smudge: small.prf.declared_shape(0, "output", plan.output_smudge_members()).unwrap(),
        };
        let mut program = ChainProgram::last(&small, &function, &plan, &keys).unwrap();
        program.label_products = Some(LabelProducts::new(parameters, function.eval_chunk_bits));
        plan.columns = parameters.columns().unwrap();
        plan.label_rows = parameters.label_rows();
        plan.label_width = parameters.label_width();
        plan.ring_dimension = parameters.ring_dimension as usize;
        let root = parameters.root(rand::random()).unwrap();
        let chain = parameters.chain(&root).unwrap();
        let compose = StepCompose {
            parameters,
            chain_parameters: &chain,
            plan: &plan,
            program: &program,
            next_root: None,
            mode: ComposeMode::Keys,
        };
        let row = root.ring().zero((1, plan.columns));
        let mask = ((row.clone(), row.clone()), ((row.clone(), row.clone()), (row.clone(), row)));
        let graph = DslContext::new("dio2-label-forest")
            .output("labels", compose.labels(&mask).unwrap())
            .unwrap()
            .build()
            .unwrap();
        mxx_ir_core::validate(&graph.graph, &mxx_ir_core::ParamEnv::default()).unwrap();
        graph.graph
    }

    /// Increasing label rows and columns adds tree levels and index bits, not scalar products.
    #[test]
    fn symbolic_label_forest_has_logarithmic_node_growth() {
        let (mut parameters, _) = toy();
        let counts = [1, 2, 8].map(|height| {
            parameters.prf.lwr_dimension = height;
            label_forest(&parameters)
                .scopes()
                .values()
                .map(|scope| scope.nodes().len())
                .sum::<usize>()
        });
        assert!(counts[2] < 4 * counts[0], "node counts: {counts:?}");
    }

    /// Graph-only label-family gate for the chosen production row, independently of the full
    /// smudge PRF's addressability. Run with `cargo test -p mxx-io --release --lib
    /// symbolic_production_label_forest -- --ignored --nocapture`. Override parameters with
    /// `MXX_IO_DIO2_LABEL_{RING_DIMENSION,PRIMES,CRT_BITS,BASE_BITS,LEVELS,PRF_NU}`.
    #[test]
    #[ignore = "production-sized label forest; graph-only, no execution"]
    fn symbolic_production_label_forest() {
        let read = |name: &str, default: usize| {
            std::env::var(format!("MXX_IO_DIO2_LABEL_{name}"))
                .map_or(default, |value| value.parse().unwrap())
        };
        let (mut parameters, _) = toy();
        parameters.ring_dimension = read("RING_DIMENSION", 65536) as u32;
        parameters.crt_bits = read("CRT_BITS", 60);
        parameters.base_bits = read("BASE_BITS", 6) as u32;
        parameters.primes = mxx_ir_core::generate_crt_basis(
            parameters.ring_dimension,
            read("PRIMES", 27),
            parameters.crt_bits,
        )
        .unwrap();
        parameters.levels = read("LEVELS", 38);
        parameters.prf.lwr_dimension = read("PRF_NU", 1024);
        parameters.prf.support_weight = 31;
        parameters.prf.input_modulus = 256;
        parameters.prf.output_modulus = 32;
        parameters.smudge_digits = 306;
        parameters.encoding_error_sigma = 4.into();
        let graph = label_forest(&parameters);
        let nodes = graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>();
        eprintln!("production label forest: {nodes} nodes, {} scopes", graph.scopes().len());
        assert!(nodes < 10000);
    }
}
