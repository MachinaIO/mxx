//! The refresh of a step's targets (spec §6.3–§6.8) and the output decoder of the last step
//! (spec §8), composed on the lowered outputs of a [`ChainProgram`] for either carrier
//! ([`super::carrier`]): the obfuscator composes the keys alone and samples the preimages of
//! the decoded keys, the evaluator composes rows and keys and decodes them. Both run the same
//! methods, so every key subexpression is built by the same helpers on the same arguments.
//!
//! The composition reads the lowered (unfinished) outputs: a private wire in the state form of
//! its level (a raw coefficient wire too), a public wire in the public form of its level. On the
//! decoding level `L` (of modulus `Q_0`, like every level) it forms
//!
//! - the new-mask rows `Sigma = (g, omega)` and `Sigma^(k) = (g^(k), omega)` in the public form of
//!   `L` (spec §4.3, §4.4): `g^(k)` is component 1 of the σ sampler's `Aut_k`, `omega` component 1
//!   of the one-encoding, each re-targeted by `Gamma` ([`Compose::mask`]);
//! - every target in the state form of `L` (spec §6.6), the labels as the public products of the
//!   label chains by `E^sigma` (spec §6.5, [`Compose::labels`]);
//! - for every part `(t, j)` and every prime `q` of `Q_0`, the decoded component `d = Sigma X - c_j
//!   Y + f Z + u` (spec §6.7), whose smudge `u` sums the weight-one smudge digit rows re-targeted
//!   to the scalar form `P^r / 2`. The CBD noise `f` of a part is shared by its primes, so the
//!   residues recombine to one row.
//!
//! [`Compose::decoded_keys`] gives the obfuscator the decoded key `[K_0; K_1]` of every part and
//! prime; [`Compose::next_rows`] unmasks every part and prime with the anchor hint and its
//! preimage, rounds it to `y_j mod q`, and sums the residues' CRT terms to the part's row of the
//! next step input modulo `Q_0`. Parts and primes are loops of the graph: its size does not
//! depend on the number of labels, primes or smudge digits.
//!
//! The output decoder ([`Compose::output_decoded`]) re-targets the state to the scalar form
//! `Q_q` of the first prime `q` and rounds the unmasked row to the state monomial modulo `q`.

use super::{
    Dio2Error, Dio2Parameters, Failure,
    carrier::{Carrier, Key, Row, block_columns},
    labels::LabelColumns,
    step::{ChainProgram, NoiseTargets, RootObject, StepPlan, StepRows},
};
use mxx_backends::poly::PolyParams;
use mxx_dsl::{Bytes, DslError, Family, HashTag, Int, Mat, Preimage, Ring};
use mxx_exponent_lut::{
    ExponentLutProgram,
    program::{AttributeKnowledge, Operand, WireId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggForm, RnsBggParameters, RnsBggRoot, refresh_combination,
        residue_term,
    },
};
use mxx_ir_core::{
    IntExpr,
    node::{ConcatAxis, ConstantMatrix},
};
use num_bigint::{BigInt, BigUint};

/// What a refresh needs of the next step: its root (the targets' keys `A'`), the root hash key
/// handle, and its anchor `B_(i+1)` (`2 x 2 (w_0 + 2)` at `Q_0`), which the refresh pads to
/// whole blocks with columns hashed under the next root (spec §4.7).
pub struct NextStep {
    pub root: RnsBggRoot,
    pub hash_key: Bytes,
    pub anchor: Mat,
}

/// The refresh targets of a step at `L` (spec §6.6) and what every part's decoder shares:
/// `g` and `omega` of `Sigma`, the scalar targets (private, the mask `E^sigma`, state), the
/// label and helper targets, the next step's scalar root keys and padded anchor.
pub struct Targets<C: Carrier> {
    g: Vec<Mat>,
    omega: Vec<Mat>,
    scalars: Family<C>,
    labels: Family<C>,
    helpers: Family<C>,
    scalar_keys: Family<Key>,
    anchor: Mat,
}

/// What a part decodes for every prime: its target component, its key block `A'_(o, (j))` and
/// its CBD noise `f`, component 1.
struct Part {
    index: IntExpr,
    target: Vec<Mat>,
    a_prime: Mat,
    fresh: Vec<Mat>,
}

/// The refresh (or output decoder) of one step on the lowered Program outputs `outputs`.
pub struct Compose<'a, C: Carrier> {
    parameters: &'a Dio2Parameters,
    chain: &'a RnsBggParameters,
    plan: &'a StepPlan,
    program: &'a ChainProgram,
    outputs: Vec<C>,
    workload: RnsBggDerivedWorkload,
}

impl<'a, C: Carrier> Compose<'a, C> {
    /// The composition of `program`'s lowered outputs `outputs` on the step chain `chain`.
    pub fn new(
        parameters: &'a Dio2Parameters,
        chain: &'a RnsBggParameters,
        plan: &'a StepPlan,
        program: &'a ChainProgram,
        outputs: Vec<C>,
    ) -> Result<Self, Dio2Error> {
        if plan.noise_targets != NoiseTargets::Full || !plan.symbolic_labels {
            return Err(Dio2Error::Input("a composition needs the production plan"));
        }
        let workload = RnsBggDerivedWorkload::from_program(&program.program)?;
        Ok(Self { parameters, chain, plan, program, outputs, workload })
    }

    /// The decoding level `L`.
    fn level(&self) -> usize {
        self.parameters.decoding_level()
    }

    /// The ring of `L` (of the modulus `Q_0`).
    fn ring(&self) -> Result<Ring, Dio2Error> {
        Ok(self.chain.ring(self.level())?)
    }

    /// The modulus `Q_0` of `L`.
    fn modulus(&self) -> Result<BigUint, Dio2Error> {
        Ok(self.chain.level(self.level())?.modulus().as_ref().clone())
    }

    /// The form `form` of `L`.
    fn form(&self, form: RnsBggForm) -> Result<Mat, Dio2Error> {
        Ok(self.chain.form(self.level(), form)?)
    }

    /// `Gamma` of output `output` to the form `target` (a row in the ring of `L`).
    fn retarget(&self, output: usize, target: Mat) -> Result<C, Dio2Error> {
        let source = carrier(&self.program.program, &self.workload, output)?;
        Ok(self.outputs[output].clone().retarget(self.chain, source, self.level(), target)?)
    }

    /// `Gamma` of the `blocks` carriers of output `output` from carrier `first` (an index
    /// expression) on to `blocks` copies of `target`: those columns of the row and the key,
    /// acted on by the block-diagonal digits of `target` (`I_blocks (x) target`).
    fn blocks(
        &self,
        output: usize,
        first: IntExpr,
        blocks: usize,
        target: &Mat,
    ) -> Result<C, Dio2Error> {
        let source = carrier(&self.program.program, &self.workload, output)?;
        let width = self.chain.component_columns(source.0)?;
        let sliced = self.outputs[output].slice_columns(block_columns(first, blocks, width));
        Ok(sliced.retarget(
            self.chain,
            source,
            self.level(),
            target.clone().block_diagonal(blocks),
        )?)
    }

    /// Component 1 of the sum of the smudge digit rows of `blocks` smudges from smudge `first`:
    /// digit `r` (columns `r s ..` of the packed row of all `s` smudges) re-targeted to the
    /// scalar form `P^r / 2` (see [`ChainProgram`]), in one loop over the digits.
    fn smudge(&self, first: IntExpr, blocks: usize) -> Result<Vec<Mat>, Dio2Error> {
        let ring = self.ring()?;
        let weights = smudge_weight_family(self.parameters, &ring, &self.modulus()?)?;
        let smudges = if self.program.outputs.private.is_none() {
            1
        } else {
            self.plan.smudges() * self.plan.columns
        };
        let mut failure = None;
        let total = mxx_dsl::iterate(
            self.parameters.smudge_digits,
            C::zero(&ring, blocks).component(1),
            |digit, total| {
                let first = first + digit.expression()? * IntExpr::constant(smudges);
                let row = self
                    .blocks(self.program.outputs.smudge, first, blocks, &weights.at(digit))
                    .map_err(|error| {
                        failure = Some(error);
                        DslError::Schema
                    })?;
                Ok(add(total, row.component(1)))
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(total?)
    }

    /// Component 1 of output `output` in the public form of `L`.
    fn public_component(&self, output: usize) -> Result<Vec<Mat>, Dio2Error> {
        Ok(self.retarget(output, self.form(RnsBggForm::Public)?)?.component(1))
    }

    /// `Gamma` of a carrier in the public form of `L` to the state form of `L`.
    fn to_state(&self, carrier: C) -> Result<C, Dio2Error> {
        let level = self.level();
        Ok(carrier.retarget(
            self.chain,
            (level, RnsBggForm::Public),
            level,
            self.form(RnsBggForm::State)?,
        )?)
    }

    /// The new-mask row `E^sigma` of `Sigma = (g, omega)` in the state form of `L` (spec §4.3,
    /// §6.3), with `g` and `omega` (in the public form of `L`), and the helper targets
    /// `Gamma(Sigma^(k))` of every automorphism `k` of the plan, in its order (`g^(1) = g`).
    fn mask(&self) -> Result<(Vec<Mat>, Vec<Mat>, C, Family<C>), Dio2Error> {
        let outputs = &self.program.outputs;
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let omega = self.public_component(outputs.public.ok_or_else(missing)?)?;
        let g = self.public_component(outputs.sigma.ok_or_else(missing)?)?;
        let mask = self.to_state(C::from_components(g.clone(), omega.clone()))?;
        let helpers = self
            .plan
            .automorphisms
            .iter()
            .map(|sigma| {
                let output = match sigma {
                    1 => outputs.sigma.ok_or_else(missing)?,
                    sigma => *outputs.automorphed.get(sigma).ok_or_else(missing)?,
                };
                self.to_state(C::from_components(self.public_component(output)?, omega.clone()))
            })
            .collect::<Result<Vec<_>, Dio2Error>>()?;
        Ok((g, omega, mask, Family::pack(helpers)?))
    }

    /// Every label target `N[r, c] = MulPub(Gamma(LBL'[r, c]), E^sigma)` (spec §6.5), member
    /// `r (m_v + 1) + c`: the label chains of chunk `chunk` (of the label leaves `leaf` and the
    /// one-encoding `one` at level 0, see [`super::labels::LabelProducts::forest`]) re-targeted
    /// to the public form of `L` and multiplied by the mask `mask`.
    fn labels(
        &self,
        (columns, chunk): (&LabelColumns, Int),
        leaf: impl Fn(Int, Int) -> Result<C::Public, Dio2Error>,
        one: C::Public,
        mask: &C,
    ) -> Result<Family<C>, Dio2Error> {
        let products = self
            .program
            .label_products
            .as_ref()
            .ok_or(Dio2Error::Input("a refreshing Program has label products"))?;
        let chains =
            products.forest::<C>(self.chain, self.parameters, columns, chunk, leaf, one)?;
        let level = self.level();
        let form = self.form(RnsBggForm::Public)?;
        let count = self.plan.label_rows * (self.plan.label_width + 1);
        let mut failure = None;
        let labels = mxx_dsl::parallel(count, |member| {
            C::retarget_public(
                chains.at(member),
                self.chain,
                (0, RnsBggForm::Public),
                level,
                form.clone(),
            )
            .and_then(|label| mask.mul_public(self.chain, level, &label))
            .map_err(|error| {
                failure = Some(Dio2Error::from(error));
                DslError::Schema
            })
        });
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(labels?)
    }

    /// The refresh targets of chunk `chunk` (its label transition `columns`) for the next step
    /// `next`, the label leaves being `leaf` and the one-encoding `one`.
    fn targets(
        &self,
        next: &NextStep,
        label_chunk: (&LabelColumns, Int),
        leaf: impl Fn(Int, Int) -> Result<C::Public, Dio2Error>,
        one: C::Public,
    ) -> Result<Targets<C>, Dio2Error> {
        let (g, omega, mask, helpers) = self.mask()?;
        let labels = self.labels(label_chunk, leaf, one, &mask)?;
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let state_form = self.form(RnsBggForm::State)?;
        let outputs = &self.program.outputs;
        let private = self.retarget(outputs.private.ok_or_else(missing)?, state_form.clone())?;
        let state = self.retarget(outputs.state, state_form)?;
        let key =
            |object, member| self.plan.root_key(&next.root, next.hash_key.clone(), object, member);
        let scalar_keys = Family::pack(vec![
            key(RootObject::Private, None)?,
            key(RootObject::Public, Some(Int::constant(self.plan.label_rows)))?,
            key(RootObject::State, None)?,
        ])?;
        Ok(Targets {
            g,
            omega,
            scalars: Family::pack(vec![private, mask, state])?,
            labels,
            helpers,
            scalar_keys,
            anchor: self.padded_anchor(next)?,
        })
    }

    /// Part `part`: target `t = part / 2` in the order private, public, state, labels,
    /// helpers, anchor blocks ([`StepPlan::full_target_count`]), component `j = part mod 2`
    /// (spec §6.6, §6.7).
    fn part(&self, part: &Int, next: &NextStep, targets: &Targets<C>) -> Result<Part, Dio2Error> {
        let plan = self.plan;
        let labels = plan.label_rows * (plan.label_width + 1);
        let helpers = plan.automorphisms.len();
        let (first_label, first_helper, first_anchor) = (3, 3 + labels, 3 + labels + helpers);
        let target = part.clone().div(2usize);
        let component = part.clone().rem(2usize);
        // 3 for a scalar target, 2 for a label, 1 for a helper, 0 for an anchor block.
        let class = [first_label, first_helper, first_anchor]
            .into_iter()
            .map(|end| target.clone().less(end).to_int())
            .reduce(Int::add)
            .expect("three boundaries");
        let label = wrap(&target, first_label, labels);
        let helper = wrap(&target, first_helper, helpers);
        let anchor = wrap(&target, first_anchor, self.anchor_blocks());
        let scalar = target.rem(3usize);
        let ring = self.ring()?;
        let carrier = mxx_dsl::select(
            class.clone(),
            vec![
                C::zero(&ring, plan.columns),
                targets.helpers.at(helper.clone()),
                targets.labels.at(label.clone()),
                targets.scalars.at(scalar.clone()),
            ],
        )?;
        let target =
            mxx_dsl::select(component.clone(), vec![carrier.component(0), carrier.component(1)])?;
        let block = |key: &Key| {
            mxx_dsl::select(component.clone(), vec![a_prime_block(key, 0), a_prime_block(key, 1)])
        };
        let key = |object, member| -> Result<Mat, Dio2Error> {
            Ok(block(&plan.root_key(&next.root, next.hash_key.clone(), object, member)?)?)
        };
        // Label `(r, c)` becomes member `r` of label column 0, or member `r m_v + c - 1` of the
        // other label columns.
        let row = label.clone().div(plan.label_width + 1);
        let column = label.rem(plan.label_width + 1);
        let other_member =
            row.clone().mul(plan.label_width).add(wrap(&column, 1, plan.label_width));
        let label_key = mxx_dsl::select(
            column.equal(0).to_int(),
            vec![key(RootObject::Label, Some(other_member))?, key(RootObject::Public, Some(row))?],
        )?;
        let anchor_block = anchor.mul(2usize).add(component.clone()).expression()?;
        let anchor_key =
            targets.anchor.clone().slice(None, Some(block_columns(anchor_block, 1, plan.columns)));
        let a_prime = mxx_dsl::select(
            class,
            vec![
                anchor_key,
                key(RootObject::Helper, Some(helper))?,
                label_key,
                block(&targets.scalar_keys.at(scalar))?,
            ],
        )?;
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let index = part.expression()?;
        let fresh = self
            .blocks(
                self.program.outputs.cbd.ok_or_else(missing)?,
                IntExpr::Mul(Box::new(index.clone()), Box::new(IntExpr::constant(plan.columns)))
                    .canonicalize(),
                plan.columns,
                &self.form(RnsBggForm::Public)?,
            )?
            .component(1);
        Ok(Part { index, target, a_prime: a_prime.reduce_modulus(&ring), fresh })
    }

    /// The anchor blocks of the targets: `ceil(2 (w_0 + 2) / (2 w_0))`.
    fn anchor_blocks(&self) -> usize {
        super::anchor_columns(self.plan.columns).div_ceil(2 * self.plan.columns)
    }

    /// `B_(i+1)` padded to whole refresh blocks with columns hashed under the next root.
    fn padded_anchor(&self, next: &NextStep) -> Result<Mat, Dio2Error> {
        let columns = super::anchor_columns(self.plan.columns);
        let padding = 2 * self.plan.columns * self.anchor_blocks() - columns;
        let tag = next.root.domain_tag(HashTag::from(b"dio2/anchor-padding".to_vec()));
        let padding = next.root.ring().hash_matrix(next.hash_key.clone(), tag, (2, padding));
        Ok(Mat::concat(ConcatAxis::Columns, vec![next.anchor.clone(), padding]))
    }

    /// The decoded entries ([`Carrier::DECODED`]) of part `part` for the prime of index `prime`
    /// in the basis of `Q_0` (spec §6.7): `d = [g | omega] X - c_j Y + f Z + u` of each
    /// decoded entry, with the refresh actions of the part's `A'` and the prime and the
    /// smudge `u` of the part and the prime (smudge `p |P| + k`).
    fn decoded(
        &self,
        part: &Part,
        prime: &Int,
        targets: &Targets<C>,
    ) -> Result<Vec<Mat>, Dio2Error> {
        let columns = self.plan.columns;
        let smudge = IntExpr::Mul(
            Box::new(IntExpr::Add(
                Box::new(IntExpr::Mul(
                    Box::new(part.index.clone()),
                    Box::new(IntExpr::constant(self.plan.primes)),
                )),
                Box::new(prime.expression()?),
            )),
            Box::new(IntExpr::constant(columns)),
        )
        .canonicalize();
        let smudge = self.smudge(smudge, columns)?;
        let actions = self.chain.refresh_actions(
            self.level(),
            part.a_prime.clone(),
            prime.clone(),
            &self.parameters.reference(),
        )?;
        C::DECODED
            .iter()
            .map(|entry| {
                Ok(refresh_combination(
                    (targets.g[*entry].clone(), targets.omega[*entry].clone()),
                    part.target[*entry].clone(),
                    (part.fresh[*entry].clone(), columns),
                    &actions,
                )? + smudge[*entry].clone())
            })
            .collect()
    }

    /// The decoded entries of the output decoder: the state re-targeted to the scalar form of
    /// the cofactor `Q_0 / q` of the first prime `q`, plus its smudge (spec §8).
    fn output_decoded(&self) -> Result<Vec<Mat>, Dio2Error> {
        let q = self.parameters.primes[0];
        let cofactor = self.chain.scalar(self.level(), BigInt::from(&self.modulus()? / q))?;
        let state = self.retarget(self.program.outputs.state, cofactor)?.component(1);
        let smudge = self.smudge(IntExpr::constant(0), 1)?;
        Ok(C::DECODED.iter().map(|entry| state[*entry].clone() + smudge[*entry].clone()).collect())
    }
}

impl Compose<'_, Key> {
    /// The decoded key `[K_0; K_1]` (`2 x w_0` at `Q_0`) of every part and prime of the
    /// refresh to the next step `next` on chunk `chunk` (its label transition `columns`),
    /// member `p |P| + k` for part `p` and the prime of index `k` in the basis of `Q_0`: what
    /// the anchor's preimages are sampled for. The label leaves are the step root's keys.
    pub fn decoded_keys(
        &self,
        next: &NextStep,
        label_chunk: (&LabelColumns, Int),
    ) -> Result<Family<Mat>, Dio2Error> {
        let plan = self.plan;
        let root = self.chain.root().ok_or(Dio2Error::Input("the step chain has a root"))?;
        let hash_key = root.hash_key_input();
        let leaf = |input: Int, _| {
            Ok(plan.root_key(root, hash_key.clone(), RootObject::Label, Some(input))?)
        };
        let one = plan.root_key(
            root,
            hash_key.clone(),
            RootObject::Public,
            Some(Int::constant(plan.label_rows)),
        )?;
        let targets = self.targets(next, label_chunk, leaf, one)?;
        let failure = Failure::default();
        let keys = mxx_dsl::parallel(plan.blocks(), |part| {
            let part = failure.keep(self.part(&part, next, &targets))?;
            mxx_dsl::parallel(plan.primes, |prime| {
                failure.keep(self.decoded(&part, &prime, &targets).map(key_rows))
            })
        });
        let keys = failure.finish(keys)?;
        let primes = plan.primes;
        Ok(mxx_dsl::parallel(plan.blocks() * primes, |member| {
            Ok(keys.at(member.clone().div(primes)).at(member.rem(primes)))
        })?)
    }

    /// The decoded key `[K_0; K_1]` (`2 x 1`) of the output decoder.
    pub fn output_key(&self) -> Result<Mat, Dio2Error> {
        Ok(key_rows(self.output_decoded()?))
    }
}

impl Compose<'_, Row> {
    /// The next step input `S_(i+1)` (spec §6.8) of the step input `rows` with the clear labels
    /// `labels`, refreshed to the next step `next` on chunk `chunk` (its label transition
    /// `columns`): every part's row, decoded for every prime with the hint `rows.hint` and its
    /// preimage `preimages.at(p |P| + k)`, recombined from the residues modulo the primes. The
    /// label leaves are the label rows with their public payloads `mono(stride a)`.
    pub fn next_rows(
        &self,
        next: &NextStep,
        (columns, chunk, labels): (&LabelColumns, Int, &Family<Int>),
        rows: &StepRows,
        preimages: &Family<Preimage>,
    ) -> Result<StepRows, Dio2Error> {
        let plan = self.plan;
        let ring = self.ring()?;
        let stride = self.parameters.profile()?.layout.stride();
        let root = self.chain.root().ok_or(Dio2Error::Input("the step chain has a root"))?;
        let hash_key = root.hash_key_input();
        let leaf = |input: Int, value: Int| {
            let key =
                plan.root_key(root, hash_key.clone(), RootObject::Label, Some(input.clone()))?;
            let payload = ring.identity(1).multiply_monomial(labels.at(value).mul(stride));
            Ok((rows.labels.at(input), key, payload))
        };
        let one_member = Int::constant(plan.label_rows);
        let one = (
            rows.public.at(one_member.clone()),
            plan.root_key(root, hash_key.clone(), RootObject::Public, Some(one_member))?,
            ring.identity(1),
        );
        let targets = self.targets(next, (columns, chunk), leaf, one)?;
        let basis = self.chain.level(self.level())?.to_crt().0;
        let half = ring
            .all_coefficients((1, plan.columns), IntExpr::constant((&self.modulus()? + 1u8) / 2u8));
        let failure = Failure::default();
        let refreshed = mxx_dsl::parallel(plan.blocks(), |index| {
            let part = failure.keep(self.part(&index, next, &targets))?;
            mxx_dsl::iterate(basis.len(), ring.zero((1, plan.columns)), |prime, sum| {
                let row = failure.keep(self.decoded(&part, &prime, &targets))?.swap_remove(0);
                let preimage = preimages.at(index.clone().mul(basis.len()).add(prime.clone()));
                let unmasked = unmask(&ring, &rows.hint, preimage, row) - half.clone();
                let bound = prime.expression()?;
                let residue = unmasked.modulus_switch(&ring.slice_crt(bound.clone(), bound + 1));
                let term = residue_term(&basis, prime, self.parameters.ring_dimension, residue);
                Ok(sum + failure.keep(term.map_err(Dio2Error::from))?)
            })
        });
        self.step_rows(&failure.finish(refreshed)?)
    }

    /// The next step input from the refreshed row of every part (member `2 t + j` for target
    /// `t` and component `j`), by the target each root object becomes (spec §6.6).
    fn step_rows(&self, refreshed: &Family<Mat>) -> Result<StepRows, Dio2Error> {
        let plan = self.plan;
        let width = plan.label_width;
        let row = |target: Int| {
            let part = target.mul(2usize);
            (refreshed.at(part.clone()), refreshed.at(part.add(1usize)))
        };
        let public = mxx_dsl::parallel(plan.label_rows + 1, |member| {
            // Member `r < v` is label `(r, 0)`; member `v` is `ONEPUB`, target 1.
            let label = member.clone().mul(width + 1).add(3usize);
            Ok(row(mxx_dsl::select(
                member.equal(plan.label_rows).to_int(),
                vec![label, Int::constant(1)],
            )?))
        })?;
        let labels = mxx_dsl::parallel(plan.label_rows * width, |member| {
            Ok(row(member.clone().div(width).mul(width + 1).add(member.rem(width)).add(4usize)))
        })?;
        let first_helper = 3 + plan.label_rows * (width + 1);
        let helpers = mxx_dsl::parallel(plan.automorphisms.len(), |member| {
            Ok(row(member.add(first_helper)))
        })?;
        // The hint: the anchor blocks side by side, cut to the hint's width.
        let first_anchor = first_helper + plan.automorphisms.len();
        let blocks = (0..2 * self.anchor_blocks())
            .map(|part| refreshed.at(Int::constant(2 * first_anchor + part)))
            .collect::<Vec<_>>();
        let hint = Mat::concat(ConcatAxis::Columns, blocks).slice(
            None,
            Some(block_columns(IntExpr::constant(0), super::anchor_columns(plan.columns), 1)),
        );
        Ok(StepRows {
            private: row(Int::constant(0)),
            state: row(Int::constant(2)),
            public,
            labels,
            helpers,
            hint,
        })
    }

    /// The plaintext `mono(z) mod q` of the output decoder (`1 x 1` in the ring of the first
    /// prime `q`), unmasked with the hint `hint` and the preimage `preimage`.
    pub fn output_plaintext(&self, hint: &Mat, preimage: Preimage) -> Result<Mat, Dio2Error> {
        let ring = self.ring()?;
        let row = self.output_decoded()?.swap_remove(0);
        let half = ring.all_coefficients((1, 1), IntExpr::constant((&self.modulus()? + 1u8) / 2u8));
        let unmasked = unmask(&ring, hint, preimage, row) - half;
        let q = self.parameters.primes[0];
        let plaintext =
            Ring::from_crt_moduli(vec![IntExpr::from(q)], self.parameters.ring_dimension);
        Ok(unmasked.modulus_switch(&plaintext))
    }
}

/// `target - offset` wrapped into `[0, count)` for a nonnegative target index: the member of a
/// target class, valid for the targets of that class (a selected-away candidate needs only an
/// in-range index).
fn wrap(target: &Int, offset: usize, count: usize) -> Int {
    let lift = count * (offset / count + 1) - offset;
    target.clone().add(lift).rem(count)
}

/// The sum of two components, entry by entry.
fn add(left: Vec<Mat>, right: Vec<Mat>) -> Vec<Mat> {
    left.into_iter().zip(right).map(|(left, right)| left + right).collect()
}

/// The decoded key rows `[K_0; K_1]` of decoded key entries.
fn key_rows(decoded: Vec<Mat>) -> Mat {
    Mat::concat(ConcatAxis::Rows, decoded)
}

/// `A'_(j) = [K_(0j); K_(1j)]` of a root key, `2 x w_0`.
fn a_prime_block(key: &Key, component: usize) -> Mat {
    let (top, bottom) = match component {
        0 => (key.0.0.clone(), key.1.0.clone()),
        _ => (key.0.1.clone(), key.1.1.clone()),
    };
    Mat::concat(ConcatAxis::Rows, vec![top, bottom])
}

/// `b K_pre - row` in the decoding ring `ring`: the hint `hint` (at `Q_0`) times the preimage
/// `preimage` (`2 (w_0 + 2) x columns` at `Q_0`), reduced to `ring`.
fn unmask(ring: &Ring, hint: &Mat, preimage: Preimage, row: Mat) -> Mat {
    hint.clone().mul_small_rhs(preimage).reduce_modulus(ring) - row
}

/// A scalar-polynomial family of `P^r 2^-1 mod Q`, with one indexed body and no literal table.
fn smudge_weight_family(
    parameters: &Dio2Parameters,
    ring: &Ring,
    modulus: &BigUint,
) -> Result<Family<Mat>, DslError> {
    let half = ring.polynomial([IntExpr::constant((modulus + 1u8) / 2u8)]);
    mxx_dsl::parallel(parameters.smudge_digits, |index| {
        Ok(ring.constant(
            (1, 1),
            ConstantMatrix::PowerOfBase {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;
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
}
