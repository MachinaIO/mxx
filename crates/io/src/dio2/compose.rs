//! The refresh of a step's targets (spec §6.3–§6.8) and the output decoder of the last step
//! (spec §8), composed on the lowered outputs of a [`ChainProgram`] for either carrier
//! ([`super::carrier`]): the obfuscator composes the keys alone and samples the preimages of
//! the decoded keys, the evaluator composes rows and keys and decodes them. Both run the same
//! methods, so every key subexpression is built by the same helpers on the same arguments.
//!
//! The composition reads the lowered (unfinished) outputs: a private wire in the state form of
//! its level (a raw coefficient wire too), a public wire in the public form of its level. The
//! CBD row and the smudge row ([`ChainProgram::deferred_outputs`]) are read by column blocks
//! ([`Blocks`]): the Program's families are lowered where their members are read
//! (`mxx_exponent_lut::api::ProgramBuild::lower_deferred`), so every PRF member a part needs is
//! computed in the loop of the part that consumes it, and no family of PRF members and no row
//! over every part is a graph value. On the decoding level `L` (of modulus `Q_0`, like every
//! level) it forms
//!
//! - the new-mask rows `Sigma = (g, omega)` and `Sigma^(k) = (g^(k), omega)` in the public form of
//!   `L` (spec §4.3, §4.4): `g^(k)` is component 1 of the σ sampler's `Aut_k`, `omega` component 1
//!   of the one-encoding, each re-targeted by `Gamma` ([`Compose::mask`]);
//! - every target in the state form of `L` (spec §6.6); a label target is the public product of its
//!   label chain by `E^sigma` (spec §6.5), formed by the part that decodes it;
//! - for every part `(t, j)` and every prime `q` of `Q_0`, the decoded component `d = Sigma X - c_j
//!   Y + f Z + u` (spec §6.7). The CBD noise `f` of a part is its `w_0` CBD blocks, each
//!   re-targeted to the public form of `L` on its own and placed in its block of the row; it is
//!   shared by the part's primes, so the residues recombine to one row. Column `c` of the smudge
//!   `u` sums the weight-one smudge digit blocks of the column re-targeted to the scalar form `P^r
//!   / 2`, one loop over the digits per column; no block-diagonal action over several blocks is
//!   formed.
//!
//! [`Compose::decoded_keys`] hands the obfuscator the decoded key `[K_0; K_1]` of every part and
//! prime in the loop that computes it; [`Compose::next_rows`] unmasks every part and prime with the
//! anchor hint and its preimage (read by its index in the obfuscation's preimage family), takes
//! the CRT term of its rounding modulo the prime, and sums the terms to the part's row of the
//! next step input modulo `Q_0`. Each root object's family of the next step input is one loop
//! over its members, each decoding its target's two parts. Parts, primes, columns, digits and
//! label leaves are loops of the graph: its size does not depend on the number of labels,
//! primes or smudge digits.
//!
//! The output decoder ([`Compose::output_decoded`]) re-targets the state to the scalar form
//! `Q_q` of the first prime `q` and rounds the unmasked row to the state monomial modulo `q`.

use super::{
    Dio2Error, Dio2Parameters, Failure,
    carrier::{Carrier, Key, Row},
    labels::LabelColumns,
    step::{ChainProgram, NoiseTargets, RootObject, StepPlan, StepRows},
};
use mxx_backends::poly::PolyParams;
use mxx_dsl::{Bytes, Family, HashTag, Int, Mat, Preimage, Ring};
use mxx_exponent_lut::{
    ExponentLutProgram,
    program::{AttributeKnowledge, DeferredColumns, LoweredOutput, Operand, WireId},
    rns_bgg::{
        RefreshActions, RnsBggDerivedWorkload, RnsBggForm, RnsBggParameters, RnsBggRoot,
        decoded_residue_term, place_in_column, refresh_combination, refresh_fresh_term,
    },
};
use mxx_ir_core::{
    IntExpr,
    node::{ConcatAxis, ConstantMatrix},
};
use num_bigint::{BigInt, BigUint};
use std::collections::BTreeMap;

/// What a refresh needs of the next step: its root (the targets' keys `A'`), the root hash key
/// handle, and its anchor `B_(i+1)` (`2 x 2 (w_0 + 2)` at `Q_0`), which the refresh pads to
/// whole blocks with columns hashed under the next root (spec §4.7).
pub struct NextStep {
    pub root: RnsBggRoot,
    pub hash_key: Bytes,
    pub anchor: Mat,
}

/// Column block `block` (an index expression) of the deferred Program output `output`, lowered
/// where it is read (`mxx_exponent_lut::api::ProgramBuild::deferred_block`): the CBD row's and
/// the smudge row's blocks, whose PRF members are computed in the loop that reads the block.
pub type Blocks<'a, C> = Box<dyn Fn(usize, IntExpr) -> Result<C, Dio2Error> + 'a>;

/// The refresh targets of a step at `L` (spec §6.6) and what every part's decoder shares:
/// `g` and `omega` of `Sigma`, the scalar targets (private, the mask `E^sigma`, state), the
/// helper targets, the label target of a label member, the next step's scalar root keys and
/// padded anchor. A part computes its own label target where it decodes it, so no family of
/// label targets is held.
pub struct Targets<'t, C: Carrier> {
    g: Vec<Mat>,
    omega: Vec<Mat>,
    scalars: Family<C>,
    helpers: Box<dyn Fn(Int) -> Result<C, Dio2Error> + 't>,
    label: Box<dyn Fn(Int) -> Result<C, Dio2Error> + 't>,
    scalar_keys: Family<Key>,
    anchor: Mat,
}

/// What a part decodes for every prime: its index, its target component and its key block
/// `A'_(o, (j))`.
struct Part {
    index: IntExpr,
    target: Vec<Mat>,
    a_prime: Mat,
}

/// The refresh (or output decoder) of one step on the lowered Program outputs `outputs` (`None`
/// for a deferred output) and the column blocks `blocks` of the deferred outputs.
pub struct Compose<'a, C: Carrier> {
    parameters: &'a Dio2Parameters,
    chain: &'a RnsBggParameters,
    plan: &'a StepPlan,
    program: &'a ChainProgram,
    outputs: Vec<Option<C>>,
    blocks: Blocks<'a, C>,
    workload: RnsBggDerivedWorkload,
}

impl<'a, C: Carrier> Compose<'a, C> {
    /// The composition of `program`'s lowered outputs `outputs` (`None` at the positions of
    /// [`ChainProgram::deferred_outputs`]) and deferred column blocks `blocks` on the step
    /// chain `chain`.
    pub fn new(
        parameters: &'a Dio2Parameters,
        chain: &'a RnsBggParameters,
        plan: &'a StepPlan,
        program: &'a ChainProgram,
        outputs: Vec<Option<C>>,
        blocks: Blocks<'a, C>,
    ) -> Result<Self, Dio2Error> {
        if plan.noise_targets != NoiseTargets::Full || !plan.symbolic_labels {
            return Err(Dio2Error::Input("a composition needs the production plan"));
        }
        let workload = RnsBggDerivedWorkload::from_program(&program.program)?;
        Ok(Self { parameters, chain, plan, program, outputs, blocks, workload })
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
        let value = self.outputs[output]
            .clone()
            .ok_or(Dio2Error::Input("a deferred output is read only by its blocks"))?;
        Ok(value.retarget(self.chain, source, self.level(), target)?)
    }

    /// `Gamma` of column block `block` of the deferred output `output` to the form `target`: a
    /// block alone, so no block-diagonal action over several blocks is formed.
    fn block(&self, output: usize, block: IntExpr, target: Mat) -> Result<C, Dio2Error> {
        let source = carrier(&self.program.program, &self.workload, output)?;
        Ok((self.blocks)(output, block.canonicalize())?.retarget(
            self.chain,
            source,
            self.level(),
            target,
        )?)
    }

    /// Component 1 of the row of `columns` columns whose column `c` is `column(c)` (component
    /// entries of one column each): one loop over the columns placing each in its column, so the
    /// graph does not grow with `columns`.
    fn columns(
        &self,
        columns: usize,
        column: impl Fn(IntExpr) -> Result<Vec<Mat>, Dio2Error>,
    ) -> Result<Vec<Mat>, Dio2Error> {
        let placed = |index: IntExpr| -> Result<Vec<Mat>, Dio2Error> {
            if columns == 1 {
                return column(index);
            }
            Ok(column(index.clone())?
                .into_iter()
                .map(|entry| place_in_column(entry, columns, &index))
                .collect())
        };
        parallel_sum(columns, |index| placed(index.expression()?))
    }

    /// Component 1 of the smudge of decoder `smudge` (`p |P| + k` for part `p` and the prime of
    /// index `k`, `None` for the output decoder's one column): column `c` is the sum over the
    /// digits `r` of the weight-one digit block of the column re-targeted to the scalar form
    /// `P^r / 2` (see [`ChainProgram`]), one loop over the digits per column.
    fn smudge(&self, smudge: Option<IntExpr>) -> Result<Vec<Mat>, Dio2Error> {
        let ring = self.ring()?;
        let modulus = self.modulus()?;
        let (columns, smudge_columns, first) = match smudge {
            Some(smudge) => (
                self.plan.columns,
                self.plan.smudges() * self.plan.columns,
                IntExpr::Mul(Box::new(smudge), Box::new(IntExpr::constant(self.plan.columns))),
            ),
            None => (1, 1, IntExpr::constant(0)),
        };
        let output = self.program.outputs.smudge;
        self.columns(columns, |column| {
            let column = IntExpr::Add(Box::new(first.clone()), Box::new(column)).canonicalize();
            if self.parameters.smudge_digits == 0 {
                return Ok(C::zero(&ring, 1).component(1));
            }
            parallel_sum(self.parameters.smudge_digits, |digit| {
                let digit = digit.expression()?;
                let block = IntExpr::Add(
                    Box::new(IntExpr::Mul(
                        Box::new(digit.clone()),
                        Box::new(IntExpr::constant(smudge_columns)),
                    )),
                    Box::new(column.clone()),
                );
                let weight = smudge_weight(self.parameters, &ring, &modulus, digit);
                let row = self.block(output, block, weight)?;
                Ok(row.component(1))
            })
        })
    }

    /// Component 1 of the fresh term `f Z = [f_k z]_k` of part `part` for the one-column action
    /// `z` of a prime's `actions`: CBD block `k` of the part (block `p w_0 + k` of the CBD row)
    /// re-targeted to the public form of `L` on its own, times `z`, placed in column `k` (one
    /// loop over the blocks, [`refresh_fresh_term`]), so neither the part's row of `w_0` blocks
    /// nor any block-diagonal action is formed.
    fn fresh(&self, part: &IntExpr, actions: &RefreshActions) -> Result<Vec<Mat>, Dio2Error> {
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let output = self.program.outputs.cbd.ok_or_else(missing)?;
        let form = self.form(RnsBggForm::Public)?;
        let first =
            IntExpr::Mul(Box::new(part.clone()), Box::new(IntExpr::constant(self.plan.columns)));
        let failure = std::cell::RefCell::new(None);
        let term = refresh_fresh_term(self.plan.columns, actions, |block| {
            let block = IntExpr::Add(Box::new(first.clone()), Box::new(block.clone()));
            self.block(output, block, form.clone()).map(|block| block.component(1)).map_err(
                |error| {
                    failure.borrow_mut().get_or_insert(error);
                    mxx_exponent_lut::rns_bgg::RnsBggError::Binding("a CBD block failed")
                },
            )
        });
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }
        Ok(term?)
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
    fn mask(&self) -> Result<(Vec<Mat>, Vec<Mat>, C), Dio2Error> {
        let outputs = &self.program.outputs;
        let missing = || Dio2Error::Parameters("a refreshing step has every refresh output");
        let omega = self.public_component(outputs.public.ok_or_else(missing)?)?;
        let g = self.public_component(outputs.sigma.ok_or_else(missing)?)?;
        let mask = self.to_state(C::from_components(g.clone(), omega.clone()))?;
        Ok((g, omega, mask))
    }

    /// The refresh targets of chunk `chunk` (its label transition `columns`) for the next step
    /// `next`, the label leaves being `leaf` and the one-encoding `one`: label member `r (m_v +
    /// 1) + c` is `N[r, c] = MulPub(Gamma(LBL'[r, c]), E^sigma)` (spec §6.5), the label product
    /// of chunk `chunk` (see [`LabelProducts::product`]) re-targeted to the public form of `L`
    /// and multiplied by the mask, computed by the part that decodes it.
    fn targets<'t>(
        &'t self,
        next: &NextStep,
        (columns, chunk): (&'t LabelColumns, Int),
        leaf: impl Fn(Int, Int) -> Result<C::Public, Dio2Error> + 't,
        one: C::Public,
    ) -> Result<Targets<'t, C>, Dio2Error> {
        let products = self
            .program
            .label_products
            .as_ref()
            .ok_or(Dio2Error::Input("a refreshing Program has label products"))?;
        let (g, omega, mask) = self.mask()?;
        let helper_mask = mask.clone();
        let helper_omega = omega.clone();
        let helpers = Box::new(move |member: Int| -> Result<C, Dio2Error> {
            let Some(output) = self.program.outputs.helper_blocks else {
                return Ok(helper_mask.clone());
            };
            let count = self.plan.automorphisms.len() - 1;
            let block = member.clone().add(count - 1).rem(count).expression()?;
            let g = self.block(output, block, self.form(RnsBggForm::Public)?)?.component(1);
            let other = self.to_state(C::from_components(g, helper_omega.clone()))?;
            Ok(mxx_dsl::select(member.equal(0usize), vec![other, helper_mask.clone()])?)
        });
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
        let level = self.level();
        let public_form = self.form(RnsBggForm::Public)?;
        let label_mask = mask.clone();
        let label = Box::new(move |member: Int| -> Result<C, Dio2Error> {
            let product = products.product::<C>(
                self.chain,
                columns,
                chunk.clone(),
                member,
                &leaf,
                one.clone(),
            )?;
            let label = C::retarget_public(
                product,
                self.chain,
                (0, RnsBggForm::Public),
                level,
                public_form.clone(),
            )?;
            Ok(label_mask.mul_public(self.chain, level, &label)?)
        });
        Ok(Targets {
            g,
            omega,
            scalars: Family::pack(vec![private, mask, state])?,
            helpers,
            label,
            scalar_keys,
            anchor: self.padded_anchor(next)?,
        })
    }

    /// Part `part`: target `t = part / 2` in the order private, public, state, labels,
    /// helpers, anchor blocks ([`StepPlan::full_target_count`]), component `j = part mod 2`
    /// (spec §6.6, §6.7).
    fn part(
        &self,
        part: &Int,
        next: &NextStep,
        targets: &Targets<'_, C>,
    ) -> Result<Part, Dio2Error> {
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
                (targets.helpers)(helper.clone())?,
                (targets.label)(label.clone())?,
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
        let anchor_key = targets
            .anchor
            .clone()
            .slice(None, Some(super::carrier::block_columns(anchor_block, 1, plan.columns)));
        let a_prime = mxx_dsl::select(
            class,
            vec![
                anchor_key,
                key(RootObject::Helper, Some(helper))?,
                label_key,
                block(&targets.scalar_keys.at(scalar))?,
            ],
        )?;
        Ok(Part { index: part.expression()?, target, a_prime: a_prime.reduce_modulus(&ring) })
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
        targets: &Targets<'_, C>,
    ) -> Result<Vec<Mat>, Dio2Error> {
        let smudge = IntExpr::Add(
            Box::new(IntExpr::Mul(
                Box::new(part.index.clone()),
                Box::new(IntExpr::constant(self.plan.primes)),
            )),
            Box::new(prime.expression()?),
        );
        let smudge = self.smudge(Some(smudge))?;
        let actions = self.chain.refresh_actions(
            self.level(),
            part.a_prime.clone(),
            prime.clone(),
            &self.parameters.reference(),
        )?;
        let fresh = self.fresh(&part.index, &actions)?;
        C::DECODED
            .iter()
            .map(|entry| {
                Ok(refresh_combination(
                    (targets.g[*entry].clone(), targets.omega[*entry].clone()),
                    part.target[*entry].clone(),
                    fresh[*entry].clone(),
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
        let smudge = self.smudge(None)?;
        Ok(C::DECODED.iter().map(|entry| state[*entry].clone() + smudge[*entry].clone()).collect())
    }
}

impl Compose<'_, Key> {
    /// `consume` of the decoded key `[K_0; K_1]` (`2 x w_0` at `Q_0`) of every part and prime of
    /// the refresh to the next step `next` on chunk `chunk` (its label transition `columns`),
    /// member `p |P| + k` for part `p` and the prime of index `k` in the basis of `Q_0`: the
    /// anchor's preimages of the decoded keys. The label leaves are the step root's keys. One
    /// loop over the members computes each key and consumes it, so no family of keys is held.
    pub fn decoded_keys<T: mxx_dsl::GraphValue>(
        &self,
        next: &NextStep,
        label_chunk: (&LabelColumns, Int),
        consume: impl Fn(Mat) -> T,
    ) -> Result<Family<T>, Dio2Error> {
        let plan = self.plan;
        let root = self.chain.root().ok_or(Dio2Error::Input("the step chain has a root"))?;
        let hash_key = root.hash_key_input();
        let leaf_key = hash_key.clone();
        let leaf = move |input: Int, _| {
            Ok(plan.root_key(root, leaf_key.clone(), RootObject::Label, Some(input))?)
        };
        let one = plan.root_key(
            root,
            hash_key.clone(),
            RootObject::Public,
            Some(Int::constant(plan.label_rows)),
        )?;
        let failure = Failure::default();
        let primes = plan.primes;
        let keys = mxx_dsl::parallel(plan.blocks() * primes, |member| {
            // The scalar target families and padded anchor belong to this lane:
            // their last readers precede its preimage sampler. Keeping them outside
            // the loop would retain their storage throughout every invocation.
            // Lane/column concurrency remains the generic backend's W/C choice.
            let targets = failure.keep(self.targets(
                next,
                (label_chunk.0, label_chunk.1.clone()),
                leaf.clone(),
                one.clone(),
            ))?;
            let part = failure.keep(self.part(&member.clone().div(primes), next, &targets))?;
            let decoded = failure.keep(self.decoded(&part, &member.rem(primes), &targets))?;
            Ok(consume(key_rows(decoded)))
        });
        failure.finish(keys)
    }

    /// The decoded key `[K_0; K_1]` (`2 x 1`) of the output decoder.
    pub fn output_key(&self) -> Result<Mat, Dio2Error> {
        Ok(key_rows(self.output_decoded()?))
    }
}

impl Compose<'_, Row> {
    /// The next step input `S_(i+1)` (spec §6.8) of the step input `rows` with the clear labels
    /// `labels`, refreshed to the next step `next` on chunk `chunk` (its label transition
    /// `columns`): the row of every part, decoded for every prime with the hint `rows.hint` and
    /// its preimage `preimages.at(first + p |P| + k)`, recombined from the residues modulo the
    /// primes. Each root object's family of `S_(i+1)` is one loop over its members, each of
    /// which decodes the two parts of its target (spec §6.6), so no family of refreshed parts
    /// is held. Temporary target families are constructed inside each component lane, so
    /// unused alternatives need not remain resident across every refreshed member. The
    /// label leaves are the label rows with their public payloads `mono(stride a)`.
    pub fn next_rows(
        &self,
        next: &NextStep,
        (columns, chunk, labels): (&LabelColumns, Int, &Family<Int>),
        rows: &StepRows,
        (preimages, first): (&Family<Preimage>, Int),
    ) -> Result<StepRows, Dio2Error> {
        let plan = self.plan;
        let ring = self.ring()?;
        let stride = self.parameters.profile()?.layout.stride();
        let root = self.chain.root().ok_or(Dio2Error::Input("the step chain has a root"))?;
        let hash_key = root.hash_key_input();
        let leaf_ring = ring.clone();
        let leaf_key = hash_key.clone();
        let leaf = move |input: Int, value: Int| {
            let key =
                plan.root_key(root, leaf_key.clone(), RootObject::Label, Some(input.clone()))?;
            let payload = leaf_ring.identity(1).multiply_monomial(labels.at(value).mul(stride));
            Ok((rows.labels.at(input), key, payload))
        };
        let one_member = Int::constant(plan.label_rows);
        let one = (
            rows.public.at(one_member.clone()),
            plan.root_key(root, hash_key.clone(), RootObject::Public, Some(one_member))?,
            ring.identity(1),
        );
        let basis = self.chain.level(self.level())?.to_crt().0;
        let half = ring
            .all_coefficients((1, plan.columns), IntExpr::constant((&self.modulus()? + 1u8) / 2u8));
        // The refreshed row of part `index`: one loop over the primes.
        let refreshed = |index: Int| -> Result<Mat, Dio2Error> {
            // Like decoded_keys, form the temporary target families inside the
            // component lane. Their unused scalar/key alternatives then die
            // before preimage import instead of spanning every refreshed member.
            // The backend still chooses lane and column concurrency.
            let targets =
                self.targets(next, (columns, chunk.clone()), leaf.clone(), one.clone())?;
            let part = self.part(&index, next, &targets)?;
            let mut rows = parallel_sum(basis.len(), |prime| {
                let row = self.decoded(&part, &prime, &targets)?.swap_remove(0);
                let member = first.clone().add(index.clone().mul(basis.len())).add(prime.clone());
                let unmasked = unmask(&ring, &rows.hint, preimages.at(member), row) - half.clone();
                Ok(vec![decoded_residue_term(
                    &basis,
                    prime,
                    self.parameters.ring_dimension,
                    unmasked,
                )?])
            })?;
            Ok(rows.remove(0))
        };
        // The two target components are independent. Expose both as lanes so the
        // backend chooses their concurrency; only this constant-size pair is resident.
        // Prime terms are independent parallel lanes with a bounded member sum.
        let target = |target: Int| -> Result<(Mat, Mat), Dio2Error> {
            let failure = Failure::default();
            let rows = mxx_dsl::parallel(2usize, |component| {
                failure.keep(refreshed(target.clone().mul(2usize).add(component)))
            });
            let rows = failure.finish(rows)?;
            Ok((rows.at(0usize), rows.at(1usize)))
        };
        let width = plan.label_width;
        let failure = Failure::default();
        // `ONEPRIV` (target 0) and the state (target 2).
        let scalars = mxx_dsl::parallel(2usize, |member| failure.keep(target(member.mul(2usize))));
        let scalars = failure.finish(scalars)?;
        let public = mxx_dsl::parallel(plan.label_rows + 1, |member| {
            // Member `r < v` is label `(r, 0)`; member `v` is `ONEPUB`, target 1: an index
            // expression, since the target's index names its blocks.
            let member = member.expression()?;
            let label = IntExpr::Add(
                Box::new(IntExpr::Mul(
                    Box::new(member.clone()),
                    Box::new(IntExpr::constant(width + 1)),
                )),
                Box::new(IntExpr::constant(3)),
            );
            let selector =
                IntExpr::FloorDiv(Box::new(member), Box::new(IntExpr::constant(plan.label_rows)));
            failure.keep(target(Int::evaluate(
                IntExpr::Select {
                    selector: Box::new(selector),
                    branches: vec![label, IntExpr::constant(1)],
                }
                .canonicalize(),
            )))
        });
        let public = failure.finish(public)?;
        let labels = mxx_dsl::parallel(plan.label_rows * width, |member| {
            failure.keep(target(
                member.clone().div(width).mul(width + 1).add(member.rem(width)).add(4usize),
            ))
        });
        let labels = failure.finish(labels)?;
        let first_helper = 3 + plan.label_rows * (width + 1);
        let helpers = mxx_dsl::parallel(plan.automorphisms.len(), |member| {
            failure.keep(target(member.add(first_helper)))
        });
        let helpers = failure.finish(helpers)?;
        // The hint's independent anchor parts use the same component-lane
        // placement as the other targets. The generic backend can fuse their
        // disjoint-column sum and choose W without retaining target alternatives
        // across the sequential construction of every hint block.
        let first_anchor = first_helper + plan.automorphisms.len();
        let anchor_parts = 2 * self.anchor_blocks();
        let hint_parts = mxx_dsl::parallel(anchor_parts, |part| {
            let row = failure.keep(refreshed(part.clone().add(2 * first_anchor)))?;
            Ok(place_in_column(row, anchor_parts, &part.expression()?))
        });
        let hint_parts = failure.finish(hint_parts)?;
        let hint =
            mxx_dsl::iterate(anchor_parts - 1, hint_parts.at(Int::constant(0)), |part, sum| {
                Ok(sum + hint_parts.at(part.add(1usize)))
            })?;
        let hint = hint.slice(
            None,
            Some(super::carrier::block_columns(
                IntExpr::constant(0),
                super::anchor_columns(plan.columns),
                1,
            )),
        );
        Ok(StepRows {
            private: scalars.at(Int::constant(0)),
            state: scalars.at(Int::constant(1)),
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

/// The lowered outputs of `ProgramBuild::lower_deferred`: the values (`None` at a deferred
/// output) and the deferred rows by output position.
pub(crate) fn split_deferred<W, WF>(
    lowered: Vec<LoweredOutput<W, WF>>,
) -> (Vec<Option<W>>, BTreeMap<usize, DeferredColumns<W, WF>>) {
    let mut deferred = BTreeMap::new();
    let values = lowered
        .into_iter()
        .enumerate()
        .map(|(position, output)| match output {
            LoweredOutput::Value(value) => Some(value),
            LoweredOutput::Columns(columns) => {
                deferred.insert(position, columns);
                None
            }
        })
        .collect();
    (values, deferred)
}

/// `target - offset` wrapped into `[0, count)` for a nonnegative target index: the member of a
/// target class, valid for the targets of that class (a selected-away candidate needs only an
/// in-range index).
fn wrap(target: &Int, offset: usize, count: usize) -> Int {
    let lift = count * (offset / count + 1) - offset;
    target.clone().add(lift).rem(count)
}

/// Independent terms followed by the canonical member sum recognized by the backend.
/// No application concurrency limit is imposed; the planner chooses wave width.
/// A singleton uses its term directly; larger counts retain the same graph structure.
fn parallel_sum(
    count: usize,
    term: impl Fn(Int) -> Result<Vec<Mat>, Dio2Error>,
) -> Result<Vec<Mat>, Dio2Error> {
    match count {
        0 => Err(Dio2Error::Input("parallel sum needs at least one term")),
        1 => term(Int::constant(0)),
        _ => {
            let failure = Failure::default();
            let members =
                failure.finish(mxx_dsl::parallel(count, |index| failure.keep(term(index))))?;
            let initial = add(members.at(0usize), members.at(1usize));
            Ok(mxx_dsl::iterate(count - 2, initial, |index, total| {
                Ok(add(total, members.at(index.add(2usize))))
            })?)
        }
    }
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

/// The scalar polynomial `P^r 2^-1 mod Q` of digit `digit` (an index expression): one indexed
/// constant, so no literal table and no family of weights.
fn smudge_weight(
    parameters: &Dio2Parameters,
    ring: &Ring,
    modulus: &BigUint,
    digit: IntExpr,
) -> Mat {
    let half = ring.polynomial([IntExpr::constant((modulus + 1u8) / 2u8)]);
    ring.constant(
        (1, 1),
        ConstantMatrix::PowerOfBase {
            base: IntExpr::constant(parameters.prf.output_modulus),
            exponent: digit,
        },
    ) * half
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
            let coefficients = mxx_dsl::parallel(digits, |index| {
                Ok(smudge_weight(&parameters, &ring, &modulus, index.expression()?)
                    .coefficients()
                    .at(0))
            })
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
