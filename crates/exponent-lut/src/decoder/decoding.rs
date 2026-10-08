//! Smudged trapdoor decoder.
//!
//! The source is an encoding `c` whose payload is `(q/p) mu`. The decoder
//! adds the smudge encoding `m_enc` with a small payload `m`, removes `s A`
//! from the sum with [`super::unmask`], and rounds:
//!
//! `x = d - c - m_enc = (q/p) mu + m + E`, output `round((p/q) x) mod p = mu`
//! whenever `|m + E| < q/(2p)`.
//!
//! The output is a plaintext modulo `p`; no encoding at the source modulus is
//! recovered. The unmask preimage `K` is sampled for the summed key of `c`
//! and `m_enc` only, so the evaluator cannot unmask `c` or `m_enc` alone.
//!
//! # Smudge
//!
//! The smudge is evaluated by the [`DecoderSmudgeProgram`], a sparse-LWR
//! weak PRF whose terminal LUT is [`LutOutputRepresentation::Coefficient`]:
//! its output carries the PRF value itself (scale 1); the smudge Program is
//! never finalized, so it is never scaled for decryption. The
//! negacyclic coefficient LUT returns `-T[z - n']` for inputs `z >= n'`, so the
//! digit table `T = 2 f - 1` over `f(k) = floor(p' k / n') + 1` gives a digit
//! `delta = 2u - (P - 1)` uniform over the `P = 2 p'` odd values of
//! `[-(P-1), P-1]`: twice the half-integer PRF of eprint 2024/665, Section 5.3.
//! A single negacyclic table cannot be uniform on an integer interval, since its
//! values come in sign-symmetric pairs; the half-integers can.
//!
//! The scheme weights digit `i` by `2^-1 P^i` modulo its odd decoding modulus,
//! so the smudge sum is `sum_i 2^-1 P^i delta_i = U - P^d/2 + 2^-1` with `U`
//! uniform on `[0, P^d)`, and subtracts the public constant `2^-1` from every
//! decoded coefficient: the smudge is then exactly `m = U - P^d/2`, uniform on
//! `[-P^d/2, P^d/2)`. Both read only the rounded LWR value, never the low bits
//! of the inner product.
//!
//! The family has one member per coefficient and digit, member
//! `o = coefficient d + digit`, weighted by `2^-1 P^digit`. The scheme adapter
//! places member `o` at `X^coefficient` with its weight; see
//! [`DecoderDeclaration::member_placement`] and
//! [`DecoderDeclaration::member_weights`].
//!
//! # Hiding criterion
//!
//! The smudge range follows the smudging-lemma ratio `P^d > E 2^bits` for the
//! accumulated noise bound `E`: for `|e| <= E`, `m` and `m + e` are within
//! statistical distance `E / P^d < 2^-bits` per smudged value (a caller that
//! counts several observed values adds their `log2` to `bits`). The smudged
//! value fits the decoding margin, `P^d/2 + E < spacing / 2`
//! ([`fit_smudge_digits`]).

use crate::{
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId,
        LutId, LutOutputRepresentation, Operand, ProgramValidationError, WireId,
    },
    program_gadgets::{
        pbc::PbcPublicLayout,
        prf::{PrfOutput, SparseLwrPrf, SparseLwrPrfProfile},
    },
};
use mxx_ir_core::IntExpr;
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Finish one concrete decoder stage from the aggregated smudge.
///
/// The common `api::apply_decoder` orchestration validates the declaration,
/// evaluates the smudge Program exactly once, and passes its aggregated,
/// member-weighted carrier to this stage-specific operation.
pub trait Decoder {
    type Smudge;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    fn finish_decoding(
        &mut self,
        declaration: &DecoderDeclaration,
        smudge: Self::Smudge,
    ) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Validation and graph-linkage failures in the decoder protocol.
pub enum DecoderError {
    #[error("decoder source output is not a private, scalar, non-coefficient Program output")]
    InvalidSourceOutput,
    #[error("decoder smudge Program or its layouts are inconsistent")]
    InvalidSmudgeProgram,
    #[error("decoder source and smudge carriers do not share the decoding ring")]
    InvalidModulusRelation,
    #[error("the smudge range and the accumulated noise do not fit the decoding margin")]
    SmudgeRange,
    #[error("decoder target does not match its declaration")]
    TargetMismatch,
    #[error("decoder metadata commitment failed")]
    Commitment,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// The smudge Program: the sparse-LWR PRF of every smudge member
/// ([`SparseLwrPrf`] with its half-integer table read as a coefficient), in
/// its member-family form, followed by the `Sum` of
/// the members. The PRF's root member loop is what the scheme's member
/// transform keys on.
#[derive(Clone, Debug)]
pub struct DecoderSmudgeProgram {
    program: ExponentLutProgram,
    prf: SparseLwrPrf,
    output: WireId,
}

impl DecoderSmudgeProgram {
    /// Build the smudge Program for `layouts`, one layout per member; at
    /// least two members, so the PRF takes its member-family form.
    pub fn new(
        profile: &SparseLwrPrfProfile,
        layouts: Vec<PbcPublicLayout>,
    ) -> Result<Self, DecoderError> {
        let invalid = |_| DecoderError::InvalidSmudgeProgram;
        let mut builder = ExponentLutProgramBuilder::new();
        let prf = SparseLwrPrf::build(
            &mut builder,
            *profile,
            layouts,
            LutOutputRepresentation::Coefficient,
        )
        .map_err(invalid)?;
        let PrfOutput::Family(family) = prf.output() else {
            return Err(DecoderError::InvalidSmudgeProgram);
        };
        let output = builder.sum(family)?;
        let program = builder.finish(vec![Operand::Scalar(output)])?;
        Ok(Self { program, prf, output })
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.program.id()
    }

    /// The smudge PRF.
    pub const fn prf(&self) -> &SparseLwrPrf {
        &self.prf
    }

    /// The number of members.
    pub fn members(&self) -> usize {
        self.prf.members()
    }

    /// The terminal digit LUT.
    pub const fn lut(&self) -> LutId {
        self.prf.lut()
    }

    /// The Program output: the `Sum` of the members.
    pub const fn output(&self) -> WireId {
        self.output
    }
}

/// Immutable, checked declaration shared by preprocessing, encryption,
/// online evaluation and noise simulation for one decoder.
///
/// It owns protocol metadata only: the source output, the smudge Program and
/// its digit plan. Scheme carriers, plaintexts, the unmask trapdoor and
/// sampled matrices are bound by the concrete stages.
#[derive(Clone, Debug)]
pub struct DecoderDeclaration {
    source_program: ExponentLutProgramId,
    source_output: WireId,
    profile: SparseLwrPrfProfile,
    coefficients: usize,
    digits: usize,
    smudge: DecoderSmudgeProgram,
    /// The explicit plaintext modulus `p` of the decoded output, if any; see
    /// [`Self::with_plaintext_modulus`].
    plaintext_modulus: Option<BigUint>,
    commitment: [u8; 32],
}

impl DecoderDeclaration {
    /// Check and freeze one decoder of `source_output`, smudged over
    /// `coefficients` coefficients with `digits` smudge digits each, member
    /// `coefficient * digits + digit` on `layouts[member]`. Production code
    /// obtains it from [`crate::utils::select_decoder_at_crt_depth`], which
    /// chooses `digits`.
    pub(crate) fn new(
        source_program: &ExponentLutProgram,
        source_output: WireId,
        profile: SparseLwrPrfProfile,
        coefficients: usize,
        digits: usize,
        layouts: Vec<PbcPublicLayout>,
    ) -> Result<Self, DecoderError> {
        validate_decoder_source(source_program, source_output)?;
        // The member transform keys on the PRF's root member loop, which
        // needs at least two members.
        if coefficients.checked_mul(digits).is_none_or(|members| members < 2) ||
            layouts.len() != coefficients * digits
        {
            return Err(DecoderError::InvalidSmudgeProgram);
        }
        let smudge = DecoderSmudgeProgram::new(&profile, layouts)?;
        let source_program = source_program.id();
        let commitment = commitment(&(
            b"mxx-exponent-lut/decoder/v1".as_slice(),
            source_program,
            source_output,
            &profile,
            coefficients,
            digits,
            smudge.id(),
        ))?;
        Ok(Self {
            source_program,
            source_output,
            profile,
            coefficients,
            digits,
            smudge,
            plaintext_modulus: None,
            commitment,
        })
    }

    /// This declaration with the explicit plaintext modulus `p`, committed
    /// with it. Without one, the plaintext modulus is the modulus of the
    /// level after the source (a Static chain drops to it). A Dynamic chain
    /// keeps one modulus `q`, so its decoder names an odd proper CRT divisor
    /// `p` of `q`; the decoder then reads the source at the spacing `q / p`
    /// and rounds the decoded row to `R_p`.
    pub fn with_plaintext_modulus(
        mut self,
        plaintext_modulus: BigUint,
    ) -> Result<Self, DecoderError> {
        self.commitment = commitment(&(
            b"mxx-exponent-lut/decoder/v1".as_slice(),
            self.source_program,
            self.source_output,
            &self.profile,
            self.coefficients,
            self.digits,
            self.smudge.id(),
            &plaintext_modulus,
        ))?;
        self.plaintext_modulus = Some(plaintext_modulus);
        Ok(self)
    }

    /// The explicit plaintext modulus, if any.
    pub fn plaintext_modulus(&self) -> Option<&BigUint> {
        self.plaintext_modulus.as_ref()
    }

    pub const fn source_program(&self) -> ExponentLutProgramId {
        self.source_program
    }

    pub const fn source_output(&self) -> WireId {
        self.source_output
    }

    pub const fn profile(&self) -> &SparseLwrPrfProfile {
        &self.profile
    }

    /// The number `N` of masked coefficients.
    pub const fn coefficients(&self) -> usize {
        self.coefficients
    }

    /// The number `d` of smudge digits per coefficient.
    pub const fn digits(&self) -> usize {
        self.digits
    }

    pub const fn smudge(&self) -> &DecoderSmudgeProgram {
        &self.smudge
    }

    pub const fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// The digit base `P = 2 p'`.
    pub fn digit_base(&self) -> BigUint {
        BigUint::from(self.profile.p())
    }

    /// The smudge range `P^d`: the smudge is uniform on `[-P^d/2, P^d/2)`.
    pub fn smudge_range(&self) -> BigUint {
        self.digit_base().pow(self.digits as u32)
    }

    /// The bound `P^d / 2` on the smudge magnitude.
    pub fn smudge_bound(&self) -> BigUint {
        self.smudge_range() / 2u8
    }

    /// The member weight numerators `[P^0, ..., P^(d-1)]`, by digit. The
    /// scheme multiplies each by `2^-1` modulo its odd decoding modulus and
    /// subtracts `2^-1` from every decoded coefficient (see the module
    /// documentation).
    pub fn member_weights(&self) -> Vec<BigUint> {
        let base = self.digit_base();
        (0..self.digits as u32).map(|power| base.pow(power)).collect()
    }

    /// Where member `member` lands, as expressions of the member index:
    /// `(coefficient, digit)`, the digit indexing [`Self::member_weights`].
    pub fn member_placement(&self, member: IntExpr) -> (IntExpr, IntExpr) {
        let digits = Box::new(IntExpr::constant(self.digits));
        let coefficient =
            IntExpr::FloorDiv(Box::new(member.clone()), digits.clone()).canonicalize();
        (coefficient, IntExpr::Rem(Box::new(member), digits).canonicalize())
    }

    /// Check that `program` is this declaration's smudge Program.
    pub fn validate_smudge_program(
        &self,
        program: ExponentLutProgramId,
    ) -> Result<(), DecoderError> {
        if program == self.smudge.id() { Ok(()) } else { Err(DecoderError::InvalidSmudgeProgram) }
    }
}

/// The source must be a private, scalar, non-coefficient output of
/// `program`: its raw encoding is already final and carries `(q/p) mu`.
fn validate_decoder_source(
    program: &ExponentLutProgram,
    source_output: WireId,
) -> Result<(), DecoderError> {
    let is_output = program
        .outputs()
        .iter()
        .any(|output| matches!(output, Operand::Scalar(wire) if *wire == source_output));
    if !is_output ||
        program.wire_knowledge(source_output)? != AttributeKnowledge::PrivatePayload ||
        program.wire_is_coefficient_output(source_output)?
    {
        return Err(DecoderError::InvalidSourceOutput);
    }
    Ok(())
}

fn commitment(value: &impl Serialize) -> Result<[u8; 32], DecoderError> {
    let bytes = serde_json::to_vec(value).map_err(|_| DecoderError::Commitment)?;
    Ok(Sha256::digest(bytes).into())
}

/// Choose the smallest digit count `d` for digit base `base` such that the
/// smudge range `base^d` exceeds `error(d) 2^statistical_bits`,
/// and check that `base^d / 2 + error(d) < spacing / 2`, i.e. that the
/// smudged value stays inside the decoding margin of `spacing = q / p`.
/// `error(d)` is the accumulated noise bound with `d` digits; it may grow with
/// `d`, and it is not evaluated once `base^d >= spacing`, where no error fits.
/// Returns `d` and its error bound.
pub fn fit_smudge_digits(
    base: &BigUint,
    statistical_bits: usize,
    spacing: &BigUint,
    mut error: impl FnMut(usize) -> Result<BigUint, DecoderError>,
) -> Result<(usize, BigUint), DecoderError> {
    if base < &BigUint::from(2u8) {
        return Err(DecoderError::SmudgeRange);
    }
    for digits in 1.. {
        let range = base.pow(digits as u32);
        if &range >= spacing {
            break;
        }
        let bound = error(digits)?;
        if range <= &bound << statistical_bits {
            continue;
        }
        if range + &bound * 2u8 >= *spacing {
            break;
        }
        return Ok((digits, bound));
    }
    Err(DecoderError::SmudgeRange)
}

/// Small decoder declarations for scheme tests.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;
    use crate::{
        ExpLayout,
        program_gadgets::pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, generate_key_layout,
        },
    };

    /// `n' = 4`, `p' = 2` (`P = 4`, `Q = 8`) in the ring of dimension 8, over
    /// an LWR dimension of four.
    pub(crate) fn profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 2 }, 4, 4).unwrap()
    }

    /// `count` random weight-one keys over four coordinates in two buckets,
    /// so each smudge member is a two-level chain.
    pub(crate) fn keys(count: usize) -> Vec<PbcGeneratedKeyLayout> {
        let parameters = PbcParameters::custom(4, 1, 2, 2, 64, None);
        (0..count)
            .map(|_| {
                let support = vec![rand::random::<u32>() as usize % 4];
                (0..64)
                    .find_map(|_| {
                        generate_key_layout(&parameters, PbcRootSeed(rand::random()), &support).ok()
                    })
                    .expect("PBC layout")
            })
            .collect()
    }

    /// A decoder of `source_output` over `coefficients` coefficients with
    /// `digits` digits each, and its keys.
    pub(crate) fn declaration(
        source: &ExponentLutProgram,
        source_output: WireId,
        coefficients: usize,
        digits: usize,
    ) -> (DecoderDeclaration, Vec<PbcGeneratedKeyLayout>) {
        let keys = keys(coefficients * digits);
        let layouts = keys.iter().map(|key| key.public_layout.clone()).collect();
        let declaration = DecoderDeclaration::new(
            source,
            source_output,
            profile(),
            coefficients,
            digits,
            layouts,
        )
        .unwrap();
        (declaration, keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExpLayout,
        program::{FamilyRange, InputBinding, LutTable},
        program_gadgets::pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, generate_key_layout,
        },
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap as Map;

    /// n' = 8, p' = 2 (P = 4), over an LWR dimension of eight.
    fn profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 4, 8).unwrap()
    }

    /// The signed coefficient a negacyclic LUT returns for input `z in Z_Q`.
    fn negacyclic(table: &LutTable, z: usize) -> i64 {
        let values = table.values().expect("LUT values");
        let width = values.len();
        let value = values[z % width] as i64;
        if z < width { value } else { -value }
    }

    #[test]
    fn half_integer_readout_gives_uniform_odd_digits() {
        let profile = profile();
        let q = profile.q().unwrap();
        let p = profile.p() as i64;
        let (declaration, _) = declaration(2, 1);
        let program = declaration.smudge().program();
        let inputs = program.encryption_inputs();
        let table = &inputs.luts()[declaration.smudge().lut().0 as usize];
        assert_eq!(table.output_representation(), LutOutputRepresentation::Coefficient);
        let mut digits = Map::new();
        for z in 0..q {
            *digits.entry(negacyclic(table, z)).or_insert(0usize) += 1;
        }
        let expected = (0..p).map(|u| (2 * u - (p - 1), q / p as usize)).collect::<Map<_, _>>();
        assert_eq!(digits, expected, "uniform over the P odd values");
    }

    /// `sum_i 2^-1 P^i delta_i - 2^-1` over all digit tuples is uniform on
    /// `[-P^d/2, P^d/2)`, as an integer and modulo an odd modulus.
    #[test]
    fn halved_digit_sum_minus_half_is_uniform_on_the_centered_interval() {
        let modulus = BigInt::from(97 * 101);
        let half = (&modulus + 1) / 2;
        for (base, digits) in [(2i64, 1u32), (4, 1), (4, 2), (2, 3), (8, 2)] {
            let odd = (0..base).map(|u| 2 * u - (base - 1)).collect::<Vec<_>>();
            let support = base.pow(digits);
            let mut counts = Map::<i64, i64>::new();
            for tuple in 0..support {
                let mut sum = 0;
                let mut halved = BigInt::from(0);
                let mut rest = tuple;
                for power in 0..digits {
                    let delta = odd[(rest % base) as usize];
                    sum += base.pow(power) * delta;
                    halved += &half * base.pow(power) * delta;
                    rest /= base;
                }
                // The digit sum is odd, so subtracting 1 halves exactly.
                let value = (sum - 1) / 2;
                let modular = ((halved - &half) % &modulus + &modulus) % &modulus;
                assert_eq!(modular, (BigInt::from(value) % &modulus + &modulus) % &modulus);
                *counts.entry(value).or_default() += 1;
            }
            assert_eq!(
                counts.keys().copied().collect::<Vec<_>>(),
                (-support / 2..support / 2).collect::<Vec<_>>(),
                "base {base}, d {digits}"
            );
            assert!(counts.values().all(|count| *count == 1), "base {base}, d {digits}");
        }
    }

    fn keys(count: usize, seed_base: u8) -> Vec<PbcGeneratedKeyLayout> {
        let parameters = PbcParameters::paper_evaluation(8, 2);
        (0..count)
            .map(|index| {
                let support = [index % 8, (index + 3) % 8];
                (0..64u8)
                    .find_map(|attempt| {
                        generate_key_layout(
                            &parameters,
                            PbcRootSeed([seed_base.wrapping_add(attempt); 32]),
                            &support,
                        )
                        .ok()
                    })
                    .expect("PBC layout")
            })
            .collect()
    }

    fn source() -> (ExponentLutProgram, WireId) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let output = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        (builder.finish(vec![Operand::Scalar(output)]).unwrap(), output)
    }

    fn declaration(
        coefficients: usize,
        digits: usize,
    ) -> (DecoderDeclaration, Vec<PbcGeneratedKeyLayout>) {
        let keys = keys(coefficients * digits, 0);
        let layouts = keys.iter().map(|key| key.public_layout.clone()).collect();
        let (program, output) = source();
        (
            DecoderDeclaration::new(&program, output, profile(), coefficients, digits, layouts)
                .unwrap(),
            keys,
        )
    }

    #[test]
    fn smudge_program_has_one_coefficient_sum() {
        let (declaration, member_keys) = declaration(3, 2);
        let smudge = declaration.smudge();
        assert_eq!(smudge.members(), 6);
        let program = smudge.program();
        assert_eq!(program.outputs(), &[Operand::Scalar(smudge.output())]);
        assert!(program.wire_is_coefficient_output(smudge.output()).unwrap());
        // Every selector member is one bit of one key, one-hot per bucket.
        let parameters =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(16, 1, 17, 4, None, None);
        let selectors = smudge
            .prf()
            .selector_plaintexts::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &parameters,
                &member_keys,
            )
            .unwrap();
        assert_eq!(selectors.len(), program.rhs_inputs().len());
        let others = keys(6, 101);
        assert!(
            smudge
                .prf()
                .selector_plaintexts::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                    &parameters,
                    &others,
                )
                .is_err(),
            "keys of other layouts are rejected"
        );
    }

    /// Counts the lowered operations of a Program symbolically: each body
    /// is lowered once, so the counts are those of the graph, not of the
    /// members.
    #[derive(Default)]
    struct Counter {
        site: Option<crate::program::LoweringSite>,
        bodies: usize,
        products: usize,
        luts: usize,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("counter")]
    struct CounterError;

    impl crate::program::ExponentLUTProgramLowering for Counter {
        type Wire = ();
        type Rhs = ();
        type Error = CounterError;
        fn add(&mut self, _: &(), _: &()) -> Result<(), CounterError> {
            Ok(())
        }
        fn mul_public(&mut self, _: &(), _: &()) -> Result<(), CounterError> {
            Ok(())
        }
        fn mul_private(&mut self, _: &(), _: &()) -> Result<(), CounterError> {
            self.products += 1;
            Ok(())
        }
        fn lut(&mut self, _: &(), _: &LutTable) -> Result<(), CounterError> {
            self.luts += 1;
            Ok(())
        }
    }

    impl crate::program::ProgramLoweringContext for Counter {
        fn replace_site(
            &mut self,
            site: Option<crate::program::LoweringSite>,
        ) -> Option<crate::program::LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    struct Families;

    impl crate::program::FamilyExecution<Counter> for Families {
        type WireFamily = ();
        type RhsFamily = ();
        fn wire_at(
            &mut self,
            _: &(),
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CounterError>> {
            Ok(())
        }
        fn rhs_at(
            &mut self,
            _: &(),
            _: &IntExpr,
        ) -> Result<(), crate::program::ProgramLowerError<CounterError>> {
            Ok(())
        }
        fn parallel(
            &mut self,
            evaluator: &mut Counter,
            _: &FamilyRange,
            body: impl FnOnce(
                &mut Self,
                &mut Counter,
                IntExpr,
            )
                -> Result<(), crate::program::ProgramLowerError<CounterError>>,
        ) -> Result<(), crate::program::ProgramLowerError<CounterError>> {
            evaluator.bodies += 1;
            body(self, evaluator, IntExpr::LoopIndex(0))
        }
        fn sum(
            &mut self,
            _: &mut Counter,
            _: &(),
        ) -> Result<(), crate::program::ProgramLowerError<CounterError>> {
            Ok(())
        }
    }

    fn counts(program: &ExponentLutProgram) -> (usize, usize, usize) {
        let mut environment = crate::program::Environment::new();
        for (wire, declaration) in program.inputs() {
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(()),
                InputBinding::Family(_) => InputBinding::Family(()),
            };
            environment.bind_wire(*wire, binding).unwrap();
        }
        for (rhs, shape) in program.rhs_inputs() {
            let binding = match shape {
                InputBinding::Scalar(()) => InputBinding::Scalar(()),
                InputBinding::Family(_) => InputBinding::Family(()),
            };
            environment.bind_rhs(*rhs, binding).unwrap();
        }
        let mut counter = Counter::default();
        crate::program::lower_program(program, &mut counter, &mut Families, &mut environment)
            .unwrap();
        (counter.bodies, counter.products, counter.luts)
    }

    #[test]
    fn smudge_program_graph_does_not_grow_with_members() {
        let small = counts(declaration(2, 1).0.smudge().program());
        let large = counts(declaration(5, 2).0.smudge().program());
        assert_eq!(small.0, 1, "one root body");
        assert_eq!(small.2, 1, "one LUT");
        assert_eq!(small, large, "only the index tables scale with the members");
    }

    #[test]
    fn member_placement_maps_digit_members_to_weighted_coefficients() {
        let (declaration, _) = declaration(3, 2);
        let env = mxx_ir_core::ParamEnv::default();
        assert_eq!(declaration.member_weights(), vec![BigUint::from(1u8), BigUint::from(4u8)]);
        for member in 0..6 {
            let (coefficient, index) = declaration.member_placement(IntExpr::constant(member));
            assert_eq!(coefficient.evaluate(&env).unwrap(), BigInt::from(member / 2));
            assert_eq!(index.evaluate(&env).unwrap(), BigInt::from(member % 2));
        }
        assert_eq!(declaration.smudge_range(), BigUint::from(16u8));
        assert_eq!(declaration.smudge_bound(), BigUint::from(8u8));
    }

    #[test]
    fn declaration_rejects_coefficient_and_misshaped_inputs() {
        let (program, output) = source();
        let layouts =
            keys(2, 0).into_iter().map(|key| key.public_layout).collect::<Vec<PbcPublicLayout>>();
        assert!(
            DecoderDeclaration::new(&program, output, profile(), 2, 2, layouts.clone()).is_err(),
            "layouts must cover coefficients * digits"
        );
        assert!(
            DecoderDeclaration::new(&program, output, profile(), 2, 1, layouts.clone()).is_ok()
        );
        assert_eq!(
            DecoderDeclaration::new(&program, output, profile(), 1, 1, layouts[..1].to_vec())
                .unwrap_err(),
            DecoderError::InvalidSmudgeProgram,
            "one member has no root member loop"
        );
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(
                    2,
                    ExpLayout { stride_log2: 0, width_log2: 1 },
                    2,
                    vec![0, 1],
                    LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let coefficient = builder.lut(Operand::Scalar(input), table).unwrap();
        let program = builder.finish(vec![Operand::Scalar(coefficient)]).unwrap();
        assert_eq!(
            DecoderDeclaration::new(&program, coefficient, profile(), 2, 1, layouts).unwrap_err(),
            DecoderError::InvalidSourceOutput
        );
    }

    #[test]
    fn fit_smudge_digits_grows_until_the_range_hides_the_noise() {
        let base = BigUint::from(4u8);
        let error = |digits: usize| Ok(BigUint::from(3u8 + digits as u8));
        // error * 2^2: base^2 = 16 <= 5 * 4, base^3 = 64 > 6 * 4.
        let (digits, bound) = fit_smudge_digits(&base, 2, &BigUint::from(1000u32), error).unwrap();
        assert_eq!((digits, bound), (3, BigUint::from(6u8)));
        assert_eq!(
            fit_smudge_digits(&base, 2, &BigUint::from(70u8), error).unwrap_err(),
            DecoderError::SmudgeRange,
            "64/2 + 6 = 38 does not fit below 70 / 2"
        );
        // An error that never hides stops once the range reaches the spacing,
        // without simulating that trial.
        let mut calls = 0;
        let never = |_: usize| {
            calls += 1;
            Ok(BigUint::from(1u32) << 40)
        };
        assert_eq!(
            fit_smudge_digits(&base, 2, &BigUint::from(1000u32), never).unwrap_err(),
            DecoderError::SmudgeRange
        );
        assert_eq!(calls, 4, "4^5 = 1024 reaches the spacing before its trial");
    }
}
