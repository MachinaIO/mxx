//! Masked trapdoor-decoder noise refresh.
//!
//! The refreshed source is an encoding `c` whose payload is `(q/p) mu`. The
//! refresh adds the mask encoding `m_enc` with a small payload `m`, removes
//! `s A` from the sum with the [`super::decoder`], and rounds:
//!
//! `x = d - c - m_enc = (q/p) mu + m + E`, output `round((p/q) x) mod p = mu`
//! whenever `|m + E| < q/(2p)`.
//!
//! The output is a plaintext modulo `p`; no encoding at the source modulus is
//! recovered. The decoder's preimage `K` is sampled for the summed key of `c`
//! and `m_enc` only, so the evaluator cannot decode `c` or `m_enc` alone.
//!
//! # Mask
//!
//! The mask is evaluated by the [`RefreshMaskProgram`], a sparse-LWR weak PRF
//! whose terminal LUTs are [`LutOutputRepresentation::UnscaledCoefficient`]:
//! their outputs carry the PRF value itself, not a decoding-scaled one. The
//! negacyclic coefficient LUT returns `-T[z - n']` for inputs `z >= n'`, so a
//! table over `f(k) = floor(p' k / n') + 1` alone yields the digit set
//! `+-[1, p']`, which is not an interval. Two tables fix that:
//!
//! - the digit table `2 f - 1` gives a digit `delta = 2u - (P - 1)` uniform over the `P = 2 p'` odd
//!   values of `[-(P-1), P-1]`;
//! - the parity table `(f - 1) mod 2` gives `beta in {-1, 0, 0, 1}` for even `p'`.
//!
//! Per coefficient, `m = beta + sum_(i<d) P^i delta_i = beta + 2U - (P^d - 1)`
//! with `U` uniform on `[0, P^d)`: centered, with mass `1/(2 P^d)` on every
//! integer of `[-P^d, P^d]` except the two endpoints, which have half that
//! mass. Both tables read only the rounded LWR value, never the low bits of
//! the inner product.
//!
//! The digit family has one member per coefficient and digit, member
//! `o = coefficient d + digit`, weighted by `P^digit`; the parity family has
//! one member per coefficient, member `o = coefficient`, weighted by 1. The
//! scheme adapter places member `o` at `X^coefficient` with its weight; see
//! [`RefreshDeclaration::member_placement`] and
//! [`RefreshDeclaration::member_weights`].
//!
//! # Hiding criterion
//!
//! As in `references/diamond_io.pdf` (§5, Key-Homomorphic PRF via
//! Specialized Noise Refreshing, and §5.4), the mask is sized by the concrete
//! criterion only: its range makes brute-force recovery of the masked value
//! cost more than `2^lambda`, and the accumulated noise fits the decoding
//! margin left after reserving that range ([`fit_mask_digits`]). The
//! proof-level noise-flooding requirement, statistical closeness to uniform
//! of the mask plus its error under their joint distribution, is a documented
//! limitation: the mask carriers' errors are not independent of the mask,
//! since LUT switching material is shared across members and unselected
//! selector branches still contribute their errors.

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId,
        FamilyRange, InputBinding, LutId, LutOutputRepresentation, LutTable, Operand,
        ProgramValidationError, RhsId, WireId,
    },
    program_gadgets::{
        pbc::{PbcCell, PbcGeneratedKeyLayout, PbcPublicLayout},
        prf::SparseLwrPrfProfile,
    },
};
use mxx_backends::poly::{Poly, PolyParams};
use mxx_ir_core::IntExpr;
use num_bigint::BigUint;
use num_traits::Zero;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// Finish one concrete refresh stage from the aggregated mask.
///
/// The common `api::apply_refresh` orchestration validates the declaration,
/// evaluates the mask Program exactly once, and passes its aggregated,
/// member-weighted carrier to this stage-specific operation.
pub trait NoiseRefresh {
    type Mask;
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        mask: Self::Mask,
    ) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Validation and graph-linkage failures in the refresh protocol.
pub enum RefreshError {
    #[error("refresh source output is not a private, scalar, non-coefficient Program output")]
    InvalidSourceOutput,
    #[error("refresh mask Program or its layouts are inconsistent")]
    InvalidMaskProgram,
    #[error("refresh source and mask carriers do not share the decoding ring")]
    InvalidModulusRelation,
    #[error("the mask range and the accumulated noise do not fit the decoding margin")]
    MaskRange,
    #[error("refresh target does not match its declaration")]
    TargetMismatch,
    #[error("refresh metadata commitment failed")]
    Commitment,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// The two member families of the mask Program.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum MaskFamily {
    /// `d` digits per coefficient, read through the digit table `2 f - 1`.
    Digit,
    /// One parity digit per coefficient, read through `(f - 1) mod 2`.
    Parity,
}

impl MaskFamily {
    pub const ALL: [Self; 2] = [Self::Digit, Self::Parity];

    fn index(self) -> usize {
        match self {
            Self::Digit => 0,
            Self::Parity => 1,
        }
    }
}

/// The unscaled readout table of `family` over the canonical sparse-LWR
/// value `f(k) = floor(p' k / n') + 1`.
pub fn mask_table(
    profile: &SparseLwrPrfProfile,
    family: MaskFamily,
) -> Result<LutTable, RefreshError> {
    let readout = profile.canonical_readout().map_err(|_| RefreshError::InvalidMaskProgram)?;
    let (output_width, values) = match family {
        MaskFamily::Digit => (
            profile.p().map_err(|_| RefreshError::InvalidMaskProgram)?,
            readout.iter().map(|f| 2 * f - 1).collect(),
        ),
        MaskFamily::Parity => (2, readout.iter().map(|f| (f - 1) % 2).collect()),
    };
    Ok(LutTable::unary(
        profile.native_width(),
        profile.layout,
        output_width,
        values,
        LutOutputRepresentation::UnscaledCoefficient,
    )?)
}

/// The per-family part of a [`RefreshMaskProgram`].
#[derive(Clone, Debug)]
struct MaskFamilyProgram {
    /// The public key layout of each member, in member order.
    layouts: Vec<PbcPublicLayout>,
    /// One selector RHS family per bucket and slot, of one member each.
    selectors: Vec<Vec<RhsId>>,
    lut: LutId,
    output: WireId,
}

/// The sparse-LWR weak PRF evaluating every mask member, built so that each
/// member is read by the root loop index alone.
///
/// Inputs: the accumulator start `mono(0)` (private, shared by every member)
/// and the public vector `a` of `nu + 1` members, member `nu` being the
/// encoding of zero read by dummy and padding cells. Each family is one root
/// `parallel` over its members. Its body runs the member's buckets in order;
/// for bucket `b` it unrolls the common width `W` of the layouts, multiplying
/// the accumulator by the cell's `a` coordinate (a `Select` over the member
/// index) and weighting it by member `o` of the cell's selector family, then
/// sums the cells. A slot beyond a layout's own width is padding. The body
/// ends with the family's unscaled LUT, and the Program outputs are the two
/// family `Sum`s. No body is nested, so selector members are read at the bare
/// root binder.
#[derive(Clone, Debug)]
pub struct RefreshMaskProgram {
    program: ExponentLutProgram,
    initial_input: WireId,
    input_vector: WireId,
    families: Vec<MaskFamilyProgram>,
}

impl RefreshMaskProgram {
    /// Build the mask Program for `layouts` (`[digit, parity]`, one layout per
    /// member). Every layout must cover the profile's LWR dimension and all
    /// must share one bucket count.
    pub fn new(
        profile: &SparseLwrPrfProfile,
        layouts: [Vec<PbcPublicLayout>; 2],
    ) -> Result<Self, RefreshError> {
        profile.validate().map_err(|_| RefreshError::InvalidMaskProgram)?;
        let bucket_count =
            layouts.iter().flatten().next().ok_or(RefreshError::InvalidMaskProgram)?.cells.len();
        for layout in layouts.iter().flatten() {
            layout.validate().map_err(|_| RefreshError::InvalidMaskProgram)?;
            if layout.parameters.universe_size != profile.lwr_dimension ||
                layout.cells.len() != bucket_count
            {
                return Err(RefreshError::InvalidMaskProgram);
            }
        }
        let zero_member = profile.lwr_dimension;
        let mut builder = ExponentLutProgramBuilder::new();
        let initial_input = builder.input(AttributeKnowledge::PrivatePayload)?;
        let input_vector = builder.input_family(
            IntExpr::constant(profile.lwr_dimension + 1),
            AttributeKnowledge::PublicPayload,
        )?;
        let mut families = Vec::with_capacity(2);
        for (family, layouts) in MaskFamily::ALL.into_iter().zip(layouts) {
            let members = layouts.len();
            if members == 0 {
                return Err(RefreshError::InvalidMaskProgram);
            }
            let width = layouts
                .iter()
                .flat_map(|layout| layout.cells.iter().map(Vec::len))
                .max()
                .ok_or(RefreshError::InvalidMaskProgram)?;
            let selectors = (0..bucket_count)
                .map(|_| {
                    (0..width)
                        .map(|_| builder.rhs_family(IntExpr::constant(members)))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let lut = builder.register_lut(mask_table(profile, family)?)?;
            let range = FamilyRange::full(IntExpr::constant(members))?;
            let family_wire = builder.parallel(range, |builder, member| {
                let mut state = initial_input;
                for (bucket, selectors) in selectors.iter().enumerate() {
                    let mut cells = Vec::with_capacity(width);
                    for (slot, selector) in selectors.iter().enumerate() {
                        // A real cell reads its `a` coordinate; dummy,
                        // padding and missing slots read the zero member.
                        let coordinate = IntExpr::Select {
                            selector: Box::new(member.clone()),
                            branches: layouts
                                .iter()
                                .map(|layout| {
                                    IntExpr::constant(match layout.cells[bucket].get(slot) {
                                        Some(PbcCell::Real { coordinate, .. }) => *coordinate,
                                        _ => zero_member,
                                    })
                                })
                                .collect(),
                        };
                        let shifted = builder.mul_public(
                            Operand::Member { family: input_vector, index: coordinate },
                            Operand::Scalar(state),
                        )?;
                        cells.push(builder.mul_private(
                            Operand::Scalar(shifted),
                            Operand::Member { family: *selector, index: member.clone() },
                        )?);
                    }
                    let mut cells = cells.into_iter();
                    let first = cells.next().ok_or(ProgramValidationError::InvalidLut)?;
                    state = cells.try_fold(first, |sum, cell| {
                        builder.add(Operand::Scalar(sum), Operand::Scalar(cell))
                    })?;
                }
                builder.lut(Operand::Scalar(state), lut).map(Operand::Scalar)
            })?;
            let output = builder.sum(family_wire)?;
            families.push(MaskFamilyProgram { layouts, selectors, lut, output });
        }
        let program = builder
            .finish(families.iter().map(|family| Operand::Scalar(family.output)).collect())?;
        Ok(Self { program, initial_input, input_vector, families })
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.program.id()
    }

    /// The private accumulator start, bound to `mono(0)`.
    pub const fn initial_input(&self) -> WireId {
        self.initial_input
    }

    /// The public vector `a` followed by the encoding of zero.
    pub const fn input_vector(&self) -> WireId {
        self.input_vector
    }

    /// The number of members of `family`.
    pub fn members(&self, family: MaskFamily) -> usize {
        self.families[family.index()].layouts.len()
    }

    /// The terminal LUT of `family`.
    pub fn lut(&self, family: MaskFamily) -> LutId {
        self.families[family.index()].lut
    }

    /// The family whose terminal LUT is `lut`, if any.
    pub fn family_of_lut(&self, lut: LutId) -> Option<MaskFamily> {
        MaskFamily::ALL.into_iter().find(|family| self.families[family.index()].lut == lut)
    }

    /// The Program output of `family`: the `Sum` of its members.
    pub fn output(&self, family: MaskFamily) -> WireId {
        self.families[family.index()].output
    }

    /// The LHS plaintexts: `mono(0)` for the accumulator start, then
    /// `mono(a_i)` for every coordinate of the public vector `a` in `Z_Q` and
    /// `mono(0)` for the zero member, in the profile's exponent layout.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        profile: &SparseLwrPrfProfile,
        parameters: &P::Params,
        a: &[usize],
    ) -> Result<PlaintextInputs<WireId, P>, RefreshError> {
        let dimension = parameters.ring_dimension() as usize;
        if a.len() != profile.lwr_dimension ||
            profile.ring_dimension().ok() != Some(dimension) ||
            a.iter().any(|value| Some(*value) >= profile.q().ok())
        {
            return Err(RefreshError::InvalidMaskProgram);
        }
        let monomial = |exponent: usize| {
            let rotation = (exponent * profile.layout.stride()) % (2 * dimension);
            let value = P::const_rotate_poly(parameters, rotation % dimension);
            if rotation >= dimension { -value } else { value }
        };
        let members = a.iter().copied().chain([0]).map(monomial).collect();
        Ok(BTreeMap::from([
            (self.initial_input, InputBinding::Scalar(monomial(0))),
            (self.input_vector, InputBinding::Family(members)),
        ]))
    }

    /// The private selector plaintexts of the generated `keys`
    /// (`[digit, parity]`, one key per member, each on its member's layout):
    /// member `o` of the selector of bucket `b` and slot `j` is 1 exactly
    /// when key `o` selects slot `j` of bucket `b`.
    pub fn selector_plaintexts<P: Poly>(
        &self,
        parameters: &P::Params,
        keys: [&[PbcGeneratedKeyLayout]; 2],
    ) -> Result<PlaintextInputs<RhsId, P>, RefreshError> {
        let mut plaintexts = BTreeMap::new();
        for (family, keys) in self.families.iter().zip(keys) {
            if keys.len() != family.layouts.len() {
                return Err(RefreshError::InvalidMaskProgram);
            }
            for (key, layout) in keys.iter().zip(&family.layouts) {
                if key.public_layout.layout_id != layout.layout_id ||
                    key.private_schedule().validate(layout).is_err()
                {
                    return Err(RefreshError::InvalidMaskProgram);
                }
            }
            for (bucket, selectors) in family.selectors.iter().enumerate() {
                for (slot, selector) in selectors.iter().enumerate() {
                    let bits = keys
                        .iter()
                        .map(|key| {
                            if key.private_schedule().selected_slot(bucket) == slot {
                                P::const_one(parameters)
                            } else {
                                P::const_zero(parameters)
                            }
                        })
                        .collect();
                    plaintexts.insert(*selector, InputBinding::Family(bits));
                }
            }
        }
        Ok(plaintexts)
    }
}

/// Immutable, checked declaration shared by preprocessing, encryption,
/// online evaluation and noise simulation for one refresh.
///
/// It owns protocol metadata only: the source output, the mask Program and
/// its digit plan. Scheme carriers, plaintexts, the decoder trapdoor and
/// sampled matrices are bound by the concrete stages.
#[derive(Clone, Debug)]
pub struct RefreshDeclaration {
    source_program: ExponentLutProgramId,
    source_output: WireId,
    profile: SparseLwrPrfProfile,
    coefficients: usize,
    digits: usize,
    mask: RefreshMaskProgram,
    commitment: [u8; 32],
}

impl RefreshDeclaration {
    /// Check and freeze one refresh of `source_output`, masked over
    /// `coefficients` coefficients with `digits` mask digits each. `layouts`
    /// are `[digit, parity]`: `coefficients * digits` digit layouts, member
    /// `coefficient * digits + digit`, then `coefficients` parity layouts.
    pub fn new(
        source_program: &ExponentLutProgram,
        source_output: WireId,
        profile: SparseLwrPrfProfile,
        coefficients: usize,
        digits: usize,
        layouts: [Vec<PbcPublicLayout>; 2],
    ) -> Result<Self, RefreshError> {
        validate_refresh_source(source_program, source_output)?;
        if coefficients == 0 ||
            digits == 0 ||
            profile.p_prime % 2 != 0 ||
            layouts[0].len() !=
                coefficients.checked_mul(digits).ok_or(RefreshError::InvalidMaskProgram)? ||
            layouts[1].len() != coefficients
        {
            return Err(RefreshError::InvalidMaskProgram);
        }
        let mask = RefreshMaskProgram::new(&profile, layouts)?;
        let source_program = source_program.id();
        let commitment = commitment(&(
            b"mxx-exponent-lut/refresh/v2".as_slice(),
            source_program,
            source_output,
            &profile,
            coefficients,
            digits,
            mask.id(),
        ))?;
        Ok(Self { source_program, source_output, profile, coefficients, digits, mask, commitment })
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

    /// The number `d` of mask digits per coefficient.
    pub const fn digits(&self) -> usize {
        self.digits
    }

    pub const fn mask(&self) -> &RefreshMaskProgram {
        &self.mask
    }

    pub const fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// The digit base `P = 2 p'`.
    pub fn digit_base(&self) -> BigUint {
        BigUint::from(self.profile.p_prime) * 2u8
    }

    /// The bound `P^d` on the mask magnitude.
    pub fn mask_bound(&self) -> BigUint {
        self.digit_base().pow(self.digits as u32)
    }

    /// The member weights of `family`: `[P^0, ..., P^(d-1)]` for digits and
    /// `[1]` for parity.
    pub fn member_weights(&self, family: MaskFamily) -> Vec<BigUint> {
        match family {
            MaskFamily::Digit => {
                let base = self.digit_base();
                (0..self.digits as u32).map(|power| base.pow(power)).collect()
            }
            MaskFamily::Parity => vec![BigUint::from(1u8)],
        }
    }

    /// Where member `member` of `family` lands, as expressions of the member
    /// index: `(coefficient, weight index)` into [`Self::member_weights`]. A
    /// digit member `coefficient d + digit` has weight index `digit`; a
    /// parity member `coefficient` has weight index 0.
    pub fn member_placement(&self, family: MaskFamily, member: IntExpr) -> (IntExpr, IntExpr) {
        match family {
            MaskFamily::Digit => {
                let digits = Box::new(IntExpr::constant(self.digits));
                let coefficient =
                    IntExpr::FloorDiv(Box::new(member.clone()), digits.clone()).canonicalize();
                (coefficient, IntExpr::Rem(Box::new(member), digits).canonicalize())
            }
            MaskFamily::Parity => (member, IntExpr::constant(0)),
        }
    }

    /// Check that `program` is this declaration's mask Program.
    pub fn validate_mask_program(&self, program: ExponentLutProgramId) -> Result<(), RefreshError> {
        if program == self.mask.id() { Ok(()) } else { Err(RefreshError::InvalidMaskProgram) }
    }
}

/// The source must be a private, scalar, non-coefficient output of
/// `program`: its raw encoding is already final and carries `(q/p) mu`.
fn validate_refresh_source(
    program: &ExponentLutProgram,
    source_output: WireId,
) -> Result<(), RefreshError> {
    let is_output = program
        .outputs()
        .iter()
        .any(|output| matches!(output, Operand::Scalar(wire) if *wire == source_output));
    if !is_output ||
        program.wire_knowledge(source_output)? != AttributeKnowledge::PrivatePayload ||
        program.wire_is_coefficient_output(source_output)?
    {
        return Err(RefreshError::InvalidSourceOutput);
    }
    Ok(())
}

fn commitment(value: &impl Serialize) -> Result<[u8; 32], RefreshError> {
    let bytes = serde_json::to_vec(value).map_err(|_| RefreshError::Commitment)?;
    Ok(Sha256::digest(bytes).into())
}

/// Choose the smallest digit count `d` for digit base `base` such that the
/// mask range `base^d` is at least `2 observations error(d) 2^statistical_bits`,
/// and check that `base^d + error(d) < spacing / 2`, i.e. that the masked
/// value stays inside the decoding margin of `spacing = q / p`. `error(d)` is
/// the accumulated noise bound with `d` digits; it may grow with `d`. Returns
/// `d` and its error bound.
pub fn fit_mask_digits(
    base: &BigUint,
    observations: &BigUint,
    statistical_bits: usize,
    spacing: &BigUint,
    error: impl Fn(usize) -> Result<BigUint, RefreshError>,
) -> Result<(usize, BigUint), RefreshError> {
    if base < &BigUint::from(2u8) || observations.is_zero() {
        return Err(RefreshError::MaskRange);
    }
    let mut digits = 1;
    loop {
        let bound = error(digits)?;
        let required = (observations * &bound * 2u8) << statistical_bits;
        let range = base.pow(digits as u32);
        if range < required {
            digits += 1;
            continue;
        }
        if (range + &bound) * 2u8 >= *spacing {
            return Err(RefreshError::MaskRange);
        }
        return Ok((digits, bound));
    }
}

/// Small refresh declarations for scheme tests.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;
    use crate::{
        ExpLayout,
        program_gadgets::pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
    };

    /// `n' = 4`, `p' = 2` (`P = 4`, `Q = 8`) in the ring of dimension 8, over
    /// an LWR dimension of four.
    pub(crate) fn profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 2 }, 2, 4).unwrap()
    }

    /// `count` random weight-one keys over four coordinates in two buckets,
    /// so each mask member is a two-level chain.
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

    /// A refresh of `source_output` over `coefficients` coefficients with
    /// `digits` digits each, and its keys.
    pub(crate) fn declaration(
        source: &ExponentLutProgram,
        source_output: WireId,
        coefficients: usize,
        digits: usize,
    ) -> (RefreshDeclaration, [Vec<PbcGeneratedKeyLayout>; 2]) {
        let keys = [keys(coefficients * digits), keys(coefficients)];
        let layouts =
            keys.clone().map(|keys| keys.into_iter().map(|key| key.public_layout).collect());
        let declaration = RefreshDeclaration::new(
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
        program_gadgets::pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
    };
    use num_bigint::BigInt;
    use num_traits::One;
    use std::collections::BTreeMap as Map;

    /// n' = 8, p' = 2 (P = 4), over an LWR dimension of eight.
    fn profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 2, 8).unwrap()
    }

    /// The signed coefficient a negacyclic LUT returns for input `z in Z_Q`.
    fn negacyclic(table: &LutTable, z: usize) -> i64 {
        let width = table.values().len();
        let value = table.values()[z % width] as i64;
        if z < width { value } else { -value }
    }

    #[test]
    fn mask_tables_give_uniform_odd_digits_and_balanced_parity() {
        let profile = profile();
        let q = profile.q().unwrap();
        let p = profile.p().unwrap() as i64;
        let digit = mask_table(&profile, MaskFamily::Digit).unwrap();
        let parity = mask_table(&profile, MaskFamily::Parity).unwrap();
        let mut digits = Map::new();
        let mut parities = Map::new();
        for z in 0..q {
            *digits.entry(negacyclic(&digit, z)).or_insert(0usize) += 1;
            *parities.entry(negacyclic(&parity, z)).or_insert(0usize) += 1;
        }
        let expected = (0..p).map(|u| (2 * u - (p - 1), q / p as usize)).collect::<Map<_, _>>();
        assert_eq!(digits, expected, "uniform over the P odd values");
        assert_eq!(parities, Map::from([(-1, q / 4), (0, q / 2), (1, q / 4)]));
    }

    #[test]
    fn digit_and_parity_sum_is_centered_and_uniform_on_an_interval() {
        // Weights 1, 2, 1 for beta = -1, 0, 1, so every tuple contributes 4.
        for (base, digits) in [(2i64, 1u32), (4, 1), (4, 2), (2, 3)] {
            let odd = (0..base).map(|u| 2 * u - (base - 1)).collect::<Vec<_>>();
            let support = base.pow(digits);
            let mut counts = Map::<i64, i64>::new();
            for tuple in 0..support {
                let mut value = 0;
                let mut rest = tuple;
                for power in 0..digits {
                    value += base.pow(power) * odd[(rest % base) as usize];
                    rest /= base;
                }
                for (beta, weight) in [(-1, 1), (0, 2), (1, 1)] {
                    *counts.entry(value + beta).or_default() += weight;
                }
            }
            assert_eq!(
                counts.keys().copied().collect::<Vec<_>>(),
                (-support..=support).collect::<Vec<_>>()
            );
            for (value, count) in counts {
                let expected = if value.abs() == support { 1 } else { 2 };
                assert_eq!(count, expected, "base {base}, d {digits}, value {value}");
            }
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
    ) -> (RefreshDeclaration, [Vec<PbcGeneratedKeyLayout>; 2]) {
        let keys = [keys(coefficients * digits, 0), keys(coefficients, 101)];
        let layouts =
            keys.clone().map(|keys| keys.into_iter().map(|key| key.public_layout).collect());
        let (program, output) = source();
        (
            RefreshDeclaration::new(&program, output, profile(), coefficients, digits, layouts)
                .unwrap(),
            keys,
        )
    }

    #[test]
    fn mask_program_has_two_unscaled_family_sums() {
        let (declaration, keys) = declaration(3, 2);
        let mask = declaration.mask();
        assert_eq!(mask.members(MaskFamily::Digit), 6);
        assert_eq!(mask.members(MaskFamily::Parity), 3);
        let program = mask.program();
        assert_eq!(
            program.outputs(),
            &[
                Operand::Scalar(mask.output(MaskFamily::Digit)),
                Operand::Scalar(mask.output(MaskFamily::Parity))
            ]
        );
        for family in MaskFamily::ALL {
            assert!(program.wire_is_unscaled_coefficient_output(mask.output(family)).unwrap());
        }
        // Every selector member is one bit of one key, one-hot per bucket.
        let parameters =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(16, 1, 17, 4, None, None);
        let selectors = mask
            .selector_plaintexts::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &parameters,
                [&keys[0], &keys[1]],
            )
            .unwrap();
        assert_eq!(selectors.len(), program.rhs_inputs().len());
        assert!(
            mask.selector_plaintexts::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &parameters,
                [&keys[1], &keys[0]],
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
    fn mask_program_graph_does_not_grow_with_members() {
        let small = counts(declaration(2, 1).0.mask().program());
        let large = counts(declaration(5, 2).0.mask().program());
        assert_eq!(small.0, 2, "one root body per family");
        assert_eq!(small.2, 2, "one LUT per family");
        assert_eq!(small, large, "only the index tables scale with the members");
    }

    #[test]
    fn member_placement_maps_digit_members_to_weighted_coefficients() {
        let (declaration, _) = declaration(3, 2);
        let env = mxx_ir_core::ParamEnv::default();
        let weights = declaration.member_weights(MaskFamily::Digit);
        assert_eq!(weights, vec![BigUint::from(1u8), BigUint::from(4u8)]);
        for member in 0..6 {
            let (coefficient, index) =
                declaration.member_placement(MaskFamily::Digit, IntExpr::constant(member));
            assert_eq!(coefficient.evaluate(&env).unwrap(), BigInt::from(member / 2));
            assert_eq!(index.evaluate(&env).unwrap(), BigInt::from(member % 2));
        }
        let (coefficient, index) =
            declaration.member_placement(MaskFamily::Parity, IntExpr::constant(2));
        assert_eq!(coefficient.evaluate(&env).unwrap(), BigInt::from(2));
        assert_eq!(index.evaluate(&env).unwrap(), BigInt::from(0));
        assert_eq!(declaration.member_weights(MaskFamily::Parity), vec![BigUint::from(1u8)]);
        assert_eq!(declaration.mask_bound(), BigUint::from(16u8));
    }

    #[test]
    fn declaration_rejects_coefficient_and_misshaped_inputs() {
        let (program, output) = source();
        let keys = [keys(2, 0), keys(2, 101)];
        let layouts =
            keys.map(|keys| keys.into_iter().map(|key| key.public_layout).collect::<Vec<_>>());
        assert!(
            RefreshDeclaration::new(&program, output, profile(), 2, 2, layouts.clone()).is_err(),
            "digit layouts must cover coefficients * digits"
        );
        assert!(
            RefreshDeclaration::new(&program, output, profile(), 2, 1, layouts.clone()).is_ok()
        );
        let odd =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 1, 8).unwrap();
        assert!(RefreshDeclaration::new(&program, output, odd, 2, 1, layouts.clone()).is_err());
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
            RefreshDeclaration::new(&program, coefficient, profile(), 2, 1, layouts).unwrap_err(),
            RefreshError::InvalidSourceOutput
        );
    }

    #[test]
    fn fit_mask_digits_grows_until_the_range_hides_the_noise() {
        let base = BigUint::from(4u8);
        let error = |digits: usize| Ok(BigUint::from(3u8 + digits as u8));
        // 2 * 1 * error * 2^2 = 32..: base^3 = 64 >= 2 * 6 * 4 = 48.
        let (digits, bound) =
            fit_mask_digits(&base, &BigUint::one(), 2, &BigUint::from(1000u32), error).unwrap();
        assert_eq!((digits, bound), (3, BigUint::from(6u8)));
        assert_eq!(
            fit_mask_digits(&base, &BigUint::one(), 2, &BigUint::from(140u8), error).unwrap_err(),
            RefreshError::MaskRange,
            "64 + 6 = 70 does not fit below 140 / 2"
        );
    }
}
