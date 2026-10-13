//! Scheme-independent sparse-LWR PRF: its profile and the one Program
//! gadget that evaluates it.
use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, DataId, ExpLayout, ExponentLutProgramBuilder, FamilyRange,
        InputBinding, LutId, LutOutputRepresentation, LutTable, Operand, ProgramData,
        ProgramValidationError, RhsId, WireId,
    },
    program_gadgets::pbc::{PbcCell, PbcGeneratedKeyLayout, PbcParameters, PbcPublicLayout},
};
use mxx_backends::poly::{Poly, PolyParams};
use mxx_ir_core::IntExpr;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Scheme-independent errors in native PRF declarations and bindings.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum PrfError {
    #[error("invalid native sparse-LWR profile, block layout or input binding")]
    InvalidSparseLwrBlock,
    #[error("the PRF keys or tables do not match the PRF's keys")]
    KeyShapeMismatch,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// Native sparse-LWR profile.
///
/// The exponent layout fixes the encoding ring dimension the layout lives in,
/// `n = 2^(stride_log2 + width_log2)` and the exponent modulus
/// `Q = 2^(width_log2 + 1) = 2n / 2^stride_log2`. The LWR output modulus `p`
/// (`P`) is even and divides `Q`; the negacyclic readout table covers half of
/// it, `p' = P/2`. The LWR dimension `lwr_dimension` (the length of the
/// public input `a` in `Z_Q^nu`) is independent of the ring dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p: usize,
    pub lwr_dimension: usize,
}

impl SparseLwrPrfProfile {
    /// Validates a deserialized or otherwise externally assembled profile.
    /// Public fields are intentional for declarative manifests, therefore all
    /// consuming boundaries must call this check instead of trusting them.
    pub fn validate(&self) -> Result<(), PrfError> {
        if self.lwr_dimension == 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let ring_log2 = self
            .layout
            .stride_log2
            .checked_add(self.layout.width_log2)
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        let ring_dimension =
            1usize.checked_shl(ring_log2).ok_or(PrfError::InvalidSparseLwrBlock)?;
        self.layout.validate(ring_dimension).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        let exponent_bits =
            self.layout.width_log2.checked_add(1).ok_or(PrfError::InvalidSparseLwrBlock)?;
        let q = 1usize.checked_shl(exponent_bits).ok_or(PrfError::InvalidSparseLwrBlock)?;
        let p = self.p;
        if p < 2 || p % 2 != 0 || p > q || q % p != 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    /// Validates a native profile; its ring is derived from the full layout.
    pub fn new(layout: ExpLayout, p: usize, lwr_dimension: usize) -> Result<Self, PrfError> {
        let profile = Self { layout, p, lwr_dimension };
        profile.validate()?;
        Ok(profile)
    }

    pub const fn native_ring_log2(&self) -> u32 {
        self.layout.stride_log2 + self.layout.width_log2
    }

    pub fn ring_dimension(&self) -> Result<usize, PrfError> {
        1usize.checked_shl(self.native_ring_log2()).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    pub fn q(&self) -> Result<usize, PrfError> {
        let exponent_bits =
            self.layout.width_log2.checked_add(1).ok_or(PrfError::InvalidSparseLwrBlock)?;
        1usize.checked_shl(exponent_bits).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    /// The LWR output modulus `P`.
    pub const fn p(&self) -> usize {
        self.p
    }

    /// Half the output modulus, `p' = P/2`: the values of the readout table.
    pub const fn p_prime(&self) -> usize {
        self.p / 2
    }

    pub const fn native_width(&self) -> usize {
        1usize << self.layout.width_log2
    }

    /// Returns the canonical sparse-LWR value table
    /// `f(k) = floor(p' * k / n') + 1` for `0 <= k < n'`.
    pub fn canonical_readout(&self) -> Result<Vec<usize>, PrfError> {
        let n_prime = self.native_width();
        let values = (0..n_prime)
            .map(|k| self.p_prime().checked_mul(k).map(|value| value / n_prime + 1))
            .collect::<Option<Vec<_>>>()
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        Ok(values)
    }
}

impl<'de> Deserialize<'de> for SparseLwrPrfProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawProfile {
            layout: ExpLayout,
            p: usize,
            lwr_dimension: usize,
        }
        let raw = RawProfile::deserialize(deserializer)?;
        let profile = Self { layout: raw.layout, p: raw.p, lwr_dimension: raw.lwr_dimension };
        profile.validate().map_err(serde::de::Error::custom)?;
        Ok(profile)
    }
}

/// The PRF output: one scalar for one member, else the family of members.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrfOutput {
    Scalar(WireId),
    Family(WireId),
}

/// One selector RHS family: `members` bits, at most `max_nonzero` of them one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrfSelectorFamily {
    pub rhs: RhsId,
    pub members: usize,
    /// The number of one bits in the family, or in each row with
    /// `row_length`.
    pub max_nonzero: usize,
    /// The row length of a row-wise one-hot family.
    pub row_length: Option<usize>,
}

/// The public shape of the keys of a multi-member PRF: `members` keys, each
/// with a PBC layout of `parameters`, whose static bucket width limit `W`
/// is the row width of every bucket (a layout narrower than `W` is padded
/// with the zero member).
///
/// The layouts themselves are runtime inputs of the PRF (one table per
/// bucket, see [`SparseLwrPrf`]), so the shape also commits to them: the
/// key-set commitment is the digest of the members' layout ids, and the
/// table digest is the digest of the tables they induce. A declaration that
/// commits the shape binds its hash namespaces to the key set, and a stage
/// checks the tables it binds against the table digest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrfKeyShape {
    parameters: PbcParameters,
    members: usize,
    key_set: [u8; 32],
    tables: [u8; 32],
}

impl PrfKeyShape {
    /// The shape of the keys with public layouts `layouts`, one per member
    /// (at least two), all on the same PBC parameters with a bucket width
    /// limit.
    pub fn new(layouts: &[PbcPublicLayout]) -> Result<Self, PrfError> {
        let invalid = PrfError::InvalidSparseLwrBlock;
        let [first, _, ..] = layouts else { return Err(invalid) };
        let parameters = first.parameters.clone();
        let width = parameters.bucket_width_limit.ok_or(PrfError::InvalidSparseLwrBlock)?;
        for layout in layouts {
            layout.validate().map_err(|_| PrfError::InvalidSparseLwrBlock)?;
            if layout.parameters != parameters || layout.bucket_width > width {
                return Err(invalid);
            }
        }
        let mut key_set = Sha256::new();
        key_set.update(b"mxx-exponent-lut/prf/key-set/v1");
        key_set.update((layouts.len() as u64).to_le_bytes());
        for layout in layouts {
            key_set.update(layout.layout_id.0);
        }
        let tables = layout_tables(layouts, parameters.bucket_count, width);
        Ok(Self {
            members: layouts.len(),
            key_set: key_set.finalize().into(),
            tables: tables_digest(tables.iter().map(Vec::as_slice)),
            parameters,
        })
    }

    /// The shape of `members` keys (at least two) on `parameters` (with a
    /// bucket width limit) from the digests [`Self::new`] computes of their
    /// layouts: the key-set commitment `key_set` and the table digest
    /// `tables`.
    ///
    /// A declaration needs no keys, so a run that never binds them (a noise
    /// simulation, or the export of a claim, whose graphs bind the tables
    /// and selectors by shape only) declares its shapes with any digests
    /// here, without generating `members` layouts; the tables a stage binds
    /// must still match `tables`.
    pub fn committed(
        parameters: PbcParameters,
        members: usize,
        key_set: [u8; 32],
        tables: [u8; 32],
    ) -> Result<Self, PrfError> {
        parameters.validate().map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        if members < 2 || parameters.bucket_width_limit.is_none() {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(Self { parameters, members, key_set, tables })
    }

    pub const fn parameters(&self) -> &PbcParameters {
        &self.parameters
    }

    pub const fn members(&self) -> usize {
        self.members
    }

    /// The row width `W`, the bucket width limit of the parameters.
    pub fn width(&self) -> usize {
        self.parameters.bucket_width_limit.unwrap_or(0)
    }

    pub const fn bucket_count(&self) -> usize {
        self.parameters.bucket_count
    }

    /// The key-set commitment: the digest of the members' layout ids.
    pub const fn key_set(&self) -> &[u8; 32] {
        &self.key_set
    }
}

/// The table of every bucket of `layouts`: entry `o W + j` of bucket `b`'s
/// table is the coordinate of cell `j` of member `o`'s bucket `b`, or the
/// zero member `nu` for a dummy, padding or missing cell.
fn layout_tables(layouts: &[PbcPublicLayout], buckets: usize, width: usize) -> Vec<Vec<usize>> {
    (0..buckets)
        .map(|bucket| {
            layouts
                .iter()
                .flat_map(|layout| {
                    let zero_member = layout.parameters.universe_size;
                    (0..width).map(move |slot| match layout.cells[bucket].get(slot) {
                        Some(PbcCell::Real { coordinate, .. }) => *coordinate,
                        _ => zero_member,
                    })
                })
                .collect()
        })
        .collect()
}

/// The digest of the tables of every bucket, in bucket order.
fn tables_digest<'a>(tables: impl ExactSizeIterator<Item = &'a [usize]>) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/prf/tables/v1");
    digest.update((tables.len() as u64).to_le_bytes());
    for table in tables {
        digest.update((table.len() as u64).to_le_bytes());
        for entry in table {
            digest.update((*entry as u64).to_le_bytes());
        }
    }
    digest.finalize().into()
}

/// The keys a [`SparseLwrPrf`] is built for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrfKeys {
    /// One member, whose public layout is a constant of the Program.
    Layout(PbcPublicLayout),
    /// Several members of the key shape, whose layouts are runtime integer data.
    Shape(PrfKeyShape),
}

impl PrfKeys {
    /// The keys of the public layouts `layouts`, one per member: one layout
    /// is the one-member form, several make up a key shape.
    pub fn new(layouts: &[PbcPublicLayout]) -> Result<Self, PrfError> {
        match layouts {
            [layout] => Ok(Self::Layout(layout.clone())),
            layouts => PrfKeyShape::new(layouts).map(Self::Shape),
        }
    }
}

/// A sparse-key LWR weak PRF added to a Program.
///
/// Member `o` computes `rho(<a, k_o> mod Q)` on the public vector `a in
/// Z_Q^nu`, one public-payload input family of `nu + 1` members (member `nu`
/// is the encoding of zero, read by dummy and padding cells), under its own
/// sparse key `k_o` placed by its probabilistic-batch-code layout: per
/// bucket, each cell multiplies the accumulator (starting at the private
/// input `mono(0)`) by its `a` coordinate (a public product) and weights it by
/// its selector bit (a private product), the cells are summed, and a terminal
/// LUT reads out `rho`.
///
/// The readout is the half-integer table `2 f(k) - 1` over the canonical
/// `f(k) = floor(p' k / n') + 1` (output width `P`): through the negacyclic
/// wrap the output `delta = (-1)^b (2 f(z mod n') - 1)` is uniform on the `P`
/// odd values of `[-(P-1), P-1]` for a uniform `z = <a, k_o> mod Q`, twice the
/// half-integer PRF of eprint 2024/665, Section 5.3.
///
/// The shape depends on the keys ([`PrfKeys`]):
///
/// - One member (its layout): per bucket, one root `parallel` over the bucket's cells, selector
///   family of one bit per cell (one-hot), and a `Sum`; the cell coordinates are constants of the
///   Program, and the output is a scalar.
/// - Several members (a [`PrfKeyShape`] of row width `W`): per bucket, one root `parallel` over the
///   `M W` cells `f = o W + j` (member `o`, slot `j`) multiplies member `o`'s accumulator by the
///   cell's coordinate and weights it by member `f` of the bucket's selector family of `M W` bits
///   (member `o`'s one-hot row at `o W`), read at the bare root binder; then a root `parallel` over
///   the members sums each member's row (a nested `parallel` over the slots reads the cells through
///   the identity `1 * X^0`). The coordinate of cell `f` is entry `f` of the bucket's coordinate
///   table ([`Operand::Selected`] of the input vector by the bucket's bounded integer data family;
///   `M W` entries in `[0, nu]`, the zero member for a dummy, padding or missing slot), so the
///   Program depends on the key shape only and the stages bind the layouts as Program data
///   ([`Self::key_inputs`]). The graph does not grow with the members, and the noise model lowers
///   each body once: the row contract ([`PrfSelectorFamily::row_length`]) keeps every row sum's
///   payload support at one. A last root `parallel` applies the readout LUT; the output is the
///   member family.
#[derive(Clone, Debug)]
pub struct SparseLwrPrf {
    profile: SparseLwrPrfProfile,
    keys: PrfKeys,
    initial_input: WireId,
    input_vector: WireId,
    lut: LutId,
    /// The family of each bucket (one entry per bucket).
    selectors: Vec<Vec<RhsId>>,
    /// The coordinate table of each bucket of a multi-member PRF: a bounded
    /// integer data family.
    tables: Vec<DataId>,
    output: PrfOutput,
}

impl SparseLwrPrf {
    /// Add the PRF of `keys` (each layout covering the profile's LWR
    /// dimension) to `builder`, its terminal LUT returning the readout in
    /// `representation`.
    pub fn build(
        builder: &mut ExponentLutProgramBuilder,
        profile: SparseLwrPrfProfile,
        keys: PrfKeys,
        representation: LutOutputRepresentation,
    ) -> Result<Self, PrfError> {
        profile.validate()?;
        match &keys {
            PrfKeys::Layout(layout) => {
                layout.validate().map_err(|_| PrfError::InvalidSparseLwrBlock)?;
                if layout.parameters.universe_size != profile.lwr_dimension {
                    return Err(PrfError::InvalidSparseLwrBlock);
                }
            }
            PrfKeys::Shape(shape) => {
                if shape.parameters.universe_size != profile.lwr_dimension ||
                    shape.members < 2 ||
                    shape.width() == 0
                {
                    return Err(PrfError::InvalidSparseLwrBlock);
                }
            }
        }
        let values =
            profile.canonical_readout()?.into_iter().map(|f| 2 * f - 1).collect::<Vec<_>>();
        let initial_input = builder.input(AttributeKnowledge::PrivatePayload)?;
        let input_vector = builder.input_family(
            IntExpr::constant(profile.lwr_dimension + 1),
            AttributeKnowledge::PublicPayload,
        )?;
        let lut = builder.register_lut(LutTable::unary(
            profile.native_width(),
            profile.layout,
            profile.p(),
            values,
            representation,
        )?)?;
        let zero_member = profile.lwr_dimension;
        let coordinate = |cell: Option<&PbcCell>| match cell {
            Some(PbcCell::Real { coordinate, .. }) => *coordinate,
            _ => zero_member,
        };
        let (selectors, tables, output) = match &keys {
            PrfKeys::Layout(layout) => {
                let mut selectors = Vec::with_capacity(layout.cells.len());
                let mut state = initial_input;
                for cells in &layout.cells {
                    // The bucket's selector is consumed by its own private
                    // multiplication level, so it is its own RHS family.
                    let selector = builder.rhs_family(IntExpr::constant(cells.len()))?;
                    let coordinates =
                        cells.iter().map(|cell| coordinate(Some(cell))).collect::<Vec<_>>();
                    let range = FamilyRange::full(IntExpr::constant(cells.len()))?;
                    let family = builder.parallel(range, |builder, slot| {
                        let coordinate = IntExpr::Select {
                            selector: Box::new(slot.clone()),
                            branches: coordinates.iter().copied().map(IntExpr::constant).collect(),
                        };
                        // X^(acc + a_r), then weighted by the cell's selector bit.
                        let shifted = builder.mul_public(
                            Operand::Member { family: input_vector, index: coordinate },
                            Operand::Scalar(state),
                        )?;
                        let weighted = builder.mul_private(
                            Operand::Scalar(shifted),
                            Operand::Member { family: selector, index: slot },
                        )?;
                        Ok(Operand::Scalar(weighted))
                    })?;
                    state = builder.sum(family)?;
                    selectors.push(vec![selector]);
                }
                (
                    selectors,
                    Vec::new(),
                    PrfOutput::Scalar(builder.lut(Operand::Scalar(state), lut)?),
                )
            }
            PrfKeys::Shape(shape) => {
                let (members, width) = (shape.members, shape.width());
                // One selector family per bucket, member `o`'s row of `width` bits
                // at `o width ..`: each row is one-hot (the key's slot in that
                // bucket), the row contract of [`PrfSelectorFamily::row_length`].
                let selectors = (0..shape.bucket_count())
                    .map(|_| {
                        builder.rhs_family(IntExpr::constant(members * width)).map(|rhs| vec![rhs])
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // One table per bucket: the coordinate of every cell, a member
                // of the input vector.
                let tables = (0..shape.bucket_count())
                    .map(|_| builder.data_family(members * width, Some(zero_member + 1)))
                    .collect::<Result<Vec<_>, _>>()?;
                let cells_range = FamilyRange::full(IntExpr::constant(members * width))?;
                let members_range = FamilyRange::full(IntExpr::constant(members))?;
                let row_range = FamilyRange::full(IntExpr::constant(width))?;
                // The state of every member, `None` before the first bucket (the
                // accumulator start `mono(0)` is shared).
                let mut state = None;
                for (selector, table) in selectors.iter().zip(&tables) {
                    // Cell `f = o width + j` of every member `o` and slot `j`, in one
                    // root loop that reads selector member `f` as its own member and
                    // the input vector at the table's entry `f`.
                    let cells = builder.parallel(cells_range.clone(), |builder, cell| {
                        let member = IntExpr::FloorDiv(
                            Box::new(cell.clone()),
                            Box::new(IntExpr::constant(width)),
                        );
                        let state = state.map_or(Operand::Scalar(initial_input), |family| {
                            Operand::Member { family, index: member }
                        });
                        // X^(acc + a_r), then weighted by the cell's selector bit.
                        let shifted = builder.mul_public(
                            Operand::Selected {
                                family: input_vector,
                                data: *table,
                                position: cell.clone(),
                            },
                            state,
                        )?;
                        let weighted = builder.mul_private(
                            Operand::Scalar(shifted),
                            Operand::Member { family: selector[0], index: cell },
                        )?;
                        Ok(Operand::Scalar(weighted))
                    })?;
                    // Each member's state is the sum of its row of cells; a body
                    // returns an operation, so each cell is read by the identity
                    // `1 * X^0`, which keeps its noise.
                    state = Some(builder.parallel(members_range.clone(), |builder, member| {
                        let row = builder.parallel(row_range.clone(), |builder, slot| {
                            let cell = IntExpr::Add(
                                Box::new(IntExpr::Mul(
                                    Box::new(member.clone()),
                                    Box::new(IntExpr::constant(width)),
                                )),
                                Box::new(slot),
                            );
                            builder
                                .mul_small_constant(
                                    Operand::Member { family: cells, index: cell },
                                    vec![1],
                                    IntExpr::constant(0),
                                )
                                .map(Operand::Scalar)
                        })?;
                        builder.sum(row).map(Operand::Scalar)
                    })?);
                }
                let state = state.ok_or(PrfError::InvalidSparseLwrBlock)?;
                let family = builder.parallel(members_range, |builder, member| {
                    builder
                        .lut(Operand::Member { family: state, index: member }, lut)
                        .map(Operand::Scalar)
                })?;
                (selectors, tables, PrfOutput::Family(family))
            }
        };
        Ok(Self { profile, keys, initial_input, input_vector, lut, selectors, tables, output })
    }

    pub const fn profile(&self) -> SparseLwrPrfProfile {
        self.profile
    }

    /// The keys the PRF was built for.
    pub const fn keys(&self) -> &PrfKeys {
        &self.keys
    }

    /// The key shape of a multi-member PRF; `None` for one member.
    pub const fn key_shape(&self) -> Option<&PrfKeyShape> {
        match &self.keys {
            PrfKeys::Layout(_) => None,
            PrfKeys::Shape(shape) => Some(shape),
        }
    }

    /// The number of members.
    pub fn members(&self) -> usize {
        match &self.keys {
            PrfKeys::Layout(_) => 1,
            PrfKeys::Shape(shape) => shape.members,
        }
    }

    /// The number of PBC buckets.
    pub fn bucket_count(&self) -> usize {
        self.selectors.len()
    }

    /// The coordinate table (an integer data family) of each bucket of a
    /// multi-member PRF, in bucket order; empty for one member.
    pub fn tables(&self) -> &[DataId] {
        &self.tables
    }

    /// The private accumulator start, bound to `mono(0)`.
    pub const fn initial_input(&self) -> WireId {
        self.initial_input
    }

    /// The public vector `a` followed by the encoding of zero.
    pub const fn input_vector(&self) -> WireId {
        self.input_vector
    }

    /// The terminal readout LUT.
    pub const fn lut(&self) -> LutId {
        self.lut
    }

    pub const fn output(&self) -> PrfOutput {
        self.output
    }

    /// Every selector RHS family with its member count and how many of its
    /// bits may be one: a one-member bucket family is one-hot; with several
    /// members, bucket `b`'s family holds member `o`'s one-hot row of
    /// `W` bits at `o W`, so each aligned row of length `W` has one nonzero
    /// bit.
    pub fn selector_families(&self) -> Vec<PrfSelectorFamily> {
        self.selectors
            .iter()
            .enumerate()
            .map(|(bucket, families)| match &self.keys {
                PrfKeys::Layout(layout) => PrfSelectorFamily {
                    rhs: families[0],
                    members: layout.cells[bucket].len(),
                    max_nonzero: 1,
                    row_length: None,
                },
                PrfKeys::Shape(shape) => PrfSelectorFamily {
                    rhs: families[0],
                    members: shape.members * shape.width(),
                    max_nonzero: 1,
                    row_length: Some(shape.width()),
                },
            })
            .collect()
    }

    /// The private selector plaintexts and the coordinate tables (Program data) of the
    /// generated `keys`, one key per member: with one member (on the PRF's
    /// layout), bit `j` of bucket `b`'s family is 1 exactly when the key
    /// selects slot `j` of bucket `b`, and there are no tables; with several
    /// (whose layouts must make up the PRF's key shape), bit `o W + j` of
    /// bucket `b`'s family is 1 exactly when member `o`'s key selects slot
    /// `j`, and entry `o W + j` of bucket `b`'s table is the coordinate of that
    /// cell (the zero member `nu` for a dummy, padding or missing cell).
    #[allow(clippy::type_complexity)]
    pub fn key_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        keys: &[PbcGeneratedKeyLayout],
    ) -> Result<(PlaintextInputs<RhsId, P>, ProgramData), PrfError> {
        let layouts = keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>();
        let (width, tables) = match (&self.keys, layouts.as_slice()) {
            (PrfKeys::Layout(layout), [key]) if key.layout_id == layout.layout_id => {
                (layout.bucket_width, ProgramData::new())
            }
            (PrfKeys::Shape(shape), _) if PrfKeyShape::new(&layouts).as_ref() == Ok(shape) => (
                shape.width(),
                self.tables
                    .iter()
                    .copied()
                    .zip(layout_tables(&layouts, shape.bucket_count(), shape.width()))
                    .fold(ProgramData::new(), |data, (id, table)| data.with(id, table)),
            ),
            _ => return Err(PrfError::KeyShapeMismatch),
        };
        for key in keys {
            key.private_schedule()
                .validate(&key.public_layout)
                .map_err(|_| PrfError::KeyShapeMismatch)?;
        }
        // The two constant plaintexts are built once; every key's row is filled in parallel.
        let (one, zero) = (P::const_one(parameters), P::const_zero(parameters));
        let plaintexts = self
            .selectors
            .iter()
            .enumerate()
            .map(|(bucket, families)| {
                let bits = keys
                    .par_iter()
                    .flat_map_iter(|key| {
                        let selected = key.private_schedule().selected_slot(bucket);
                        (0..width).map(move |slot| slot == selected)
                    })
                    .map(|selected| if selected { one.clone() } else { zero.clone() })
                    .collect();
                (families[0], InputBinding::Family(bits))
            })
            .collect();
        Ok((plaintexts, tables))
    }

    /// Check the Program `data` bound to this PRF's Program against its
    /// keys: no coordinate table for one member; for several, one per bucket
    /// whose digest is the key shape's table digest.
    pub fn check_tables(&self, data: &ProgramData) -> Result<(), PrfError> {
        let matches = match &self.keys {
            PrfKeys::Layout(_) => true,
            PrfKeys::Shape(shape) => self
                .tables
                .iter()
                .map(|id| data.values(*id))
                .collect::<Option<Vec<_>>>()
                .is_some_and(|ordered| tables_digest(ordered.into_iter()) == shape.tables),
        };
        if matches { Ok(()) } else { Err(PrfError::KeyShapeMismatch) }
    }

    /// The LHS plaintexts: `mono(0)` for the accumulator start, then
    /// `mono(a_i)` for every coordinate of the public vector `a` in `Z_Q` and
    /// `mono(0)` for the zero member, in the profile's exponent layout.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        a: &[usize],
    ) -> Result<PlaintextInputs<WireId, P>, PrfError> {
        let profile = &self.profile;
        let dimension = parameters.ring_dimension() as usize;
        if a.len() != profile.lwr_dimension ||
            profile.ring_dimension().ok() != Some(dimension) ||
            a.iter().any(|value| Some(*value) >= profile.q().ok())
        {
            return Err(PrfError::InvalidSparseLwrBlock);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn profile_derives_q_and_p_for_nonbinary_digit_bases() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 16, 64).unwrap();
        assert_eq!(profile.native_width(), 32);
        assert_eq!(profile.q().unwrap(), 64);
        assert_eq!((profile.p(), profile.p_prime()), (16, 8));
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 6 }, 32, 128).unwrap();
        assert_eq!(profile.q().unwrap(), 128);
        assert_eq!((profile.p(), profile.p_prime()), (32, 16));
    }

    #[test]
    fn profile_rejects_odd_or_nondividing_output_modulus() {
        let layout = ExpLayout { stride_log2: 1, width_log2: 5 };
        for p in [0, 1, 3, 24, 128] {
            assert!(SparseLwrPrfProfile::new(layout, p, 64).is_err(), "P = {p}");
        }
        assert!(SparseLwrPrfProfile::new(layout, 2, 64).is_ok());
    }

    #[test]
    fn profile_preserves_zero_and_nonzero_stride_input_layouts() {
        let zero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 0, width_log2: 4 }, 4, 16).unwrap();
        let nonzero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 4, 16).unwrap();

        assert_eq!(zero_stride.ring_dimension(), nonzero_stride.ring_dimension());
        assert_ne!(
            zero_stride.q(),
            nonzero_stride.q(),
            "the exponent modulus is selected by input width"
        );
        assert_eq!(zero_stride.canonical_readout().unwrap().len(), 16);
        assert_eq!(nonzero_stride.canonical_readout().unwrap().len(), 8);
    }

    #[test]
    fn canonical_sparse_lwr_table_has_full_q_complement_and_uniform_p_values() {
        for (ring_dimension_log2, stride_log2) in
            [(3_u32, 0_u32), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
        {
            let width_log2 = ring_dimension_log2 - stride_log2;
            let n_prime = 1usize << width_log2;
            let p_prime = n_prime / 2;
            let profile = SparseLwrPrfProfile::new(
                ExpLayout { stride_log2, width_log2 },
                2 * p_prime,
                1 << ring_dimension_log2,
            )
            .unwrap();
            let readout = profile.canonical_readout().unwrap();
            let expected = (0..n_prime).map(|k| p_prime * k / n_prime + 1).collect::<Vec<_>>();
            assert_eq!(readout, expected);
            for k in 0..n_prime {
                let complement = 1_i64 - expected[k] as i64;
                assert_eq!(expected[k] as i64 + complement, 1);
            }
            for value in 1..=p_prime {
                assert_eq!(expected.iter().filter(|entry| **entry == value).count(), 2);
            }
            assert_eq!(profile.q().unwrap(), 2 * n_prime);
        }
    }

    #[test]
    fn profile_deserialization_rechecks_derived_moduli() {
        let encoded = serde_json::json!({
            "layout": { "stride_log2": 1, "width_log2": 5 },
            "p": 24,
        });
        assert!(serde_json::from_value::<SparseLwrPrfProfile>(encoded).is_err());
    }

    /// n=32, Q=32, P=4 over an LWR dimension of ten.
    fn toy_profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4, 10).unwrap()
    }

    /// A key of weight four with its own PBC layout over ten coordinates,
    /// of bucket width at most eight. The layout depends only on the root
    /// seed, so distinct keys search distinct seed ranges starting at
    /// `seed_base`.
    fn toy_key(support: &[usize], seed_base: u8) -> PbcGeneratedKeyLayout {
        let parameters =
            PbcParameters { bucket_width_limit: Some(8), ..PbcParameters::paper_evaluation(10, 4) };
        (seed_base..seed_base.saturating_add(32))
            .find_map(|seed| {
                crate::program_gadgets::pbc::generate_key_layout(
                    &parameters,
                    crate::program_gadgets::pbc::PbcRootSeed([seed; 32]),
                    support,
                )
                .ok()
            })
            .expect("toy key must find a PBC layout")
    }

    fn toy_prf(
        keys: &[PbcGeneratedKeyLayout],
    ) -> (crate::program::ExponentLutProgram, SparseLwrPrf) {
        let mut builder = ExponentLutProgramBuilder::new();
        let layouts = keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>();
        let prf = SparseLwrPrf::build(
            &mut builder,
            toy_profile(),
            PrfKeys::new(&layouts).unwrap(),
            LutOutputRepresentation::Coefficient,
        )
        .unwrap();
        let output = match prf.output() {
            PrfOutput::Scalar(wire) => vec![Operand::Scalar(wire)],
            PrfOutput::Family(family) => (0..keys.len())
                .map(|index| Operand::Member { family, index: IntExpr::constant(index) })
                .collect(),
        };
        (builder.finish(output).unwrap(), prf)
    }

    /// The bits of `family` in `plaintexts`, as integers.
    fn bits(
        parameters: &mxx_backends::poly::dcrt::params::DCRTPolyParams,
        plaintexts: &PlaintextInputs<RhsId, mxx_backends::poly::dcrt::poly::DCRTPoly>,
        family: RhsId,
    ) -> Vec<u64> {
        let InputBinding::Family(members) = &plaintexts[&family] else { panic!("a family") };
        let one = <mxx_backends::poly::dcrt::poly::DCRTPoly as Poly>::const_one(parameters);
        members.iter().map(|bit| u64::from(bit == &one)).collect()
    }

    /// One member: one one-hot selector family per bucket, consumed at the
    /// bucket's level, whose selected coordinates sum to the key inner
    /// product, and no coordinate tables.
    #[test]
    fn one_member_prf_has_one_hot_bucket_selectors_of_the_key_inner_product() {
        let key = toy_key(&[0, 3, 5, 9], 0);
        let (program, prf) = toy_prf(std::slice::from_ref(&key));
        assert!(matches!(prf.output(), PrfOutput::Scalar(_)));
        assert!(program.data_families().is_empty() && prf.tables().is_empty());
        let workload = crate::rns_bgg::RnsBggDerivedWorkload::from_program(&program).unwrap();
        let families = prf.selector_families();
        assert_eq!(families.len(), key.public_layout.parameters.bucket_count);
        let parameters =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(32, 1, 17, 4, None, None);
        let (plaintexts, tables) = prf
            .key_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(
                &parameters,
                std::slice::from_ref(&key),
            )
            .unwrap();
        assert_eq!(tables, crate::program::ProgramData::new());
        tables.validate(&program).unwrap();
        prf.check_tables(&tables).unwrap();
        let q = toy_profile().q().unwrap() as u64;
        let a = (0..10_u64).map(|coordinate| (7 * coordinate + 3) % q).collect::<Vec<_>>();
        let mut accumulator = 0;
        for (bucket, family) in families.iter().enumerate() {
            assert_eq!(family.members, key.public_layout.cells[bucket].len());
            assert_eq!(family.max_nonzero, 1);
            assert_eq!(workload.rhs_level(family.rhs).unwrap(), bucket);
            let bits = bits(&parameters, &plaintexts, family.rhs);
            assert_eq!(bits.iter().sum::<u64>(), 1);
            let slot = bits.iter().position(|bit| *bit == 1).unwrap();
            if let PbcCell::Real { coordinate, .. } = key.public_layout.cells[bucket][slot] {
                accumulator = (accumulator + a[coordinate]) % q;
            }
        }
        assert_eq!(
            accumulator,
            crate::program_gadgets::pbc::clear_pbc_inner_product(
                &key.public_layout,
                key.private_schedule(),
                &a,
                q
            )
            .unwrap()
        );
        let other = toy_key(&[1, 2, 6, 7], 100);
        assert!(
            prf.key_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&parameters, &[other])
                .is_err()
        );
    }

    /// Several members: one member-indexed selector family per bucket,
    /// holding the slot bit of every member's key in a row of the width
    /// limit, and one coordinate table per bucket holding every cell's
    /// coordinate; the Program depends on the key shape only, and keys or
    /// tables of another key set are rejected.
    #[test]
    fn member_prf_has_one_row_one_hot_selector_family_and_table_per_bucket() {
        let keys = [toy_key(&[0, 3, 5, 9], 0), toy_key(&[1, 2, 6, 7], 100)];
        let (program, prf) = toy_prf(&keys);
        assert!(matches!(prf.output(), PrfOutput::Family(_)));
        let width = 8;
        let buckets = keys[0].public_layout.parameters.bucket_count;
        let families = prf.selector_families();
        assert_eq!(families.len(), buckets);
        assert_eq!(program.rhs_inputs().len(), buckets);
        assert_eq!(program.data_families().len(), buckets);
        assert!(
            program
                .data_families()
                .values()
                .all(|table| table.count() == 2 * width && table.bound() == Some(11))
        );
        assert!(families.iter().all(|family| family.members == 2 * width &&
            family.max_nonzero == 1 &&
            family.row_length == Some(width)));
        let parameters =
            mxx_backends::poly::dcrt::params::DCRTPolyParams::new(32, 1, 17, 4, None, None);
        let (plaintexts, tables) =
            prf.key_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&parameters, &keys).unwrap();
        tables.validate(&program).unwrap();
        prf.check_tables(&tables).unwrap();
        for (bucket, (family, table)) in families.iter().zip(prf.tables()).enumerate() {
            let expected = keys
                .iter()
                .flat_map(|key| {
                    let selected = key.private_schedule().selected_slot(bucket);
                    (0..width).map(move |slot| u64::from(slot == selected))
                })
                .collect::<Vec<_>>();
            assert_eq!(bits(&parameters, &plaintexts, family.rhs), expected);
            let coordinates = keys
                .iter()
                .flat_map(|key| {
                    (0..width).map(move |slot| match key.public_layout.cells[bucket].get(slot) {
                        Some(PbcCell::Real { coordinate, .. }) => *coordinate,
                        _ => 10,
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(tables.values(*table).unwrap(), coordinates);
        }
        let others = [toy_key(&[0, 3, 5, 9], 50), toy_key(&[1, 2, 6, 7], 150)];
        let (other_program, _) = toy_prf(&others);
        assert_eq!(other_program.id(), program.id(), "the Program depends on the shape only");
        assert!(
            prf.key_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&parameters, &others)
                .is_err()
        );
        let (_, other_tables) = toy_prf(&others)
            .1
            .key_inputs::<mxx_backends::poly::dcrt::poly::DCRTPoly>(&parameters, &others)
            .unwrap();
        assert_eq!(prf.check_tables(&other_tables), Err(PrfError::KeyShapeMismatch));
    }

    #[test]
    fn keys_must_exist_share_a_width_limit_and_cover_the_lwr_dimension() {
        let key = toy_key(&[0, 3, 5, 9], 0).public_layout;
        let build = |profile, keys| {
            SparseLwrPrf::build(
                &mut ExponentLutProgramBuilder::new(),
                profile,
                keys,
                LutOutputRepresentation::Coefficient,
            )
        };
        assert!(PrfKeyShape::new(&[]).is_err());
        assert!(PrfKeyShape::new(std::slice::from_ref(&key)).is_err(), "one member");
        let unlimited = crate::program_gadgets::pbc::generate_key_layout(
            &PbcParameters::paper_evaluation(10, 4),
            crate::program_gadgets::pbc::PbcRootSeed([7; 32]),
            &[0, 3, 5, 9],
        )
        .unwrap()
        .public_layout;
        assert!(PrfKeyShape::new(&[unlimited.clone(), unlimited]).is_err(), "no width limit");
        let wider =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4, 11).unwrap();
        assert!(build(wider, PrfKeys::Layout(key.clone())).is_err());
        let shape = PrfKeyShape::new(&[key.clone(), key.clone()]).unwrap();
        let committed = |members| {
            PrfKeyShape::committed(key.parameters.clone(), members, shape.key_set, shape.tables)
        };
        assert_eq!(committed(2), Ok(shape.clone()), "the digests of the layouts");
        assert!(committed(1).is_err(), "one member");
        let unlimited_parameters = PbcParameters { bucket_width_limit: None, ..key.parameters };
        assert!(PrfKeyShape::committed(unlimited_parameters, 2, [0; 32], [0; 32]).is_err());
        assert!(build(wider, PrfKeys::Shape(shape.clone())).is_err());
        assert!(build(toy_profile(), PrfKeys::Shape(shape)).is_ok());
    }

    #[test]
    fn tiny_native_reference_oracle_covers_every_z_mod_q() {
        let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4, 32)
            .expect("n=32, Q=32, P=4 profile");
        assert_eq!(profile.ring_dimension().unwrap(), 32);
        assert_eq!(profile.q().unwrap(), 32);
        assert_eq!(profile.p_prime(), 2);
        assert_eq!(profile.layout.stride_log2, 1);
        let coefficients = profile.canonical_readout().expect("canonical readout");
        assert_eq!(coefficients.len(), 16);

        // Readout represents f(z)-1/2 in doubled centered coordinates.
        // Native negacyclic wrap sends z+n' to the additive inverse, so both
        // halves of Z_Q are covered without an out-of-range table lookup.
        let doubled = (0..32)
            .map(|z| {
                let k = z % 16;
                let value = BigInt::from(coefficients[k]) * BigInt::from(2_u8) - BigInt::from(1_u8);
                if z < 16 { value } else { -value }
            })
            .collect::<Vec<_>>();
        for z in 0..16 {
            let expected = BigInt::from(2 * (2 * z / 16 + 1) - 1);
            assert_eq!(doubled[z], expected);
            assert_eq!(doubled[z + 16], -expected);
        }
    }
}
