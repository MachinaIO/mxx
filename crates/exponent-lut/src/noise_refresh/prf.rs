//! Scheme-independent sparse-LWR PRF declarations and Program interpretation.
use crate::{
    noise_refresh::pbc::{PbcCell, PbcGeneratedKeyLayout, PbcLayoutId, PbcPublicLayout},
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder,
        ExponentLutProgramId, FamilyRange, LutOutputRepresentation, LutTable, Operand,
        ProgramValidationError, RhsId, WireId,
    },
};
use mxx_ir_core::IntExpr;
use serde::{Deserialize, Serialize};
use sha2::Digest;

/// Scheme-independent errors in native PRF declarations and bindings.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum PrfError {
    #[error("invalid native sparse-LWR profile, block layout or input binding")]
    InvalidSparseLwrBlock,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
    #[error("PRF artifact binding or transport failed: {0}")]
    Artifact(String),
}

/// Stable identity of one value-readout declaration.
///
/// The digest is an opaque declaration identity; interpretation of the
/// readout table remains owned by the concrete backend adapter.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct LutOutputIdentity([u8; 32]);

impl LutOutputIdentity {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Opaque, canonically encoded public PRF label.
///
/// The refresh application owns the meaning of a label's fields.  The PRF
/// only commits to their unambiguous byte encoding and never parses or
/// derives semantics from them.  Length-prefixing each component keeps the
/// encoding injective while allowing callers to use arbitrary binary tags.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize)]
pub struct PrfLabel(Vec<u8>);

impl PrfLabel {
    /// Builds a label from an ordered set of application-owned components.
    pub fn from_components(components: &[&[u8]]) -> Result<Self, PrfError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"mxx-exponent-lut/prf-label/v1");
        let count = u32::try_from(components.len()).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        bytes.extend_from_slice(&count.to_le_bytes());
        for component in components {
            let length =
                u64::try_from(component.len()).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
            bytes.extend_from_slice(&length.to_le_bytes());
            bytes.extend_from_slice(component);
        }
        Ok(Self(bytes))
    }

    /// Wraps an already canonical label encoding after rejecting an empty
    /// value. Callers that construct fields dynamically should prefer
    /// [`Self::from_components`].
    pub fn from_canonical_bytes(bytes: Vec<u8>) -> Result<Self, PrfError> {
        const PREFIX: &[u8] = b"mxx-exponent-lut/prf-label/v1";
        let prefix_len = PREFIX.len();
        if bytes.len() < prefix_len + std::mem::size_of::<u32>() || !bytes.starts_with(PREFIX) {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let mut cursor = prefix_len;
        let count = u32::from_le_bytes(
            bytes[cursor..cursor + 4].try_into().map_err(|_| PrfError::InvalidSparseLwrBlock)?,
        ) as usize;
        cursor += 4;
        for _ in 0..count {
            let end = cursor.checked_add(8).ok_or(PrfError::InvalidSparseLwrBlock)?;
            if end > bytes.len() {
                return Err(PrfError::InvalidSparseLwrBlock);
            }
            let length = u64::from_le_bytes(
                bytes[cursor..end].try_into().map_err(|_| PrfError::InvalidSparseLwrBlock)?,
            );
            cursor = end;
            let length = usize::try_from(length).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
            cursor = cursor.checked_add(length).ok_or(PrfError::InvalidSparseLwrBlock)?;
            if cursor > bytes.len() {
                return Err(PrfError::InvalidSparseLwrBlock);
            }
        }
        if cursor != bytes.len() {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for PrfLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Self::from_canonical_bytes(bytes).map_err(serde::de::Error::custom)
    }
}

/// Native sparse-LWR profile.
///
/// The exponent layout fixes the encoding ring dimension the layout lives in,
/// `n = 2^(stride_log2 + width_log2)`, the exponent modulus
/// `Q = 2^(width_log2 + 1) = 2n / 2^stride_log2`, and the balanced digit base
/// `P = 2*p_prime`. The LWR dimension `lwr_dimension` (the length of the
/// public input `a` in `Z_Q^nu`) is independent of the ring dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p_prime: usize,
    pub lwr_dimension: usize,
}

impl SparseLwrPrfProfile {
    /// Validates a deserialized or otherwise externally assembled profile.
    /// Public fields are intentional for declarative manifests, therefore all
    /// consuming boundaries must call this check instead of trusting them.
    pub fn validate(&self) -> Result<(), PrfError> {
        if self.p_prime == 0 || self.lwr_dimension == 0 {
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
        let p = self.p_prime.checked_mul(2).ok_or(PrfError::InvalidSparseLwrBlock)?;
        if p < 2 || p > q || q % p != 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    /// Validates a native profile; its ring is derived from the full layout.
    pub fn new(layout: ExpLayout, p_prime: usize, lwr_dimension: usize) -> Result<Self, PrfError> {
        let profile = Self { layout, p_prime, lwr_dimension };
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

    pub fn p(&self) -> Result<usize, PrfError> {
        self.p_prime.checked_mul(2).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    pub const fn native_width(&self) -> usize {
        1usize << self.layout.width_log2
    }

    /// Returns the canonical sparse-LWR value table
    /// `f(k) = floor(p' * k / n') + 1` for `0 <= k < n'`.
    pub fn canonical_readout(&self) -> Result<Vec<usize>, PrfError> {
        let n_prime = self.native_width();
        let values = (0..n_prime)
            .map(|k| self.p_prime.checked_mul(k).map(|value| value / n_prime + 1))
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
            p_prime: usize,
            lwr_dimension: usize,
        }
        let raw = RawProfile::deserialize(deserializer)?;
        let profile =
            Self { layout: raw.layout, p_prime: raw.p_prime, lwr_dimension: raw.lwr_dimension };
        profile.validate().map_err(serde::de::Error::custom)?;
        Ok(profile)
    }
}

/// A sparse-key LWR weak PRF evaluated over encodings.
///
/// Every output `o` computes `wPRF_{k_o}(a) = rho(<a, k_o> mod Q)` on the
/// same public vector `a in Z_Q^nu`, with its own sparse key `k_o`. Key `k_o`
/// is represented by its own probabilistic-batch-code layout: block `j` of
/// that layout holds candidate coordinates of `a`, and a private one-hot
/// selector chooses the block's coordinate of the key support (or a dummy,
/// contributing zero). Starting from the accumulator `mono(0)`, block `j`
/// multiplies the accumulator by each candidate `a` coordinate (a
/// public-by-private product), weights the branch by its selector bit (a
/// private multiplication), and sums the branches; a dummy or padding cell
/// weights the accumulator itself. A terminal coefficient LUT reads out
/// `rho`.
///
/// Block `j` runs at level `j`, so `a` is supplied as one public-payload
/// input family per block: the coordinates that block reads first, over all
/// outputs. Each family is then encoded at the level of its first use rather
/// than at level zero. Within block `j`, the cells are grouped by the family
/// of their coordinate (and the dummy and padding cells form one more
/// group); each group has its own selector family and one family body, and
/// the block sums the groups.
#[derive(Clone)]
pub struct SparseLwrPrfProgram {
    pub accumulation: ExponentLutProgram,
    pub readout: Vec<usize>,
    pub(crate) initial_input: WireId,
    pub(crate) output: WireId,
    pub(crate) profile: SparseLwrPrfProfile,
    /// Public input families of `a` and the coordinates of their members, one
    /// family per block that first reads them.
    input_families: Vec<(WireId, Vec<usize>)>,
    /// Public key layout of each output, in output order.
    layouts: Vec<PbcPublicLayout>,
    /// One selector family per output and block.
    selection_families: Vec<PrfSelectionFamily>,
    output_labels: Vec<PrfLabel>,
    outputs: Vec<WireId>,
}

/// The one-hot selector family of one cell group of one output's block.
///
/// The ID refers to a declaration in [`SparseLwrPrfProgram::accumulation`].
/// Member `i` selects cell `slots[i]` of block `bucket` in the output's layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrfSelectionFamily {
    pub output: usize,
    pub bucket: usize,
    pub selector: RhsId,
    pub capacity: usize,
    pub slots: Vec<usize>,
}

/// Identity of the setup-owned PRF declaration.
///
/// It contains only public Program and layout facts plus the selector key
/// namespace. It excludes the public vector `a` and all selector bits, so one
/// setup serves every encryption of any `a`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseLwrPrfSetupIdentity {
    pub program: ExponentLutProgramId,
    /// Public key layout of each output, in output order.
    pub layouts: Vec<PbcLayoutId>,
    /// Public key-instance namespace of the selector families.
    pub input_key_identity: [u8; 32],
}

impl std::fmt::Debug for SparseLwrPrfProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparseLwrPrfProgram")
            .field("id", &self.id())
            .field("profile", &self.profile)
            .field("outputs", &self.outputs.len())
            .field("readout", &self.readout)
            .field("selection_families", &self.selection_families.len())
            .finish()
    }
}

impl SparseLwrPrfProgram {
    /// Builds the PRF declaration for one key layout per output label.
    ///
    /// Each layout must be a valid PBC layout over the profile's LWR
    /// dimension. The constructor contains no selector bits and no value of
    /// `a`; both are runtime inputs.
    pub fn new(
        profile: SparseLwrPrfProfile,
        layouts: Vec<PbcPublicLayout>,
        output_labels: Vec<PrfLabel>,
    ) -> Result<Self, PrfError> {
        profile.validate()?;
        if output_labels.is_empty() || layouts.len() != output_labels.len() {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let mut seen = std::collections::BTreeSet::new();
        if !output_labels.iter().all(|label| seen.insert(label.clone())) {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        for layout in &layouts {
            layout.validate().map_err(|_| PrfError::InvalidSparseLwrBlock)?;
            if layout.parameters.universe_size != profile.lwr_dimension {
                return Err(PrfError::InvalidSparseLwrBlock);
            }
        }
        let readout = profile.canonical_readout()?;
        let mut builder = ExponentLutProgramBuilder::new();
        let initial_input = builder.input(AttributeKnowledge::PrivatePayload)?;
        // The block that first reads each coordinate, over every output.
        let mut first_block = vec![None::<usize>; profile.lwr_dimension];
        for layout in &layouts {
            for (bucket, cells) in layout.cells.iter().enumerate() {
                for cell in cells {
                    if let PbcCell::Real { coordinate, .. } = cell {
                        let first = first_block
                            .get_mut(*coordinate)
                            .ok_or(PrfError::InvalidSparseLwrBlock)?;
                        *first = Some(first.map_or(bucket, |first| first.min(bucket)));
                    }
                }
            }
        }
        let mut coordinates_by_block = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (coordinate, block) in first_block.iter().enumerate() {
            if let Some(block) = block {
                coordinates_by_block.entry(*block).or_default().push(coordinate);
            }
        }
        // Coordinate -> (input family position, member).
        let mut location = vec![None::<(usize, usize)>; profile.lwr_dimension];
        let mut input_families = Vec::with_capacity(coordinates_by_block.len());
        for coordinates in coordinates_by_block.into_values() {
            let wire = builder.input_family(
                IntExpr::constant(coordinates.len()),
                AttributeKnowledge::PublicPayload,
            )?;
            for (member, coordinate) in coordinates.iter().enumerate() {
                location[*coordinate] = Some((input_families.len(), member));
            }
            input_families.push((wire, coordinates));
        }
        // The accumulator stays in exponent representation; this terminal
        // LUT performs the sole conversion to the coefficient-valued output.
        let lut = builder.register_lut(LutTable::unary(
            profile.native_width(),
            profile.layout,
            profile.native_width(),
            readout.clone(),
            LutOutputRepresentation::Coefficient,
        )?)?;
        let mut selection_families = Vec::new();
        let mut outputs = Vec::with_capacity(layouts.len());
        for (output, layout) in layouts.iter().enumerate() {
            let mut state = initial_input;
            for (bucket, cells) in layout.cells.iter().enumerate() {
                // The cells of this block by input family; `None` holds the
                // dummy and padding cells.
                let mut groups = std::collections::BTreeMap::<Option<usize>, Vec<usize>>::new();
                for (slot, cell) in cells.iter().enumerate() {
                    let group = match cell {
                        PbcCell::Real { coordinate, .. } => {
                            Some(location[*coordinate].ok_or(PrfError::InvalidSparseLwrBlock)?.0)
                        }
                        PbcCell::Dummy | PbcCell::Padding => None,
                    };
                    groups.entry(group).or_default().push(slot);
                }
                let mut block_sum = None;
                for (group, slots) in groups {
                    // Block `bucket` is consumed by the level-`bucket`
                    // private multiplication, so each group's selector is its
                    // own RHS family. The one-hot selector bounds the block
                    // sum across all of its groups.
                    let selector = builder.rhs_family(IntExpr::constant(slots.len()))?;
                    let members = match group {
                        Some(_) => slots
                            .iter()
                            .map(|slot| match &cells[*slot] {
                                PbcCell::Real { coordinate, .. } => location[*coordinate]
                                    .map(|(_, member)| member)
                                    .ok_or(PrfError::InvalidSparseLwrBlock),
                                _ => Err(PrfError::InvalidSparseLwrBlock),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        None => Vec::new(),
                    };
                    let source = group.map(|group| input_families[group].0);
                    // One cell: `slot` selects it, and its `a` member is `member`.
                    // A real cell yields X^(acc + a_r) weighted by its selector
                    // bit; a dummy or padding cell weights X^acc itself.
                    let cell = |builder: &mut ExponentLutProgramBuilder,
                                slot: IntExpr,
                                member: IntExpr|
                     -> Result<Operand<WireId>, ProgramValidationError> {
                        let weighted_input = match source {
                            Some(wire) => builder.mul_public(
                                Operand::Member { family: wire, index: member },
                                Operand::Scalar(state),
                            )?,
                            None => state,
                        };
                        let weighted = builder.mul_private(
                            Operand::Scalar(weighted_input),
                            Operand::Member { family: selector, index: slot },
                        )?;
                        Ok(Operand::Scalar(weighted))
                    };
                    let sum = if slots.len() == 1 {
                        // A group of one cell needs no family body.
                        let member = IntExpr::constant(members.first().copied().unwrap_or(0));
                        match cell(&mut builder, IntExpr::constant(0), member)? {
                            Operand::Scalar(wire) => wire,
                            Operand::Member { .. } => return Err(PrfError::InvalidSparseLwrBlock),
                        }
                    } else {
                        let range = FamilyRange::full(IntExpr::constant(slots.len()))?;
                        let family = builder.parallel(range, |builder, slot| {
                            let member = IntExpr::Select {
                                selector: Box::new(slot.clone()),
                                branches: members.iter().copied().map(IntExpr::constant).collect(),
                            };
                            cell(builder, slot, member)
                        })?;
                        builder.sum(family)?
                    };
                    block_sum = Some(match block_sum {
                        Some(previous) => {
                            builder.add(Operand::Scalar(previous), Operand::Scalar(sum))?
                        }
                        None => sum,
                    });
                    selection_families.push(PrfSelectionFamily {
                        output,
                        bucket,
                        selector,
                        capacity: slots.len(),
                        slots,
                    });
                }
                state = block_sum.ok_or(PrfError::InvalidSparseLwrBlock)?;
            }
            outputs.push(builder.lut(Operand::Scalar(state), lut)?);
        }
        let output = *outputs.first().ok_or(PrfError::InvalidSparseLwrBlock)?;
        let accumulation =
            builder.finish(outputs.iter().copied().map(Operand::Scalar).collect())?;
        Ok(Self {
            accumulation,
            readout,
            initial_input,
            output,
            profile,
            input_families,
            layouts,
            selection_families,
            output_labels,
            outputs,
        })
    }

    /// The selector families of every output's blocks, output-major and in
    /// block order; a block has one family per cell group.
    pub fn selection_families(&self) -> &[PrfSelectionFamily] {
        &self.selection_families
    }

    /// The public input families of `a`, each with the LWR coordinates of its
    /// members, in Program declaration order. A coordinate that no block reads
    /// is in none of them.
    pub fn input_families(&self) -> &[(WireId, Vec<usize>)] {
        &self.input_families
    }

    /// The public key layout of each output, in output order.
    pub fn layouts(&self) -> &[PbcPublicLayout] {
        &self.layouts
    }

    /// Ordered scalar roots and their opaque application labels. The order is
    /// the immutable Program root order.
    pub fn ordered_outputs(&self) -> impl Iterator<Item = (&PrfLabel, WireId)> {
        self.output_labels.iter().zip(self.outputs.iter().copied())
    }

    /// One-hot selector bits of `output`'s blocks for a generated key,
    /// block-major; each block has one bit per cell of its layout row.
    pub fn selector_bits(
        &self,
        output: usize,
        key: &PbcGeneratedKeyLayout,
    ) -> Result<Vec<Vec<u8>>, PrfError> {
        let layout = self.layouts.get(output).ok_or(PrfError::InvalidSparseLwrBlock)?;
        if key.public_layout.layout_id != layout.layout_id {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        key.private_schedule().validate(layout).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        Ok(layout
            .cells
            .iter()
            .enumerate()
            .map(|(bucket, cells)| {
                let selected = key.private_schedule().selected_slot(bucket);
                (0..cells.len()).map(|slot| u8::from(slot == selected)).collect()
            })
            .collect())
    }

    /// Setup identity for a selector key namespace.
    pub fn setup_identity(&self, selector_key_instance: [u8; 32]) -> SparseLwrPrfSetupIdentity {
        SparseLwrPrfSetupIdentity {
            program: self.id(),
            layouts: self.layouts.iter().map(|layout| layout.layout_id).collect(),
            input_key_identity: selector_key_instance,
        }
    }

    pub const fn profile(&self) -> SparseLwrPrfProfile {
        self.profile
    }

    pub const fn initial_input(&self) -> WireId {
        self.initial_input
    }

    /// Returns the accumulator's initial native attribute `mono_m(0)`.
    pub fn initial_attribute(&self, ring: &mxx_dsl::Ring) -> Result<mxx_dsl::Mat, PrfError> {
        self.profile
            .layout
            .mono(ring, &num_bigint::BigInt::from(0))
            .map_err(|_| PrfError::InvalidSparseLwrBlock)
    }

    pub const fn output(&self) -> WireId {
        self.output
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.accumulation.id()
    }

    /// Returns the complete immutable declaration, including the terminal
    /// coefficient LUTs.
    pub fn program(&self) -> &ExponentLutProgram {
        &self.accumulation
    }
}

pub fn validate_profile_program(
    profile: SparseLwrPrfProfile,
    program: &SparseLwrPrfProgram,
) -> Result<(), PrfError> {
    profile.validate()?;
    program.profile.validate()?;
    if program.profile != profile || program.readout != profile.canonical_readout()? {
        return Err(PrfError::InvalidSparseLwrBlock);
    }
    Ok(())
}

/// PRF preparation descriptor. Cryptographic helper caches are owned by the
/// concrete evaluator; this type records only the Program and the setup
/// identity needed to reopen them.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedSparseLwrPrfDescriptor {
    pub program: ExponentLutProgramId,
    pub setup: SparseLwrPrfSetupIdentity,
}

impl PreparedSparseLwrPrfDescriptor {
    pub fn identity(&self) -> Result<[u8; 32], PrfError> {
        let bytes =
            serde_json::to_vec(self).map_err(|error| PrfError::Artifact(error.to_string()))?;
        let mut digest = sha2::Sha256::new();
        digest.update(b"mxx-exponent-lut/sparse-lwr/preparation");
        digest.update(bytes);
        Ok(digest.finalize().into())
    }
}

/// PRF preparation state shared by preprocessing and online evaluation. It
/// carries no value of `a` and no secret material.
#[derive(Clone, Debug)]
pub struct PreparedSparseLwrPrf {
    program: SparseLwrPrfProgram,
    setup: SparseLwrPrfSetupIdentity,
}

impl PartialEq for PreparedSparseLwrPrf {
    fn eq(&self, other: &Self) -> bool {
        // The declaration is compared through its canonical descriptor; the
        // embedded graph handles are not value-comparable.
        self.descriptor() == other.descriptor()
    }
}

impl Eq for PreparedSparseLwrPrf {}

impl PreparedSparseLwrPrf {
    pub fn new(
        program: SparseLwrPrfProgram,
        selector_key_instance: [u8; 32],
    ) -> Result<Self, PrfError> {
        validate_profile_program(program.profile(), &program)?;
        let setup = program.setup_identity(selector_key_instance);
        Ok(Self { program, setup })
    }

    pub fn descriptor(&self) -> PreparedSparseLwrPrfDescriptor {
        PreparedSparseLwrPrfDescriptor { program: self.program.id(), setup: self.setup.clone() }
    }

    pub fn import(
        program: SparseLwrPrfProgram,
        descriptor: &PreparedSparseLwrPrfDescriptor,
        expected_identity: [u8; 32],
    ) -> Result<Self, PrfError> {
        validate_profile_program(program.profile(), &program)?;
        if descriptor.identity()? != expected_identity || descriptor.program != program.id() {
            return Err(PrfError::Artifact("PRF identity mismatch".into()));
        }
        if descriptor.setup != program.setup_identity(descriptor.setup.input_key_identity) {
            return Err(PrfError::Artifact("PRF setup mismatch".into()));
        }
        Ok(Self { program, setup: descriptor.setup.clone() })
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program.accumulation
    }

    pub fn declaration(&self) -> &SparseLwrPrfProgram {
        &self.program
    }

    pub fn setup_identity(&self) -> &SparseLwrPrfSetupIdentity {
        &self.setup
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::InputBinding;
    use num_bigint::BigInt;
    use std::collections::BTreeSet;

    #[test]
    fn profile_derives_q_and_p_for_nonbinary_digit_bases() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8, 64).unwrap();
        assert_eq!(profile.native_width(), 32);
        assert_eq!(profile.q().unwrap(), 64);
        assert_eq!(profile.p().unwrap(), 16);
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 6 }, 16, 128).unwrap();
        assert_eq!(profile.q().unwrap(), 128);
        assert_eq!(profile.p().unwrap(), 32);
    }

    #[test]
    fn profile_rejects_nondividing_digit_base() {
        assert!(
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 12, 64).is_err()
        );
    }

    #[test]
    fn profile_preserves_zero_and_nonzero_stride_input_layouts() {
        let zero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 0, width_log2: 4 }, 2, 16).unwrap();
        let nonzero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 2, 16).unwrap();

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
    fn canonical_labels_are_stable_and_injective() {
        let first_slot = 7_u64.to_le_bytes();
        let second_slot = 8_u64.to_le_bytes();
        let first =
            PrfLabel::from_components(&[b"mask".as_slice(), first_slot.as_slice()]).unwrap();
        let second =
            PrfLabel::from_components(&[b"mask".as_slice(), second_slot.as_slice()]).unwrap();
        assert_ne!(first, second);
        assert_eq!(PrfLabel::from_canonical_bytes(first.as_bytes().to_vec()).unwrap(), first);
        assert!(PrfLabel::from_canonical_bytes(b"not-a-label".to_vec()).is_err());
        assert!(serde_json::from_str::<PrfLabel>("[1, 2, 3]").is_err());
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
                p_prime,
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
            "p_prime": 12,
        });
        assert!(serde_json::from_value::<SparseLwrPrfProfile>(encoded).is_err());
    }

    /// n=32, Q=32, p'=2 over an LWR dimension of ten.
    fn toy_profile() -> SparseLwrPrfProfile {
        SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 2, 10).unwrap()
    }

    /// A key of weight four with its own PBC layout over ten coordinates.
    /// The layout depends only on the root seed, so distinct keys search
    /// distinct seed ranges starting at `seed_base`.
    fn toy_key(support: &[usize], seed_base: u8) -> PbcGeneratedKeyLayout {
        let parameters = crate::noise_refresh::pbc::PbcParameters::paper_evaluation(10, 4);
        (seed_base..seed_base.saturating_add(32))
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    support,
                )
                .ok()
            })
            .expect("toy key must find a PBC layout")
    }

    fn toy_labels(count: usize) -> Vec<PrfLabel> {
        (0..count as u64)
            .map(|index| PrfLabel::from_components(&[index.to_le_bytes().as_slice()]).unwrap())
            .collect()
    }

    #[test]
    fn outputs_share_a_and_each_output_block_has_its_own_selectors() {
        let keys = [toy_key(&[0, 3, 5, 9], 0), toy_key(&[1, 2, 6, 7], 100)];
        let layouts = keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>();
        let program =
            SparseLwrPrfProgram::new(toy_profile(), layouts.clone(), toy_labels(2)).unwrap();
        let workload =
            crate::rns_bgg::RnsBggDerivedWorkload::from_program(program.program()).unwrap();
        let families = program.selection_families();
        let selectors = families.iter().map(|family| family.selector).collect::<BTreeSet<_>>();
        assert_eq!(selectors.len(), families.len(), "every cell group has its own selector");
        for (output, layout) in layouts.iter().enumerate() {
            for (bucket, cells) in layout.cells.iter().enumerate() {
                // The groups of a block partition its cells.
                let mut slots = families
                    .iter()
                    .filter(|family| family.output == output && family.bucket == bucket)
                    .flat_map(|family| family.slots.iter().copied())
                    .collect::<Vec<_>>();
                slots.sort_unstable();
                assert_eq!(slots, (0..cells.len()).collect::<Vec<_>>());
            }
        }
        for family in families {
            assert_eq!(family.capacity, family.slots.len());
            // Block `b` is consumed by the level-`b` private multiplication.
            assert_eq!(workload.rhs_level(family.selector).unwrap(), family.bucket);
        }
        // Each input family holds the coordinates one block reads first, over
        // both outputs, and is encoded at that block's level.
        let mut covered = Vec::new();
        for (wire, coordinates) in program.input_families() {
            let first = coordinates
                .iter()
                .map(|coordinate| {
                    layouts
                        .iter()
                        .filter_map(|layout| {
                            layout.cells.iter().position(|cells| {
                                cells.iter().any(|cell| {
                                    matches!(cell, PbcCell::Real { coordinate: c, .. } if c == coordinate)
                                })
                            })
                        })
                        .min()
                        .unwrap()
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(first.len(), 1, "one first block per input family");
            assert_eq!(workload.level(*wire).unwrap(), *first.first().unwrap());
            covered.extend(coordinates.iter().copied());
        }
        covered.sort_unstable();
        covered.dedup();
        let read = layouts
            .iter()
            .flat_map(|layout| layout.cells.iter().flatten())
            .filter_map(|cell| match cell {
                PbcCell::Real { coordinate, .. } => Some(*coordinate),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(covered, read.into_iter().collect::<Vec<_>>());
        let public_families = program
            .program()
            .inputs()
            .iter()
            .filter(|(_, declaration)| matches!(declaration.shape(), InputBinding::Family(_)))
            .map(|(wire, _)| *wire)
            .collect::<Vec<_>>();
        assert_eq!(
            public_families,
            program.input_families().iter().map(|(wire, _)| *wire).collect::<Vec<_>>()
        );
        assert_eq!(program.ordered_outputs().count(), 2);
    }

    #[test]
    fn selector_bits_are_one_hot_and_select_the_key_inner_product() {
        let keys = [toy_key(&[0, 3, 5, 9], 0), toy_key(&[1, 2, 6, 7], 100)];
        let program = SparseLwrPrfProgram::new(
            toy_profile(),
            keys.iter().map(|key| key.public_layout.clone()).collect(),
            toy_labels(2),
        )
        .unwrap();
        let q = program.profile().q().unwrap() as u64;
        let a = (0..10_u64).map(|coordinate| (7 * coordinate + 3) % q).collect::<Vec<_>>();
        for (output, key) in keys.iter().enumerate() {
            let bits = program.selector_bits(output, key).unwrap();
            let mut accumulator = 0;
            for (bucket, block) in bits.iter().enumerate() {
                assert_eq!(block.iter().map(|bit| u64::from(*bit)).sum::<u64>(), 1);
                let slot = block.iter().position(|bit| *bit == 1).unwrap();
                if let PbcCell::Real { coordinate, .. } = key.public_layout.cells[bucket][slot] {
                    accumulator = (accumulator + a[coordinate]) % q;
                }
            }
            assert_eq!(
                accumulator,
                crate::noise_refresh::pbc::clear_pbc_inner_product(
                    &key.public_layout,
                    key.private_schedule(),
                    &a,
                    q
                )
                .unwrap()
            );
        }
        assert!(program.selector_bits(0, &keys[1]).is_err(), "a key of another layout");
    }

    #[test]
    fn layouts_must_cover_the_lwr_dimension_and_match_the_labels() {
        let key = toy_key(&[0, 3, 5, 9], 0);
        let layouts = vec![key.public_layout.clone()];
        assert!(SparseLwrPrfProgram::new(toy_profile(), layouts.clone(), toy_labels(2)).is_err());
        assert!(SparseLwrPrfProgram::new(toy_profile(), vec![], vec![]).is_err());
        let duplicate = vec![toy_labels(1)[0].clone(), toy_labels(1)[0].clone()];
        assert!(
            SparseLwrPrfProgram::new(toy_profile(), vec![layouts[0].clone(); 2], duplicate)
                .is_err()
        );
        let wider =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 2, 11).unwrap();
        assert!(SparseLwrPrfProgram::new(wider, layouts, toy_labels(1)).is_err());
    }

    #[test]
    fn prepared_descriptor_requires_an_external_trusted_identity() {
        let program = SparseLwrPrfProgram::new(
            toy_profile(),
            vec![toy_key(&[0, 3, 5, 9], 0).public_layout],
            toy_labels(1),
        )
        .unwrap();
        let prepared = PreparedSparseLwrPrf::new(program.clone(), [9; 32]).unwrap();
        assert_eq!(prepared.setup_identity().input_key_identity, [9; 32]);
        let descriptor = prepared.descriptor();
        assert!(PreparedSparseLwrPrf::import(program.clone(), &descriptor, [0; 32]).is_err());
        let other = SparseLwrPrfProgram::new(
            toy_profile(),
            vec![toy_key(&[1, 2, 6, 7], 100).public_layout],
            toy_labels(1),
        )
        .unwrap();
        assert!(
            PreparedSparseLwrPrf::import(other, &descriptor, descriptor.identity().unwrap())
                .is_err()
        );
        assert!(
            PreparedSparseLwrPrf::import(program, &descriptor, descriptor.identity().unwrap())
                .is_ok()
        );
    }

    #[test]
    fn profile_program_mismatch_is_rejected() {
        let profile = toy_profile();
        let program = SparseLwrPrfProgram::new(
            profile,
            vec![toy_key(&[0, 3, 5, 9], 0).public_layout],
            toy_labels(1),
        )
        .unwrap();
        let other =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 2, 10).unwrap();
        assert!(validate_profile_program(other, &program).is_err());

        let mut noncanonical = program.clone();
        noncanonical.readout[0] += 1;
        assert!(validate_profile_program(profile, &noncanonical).is_err());
    }

    #[test]
    fn tiny_native_reference_oracle_covers_every_z_mod_q() {
        let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 2, 32)
            .expect("n=32, Q=32, p'=2 profile");
        assert_eq!(profile.ring_dimension().unwrap(), 32);
        assert_eq!(profile.q().unwrap(), 32);
        assert_eq!(profile.p_prime, 2);
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
