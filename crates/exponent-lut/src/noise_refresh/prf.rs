//! Scheme-independent sparse-LWR PRF declarations and Program interpretation.
use crate::{
    noise_refresh::pbc::{
        BalancedProductSchedule, BinaryTransformMatrix, PbcActiveCellIndex, PbcLayoutId,
        PbcPublicLayout, PbcPublicVectorFamilyBinding, PbcSelectorRuntimeBinding,
    },
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder,
        ExponentLutProgramId, FamilyRange, LutOutputRepresentation, LutTable, Operand,
        ProgramValidationError, RhsId, WireId,
    },
};
use mxx_ir_core::IntExpr;
use serde::{Deserialize, Serialize};
use sha2::Digest;

/// Reduces a non-empty value list with a balanced pairwise tree.
///
/// An odd final value is carried to the next level unchanged, matching
/// [`BalancedProductSchedule`]. `None` represents an empty input product.
fn reduce_balanced<T, E>(
    mut level: Vec<T>,
    mut multiply: impl FnMut(T, T) -> Result<T, E>,
) -> Result<Option<T>, E> {
    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len().div_ceil(2));
        let mut children = level.into_iter();
        while let Some(lhs) = children.next() {
            if let Some(rhs) = children.next() {
                next_level.push(multiply(lhs, rhs)?);
            } else {
                next_level.push(lhs);
            }
        }
        level = next_level;
    }
    Ok(level.pop())
}

/// Apply a balanced product of public factors to an accumulator.
///
/// Public factors are first reduced pairwise, carrying an odd final factor to
/// the next level unchanged. The resulting public root is then applied once to
/// `rhs`, which may be private. Thus `k` public factors require
/// `ceil(log2(k)) + 1` multiplication depth for `k > 1`, rather than a linear
/// chain of `k` accumulator updates. `MulPublic` remains public-left at every
/// gate, including the final action on `rhs`.
fn apply_public_product(
    builder: &mut ExponentLutProgramBuilder,
    rhs: WireId,
    leaves: Vec<Operand<WireId>>,
) -> Result<WireId, ProgramValidationError> {
    let public_root =
        reduce_balanced(leaves, |lhs, rhs| builder.mul_public(lhs, rhs).map(Operand::Scalar))?
            .ok_or_else(|| ProgramValidationError::InvalidStructure("empty product".into()))?;
    builder.mul_public(public_root, Operand::Scalar(rhs))
}

fn transform_columns(transform: &BinaryTransformMatrix) -> Option<Vec<usize>> {
    if transform.rows() == 1 {
        return transform.selected_columns(0);
    }
    if transform.columns() == 1 {
        return Some(
            transform
                .bits()
                .iter()
                .enumerate()
                .filter_map(|(row, bit)| (*bit == 1).then_some(row))
                .collect(),
        );
    }
    None
}

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
/// The ring dimension is derived from the native layout as
/// `n = 2^(m + width_log2)`. Consequently the exponent modulus is
/// `Q = 2^(width_log2 + 1)` and the balanced digit base is `P = 2*p_prime`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p_prime: usize,
}

impl SparseLwrPrfProfile {
    /// Validates a deserialized or otherwise externally assembled profile.
    /// Public fields are intentional for declarative manifests, therefore all
    /// consuming boundaries must call this check instead of trusting them.
    pub fn validate(&self) -> Result<(), PrfError> {
        if self.p_prime == 0 {
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

    /// Validates a native profile and derives its ring from the full layout.
    pub fn new(layout: ExpLayout, p_prime: usize) -> Result<Self, PrfError> {
        let profile = Self { layout, p_prime };
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

    /// Number of columns in the public matrix `D`: `v * log2(Q_exp)`.
    /// Keeping this derivation next to the profile prevents callers from
    /// accidentally treating the exponent modulus as the coefficient
    /// modulus or introducing a configurable output-column count.
    pub fn transform_column_count(&self) -> Result<usize, PrfError> {
        self.validate()?;
        let exponent_bits = self
            .layout
            .width_log2
            .checked_add(1)
            .and_then(|bits| usize::try_from(bits).ok())
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        self.ring_dimension()?.checked_mul(exponent_bits).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    /// Derives the fixed single-column binary transform `T_label` used by a
    /// sparse-LWR public matrix. `columns` is the checked D column count; the
    /// returned matrix has exactly `columns` rows and one output column.
    /// Sampling depends only on the setup hash key and the canonical opaque
    /// label, never on label enumeration order or a fresh invocation nonce.
    pub fn derive_label_transform(
        &self,
        hash_key: &[u8],
        label: &PrfLabel,
        columns: usize,
    ) -> Result<BinaryTransformMatrix, PrfError> {
        self.validate()?;
        if columns == 0 || columns != self.transform_column_count()? {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        BinaryTransformMatrix::from_hash(hash_key, label.as_bytes(), columns, 1)
            .map_err(|_| PrfError::InvalidSparseLwrBlock)
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
        }
        let raw = RawProfile::deserialize(deserializer)?;
        let profile = Self { layout: raw.layout, p_prime: raw.p_prime };
        profile.validate().map_err(serde::de::Error::custom)?;
        Ok(profile)
    }
}

/// A native structural accumulator plus its value-form readout.
///
/// `initial_input` is the accumulator state initialized by the evaluator to
/// `mono_m(0)=1`; it is represented as a typed program input rather than a
/// label-dependent declaration. Its accumulation is the structural product
/// chain, shared by preprocessing, evaluation, and noise tracing.
#[derive(Clone)]
pub struct SparseLwrPrfProgram {
    pub accumulation: ExponentLutProgram,
    pub readout: Vec<usize>,
    pub(crate) initial_input: WireId,
    pub(crate) output: WireId,
    pub(crate) profile: SparseLwrPrfProfile,
    pub(crate) bucket_width: usize,
    pub(crate) block_count: usize,
    /// Structural selector/value families for the staged lowering path.
    selection_families: Vec<PrfSelectionFamily>,
    /// One shared encryptor-supplied public D family. Transform selections
    /// are represented by constant indexed operands in the Program AST.
    d0: WireId,
    output_labels: Vec<PrfLabel>,
    outputs: Vec<WireId>,
    /// Canonical label-major binary transforms used by the frozen program.
    /// Keeping these immutable public inputs alongside the declaration lets an
    /// application retain an independent plaintext oracle without inspecting
    /// the frozen graph or reproducing the hash-to-transform convention.
    label_transforms: Vec<BinaryTransformMatrix>,
}

/// Family declarations consumed by one structural PRF bucket body.
///
/// The IDs refer to declarations in [`SparseLwrPrfProgram::accumulation`].
/// Keeping them as metadata lets setup and encryption bind one family body,
/// while the online evaluator supplies D0 and selector members at runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrfSelectionFamily {
    pub selector: RhsId,
    /// Encryptor-supplied coordinate encodings.  This replaces the old
    /// random label carrier; the lowerer combines these wires according to
    /// the setup-owned `T_l` schedule before applying the selector.
    pub d0: WireId,
    pub capacity: usize,
}

/// Identity of the setup-owned structural PRF declaration.
///
/// This identity contains only public Program/layout/schedule facts.  In
/// particular, it deliberately excludes the encryptor's `D0` values and all
/// selector bits, allowing one setup to serve multiple encryptions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseLwrPrfSetupIdentity {
    pub program: ExponentLutProgramId,
    pub layout: PbcLayoutId,
    /// Balanced schedule used to lower the per-label `D0*T_l` product.  This
    /// is independent of the number of bucket invocations and therefore
    /// remains stable when the same label family is reused by every bucket.
    pub label_schedule: BalancedProductSchedule,
    /// Commitment to the public label transforms (`T_l`) and their canonical
    /// order.  The actual matrices are supplied by the public family binding;
    /// no matrix values are copied into setup metadata.
    pub t_l_commitment: [u8; 32],
    /// Public input-key namespace used by selector and D0 family artifacts.
    pub input_key_identity: [u8; 32],
    /// Bucket aggregation schedule retained as part of the structural
    /// declaration. It is distinct from `label_schedule` when a layout has
    /// a different number of buckets and active labels per bucket.
    pub schedule: BalancedProductSchedule,
}

/// Encrypt-stage binding for one structural PRF evaluation.
///
/// The public D0 family is content-bound here, while selector values remain a
/// key-provider runtime input.  No mask/payload secret or selector plaintext
/// is represented, so this value can safely cross into online graph setup.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseLwrPrfEncryptionBinding {
    pub setup: SparseLwrPrfSetupIdentity,
    pub d0_identity: [u8; 32],
    pub d0_modulus: usize,
    pub d0_family_count: usize,
    pub selector_input_name: String,
    pub input_key_identity: [u8; 32],
}

impl SparseLwrPrfEncryptionBinding {
    /// Stable identity of the encrypt-stage public/runtime binding.
    pub fn identity(&self) -> Result<[u8; 32], PrfError> {
        let bytes =
            serde_json::to_vec(self).map_err(|error| PrfError::Artifact(error.to_string()))?;
        let mut digest = sha2::Sha256::new();
        digest.update(b"mxx-exponent-lut/sparse-lwr/encrypt-binding");
        digest.update(bytes);
        Ok(digest.finalize().into())
    }
}

impl std::fmt::Debug for SparseLwrPrfProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparseLwrPrfProgram")
            .field("id", &self.id())
            .field("profile", &self.profile)
            .field("bucket_width", &self.bucket_width)
            .field("block_count", &self.block_count)
            .field("readout", &self.readout)
            .field("selection_families", &self.selection_families.len())
            .finish()
    }
}

impl SparseLwrPrfProgram {
    /// Canonical active-member ranges shared by ordinary bindings and cached
    /// batched preparation. PBC scheduling does not depend on the backend.
    pub fn active_block_ranges(
        &self,
        layout: &PbcPublicLayout,
    ) -> Result<(usize, Vec<(usize, usize)>), PrfError> {
        validate_profile_program(self.profile, self)?;
        if self.block_count != layout.parameters.bucket_count ||
            self.bucket_width != layout.bucket_width
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let active =
            PbcActiveCellIndex::build(layout).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        let ranges = (0..self.block_count)
            .map(|block| {
                active
                    .bucket_iter(block)
                    .and_then(|mut entries| {
                        entries.next().map(|(_, _, offset)| {
                            (offset, active.bucket_active_width(block).unwrap_or_default())
                        })
                    })
                    .ok_or(PrfError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((active.len(), ranges))
    }

    /// Builds the staged PRF declaration. Each bucket is represented by one
    /// structural `select` body (`MulPublic(label)` followed by
    /// `MulPrivate(selector)` and the family `Sum`); no selector/label member
    /// is expanded on the host.
    /// This constructor contains no selector plaintexts or D0 values itself.
    pub fn new_structural(
        profile: SparseLwrPrfProfile,
        bucket_width: usize,
        block_count: usize,
        hash_key: &[u8],
        output_labels: &[PrfLabel],
    ) -> Result<Self, PrfError> {
        let branch_count =
            block_count.checked_mul(bucket_width).ok_or(PrfError::InvalidSparseLwrBlock)?;
        let columns = profile.transform_column_count()?;
        if output_labels.is_empty() || branch_count == 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let mut transforms = Vec::with_capacity(
            branch_count.checked_mul(output_labels.len()).ok_or(PrfError::InvalidSparseLwrBlock)?,
        );
        for label in output_labels {
            let transform = profile.derive_label_transform(hash_key, label, columns)?;
            transforms.extend(std::iter::repeat_n(transform, branch_count));
        }
        Self::new_structural_with_labels(
            profile,
            bucket_width,
            block_count,
            output_labels.to_vec(),
            transforms,
        )
    }

    /// Builds a structural graph for an ordered set of scalar PRF outputs.
    ///
    /// One public D family, one private selector family, and one initial
    /// accumulator are shared by every label.  Each label gets its own
    /// sequential bucket chain and terminal LUT root, while the transform
    /// indices are embedded directly in the family-body AST.  `transforms`
    /// is laid out label-major (`label * block_count * bucket_width`).
    pub fn new_structural_with_labels(
        profile: SparseLwrPrfProfile,
        bucket_width: usize,
        block_count: usize,
        output_labels: Vec<PrfLabel>,
        transforms: Vec<BinaryTransformMatrix>,
    ) -> Result<Self, PrfError> {
        // Derive the exact width of the encryptor-supplied public vector D0.
        // A transform selects from these columns, so accepting any other
        // transform width would make its indices invalid at evaluation time.
        profile.validate()?;
        let d_columns = profile.transform_column_count()?;

        // Branches are numbered globally by flattening (bucket, local slot).
        // This same index addresses both a branch transform and its private
        // selector member.
        let branch_count =
            block_count.checked_mul(bucket_width).ok_or(PrfError::InvalidSparseLwrBlock)?;

        // Output labels are Program identities, not merely display metadata:
        // every label must be present exactly once so that output order and
        // transform ownership remain unambiguous.
        if output_labels.is_empty() ||
            output_labels.iter().enumerate().any(|(index, label)| {
                output_labels[..index].iter().any(|previous| previous == label)
            })
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }

        // `transforms` is label-major. Each label owns one binary transform
        // for every flattened branch, and every transform must select from
        // exactly the D0 column domain derived above. Both row-vector and
        // column-vector representations are accepted and normalized later by
        // `transform_columns`.
        let transform_count =
            branch_count.checked_mul(output_labels.len()).ok_or(PrfError::InvalidSparseLwrBlock)?;
        if bucket_width == 0 ||
            block_count == 0 ||
            transforms.len() != transform_count ||
            transforms.iter().any(|transform| {
                !((transform.rows() == 1 && transform.columns() == d_columns) ||
                    (transform.rows() == d_columns && transform.columns() == 1))
            })
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }

        // Declare the shared Program inputs once. `initial_input` starts each
        // label's accumulator chain; `d0` contains the public exponent factors;
        // `selector` contains one private one-hot member per flattened branch;
        // and `identity` represents the public empty product.
        let readout = profile.canonical_readout()?;
        let mut builder = ExponentLutProgramBuilder::new();
        let initial_input = builder.input(AttributeKnowledge::PrivatePayload)?;
        let d0 = builder
            .input_family(IntExpr::constant(d_columns), AttributeKnowledge::PublicPayload)?;
        let selector = builder.rhs_family(IntExpr::constant(branch_count))?;
        let identity = builder.input(AttributeKnowledge::PublicPayload)?;

        // The bucket chain stays in exponent representation. This terminal LUT
        // performs the sole conversion to the coefficient-valued PRF output.
        let values = readout.clone();
        let lut = builder.register_lut(LutTable::unary(
            profile.native_width(),
            profile.layout,
            profile.native_width(),
            values,
            LutOutputRepresentation::Coefficient,
        )?)?;
        let mut selection_families = Vec::with_capacity(block_count);
        let mut outputs = Vec::with_capacity(output_labels.len());
        for label_index in 0..output_labels.len() {
            // Locate this label's contiguous branch-transform block, then
            // begin an independent bucket chain from the shared initial input.
            let transform_offset =
                label_index.checked_mul(branch_count).ok_or(PrfError::InvalidSparseLwrBlock)?;
            let mut state = initial_input;
            for bucket in 0..block_count {
                // Convert the bucket-local slot range into global branch IDs.
                let bucket_start =
                    bucket.checked_mul(bucket_width).ok_or(PrfError::InvalidSparseLwrBlock)?;

                // A parallel family has one static body shape. Group branches
                // by the number of selected D0 columns so that every member of
                // a group executes the same number of MulPublic gates.
                let mut grouped =
                    std::collections::BTreeMap::<usize, Vec<(usize, Vec<usize>)>>::new();
                for local in 0..bucket_width {
                    let branch = bucket_start + local;
                    // The 1-bits of T_(label, branch) select the D0 factors
                    // whose exponent product represents D0 * T_(label, branch).
                    let selected = transform_columns(&transforms[transform_offset + branch])
                        .ok_or(PrfError::InvalidSparseLwrBlock)?;
                    grouped.entry(selected.len()).or_default().push((branch, selected));
                }

                // Each group is reduced independently below; `bucket_sum`
                // combines those partial sums into the complete branch sum.
                let mut bucket_sum = None;
                for (leaf_count, branches) in grouped {
                    let range = FamilyRange::full(IntExpr::constant(branches.len()))?;
                    let family = builder.parallel(range, |builder, branch_index| {
                        // `branch_index` is local to this equal-width group.
                        // Translate it back to the flattened branch index used
                        // by the shared selector family.
                        let branch_indices = branches
                            .iter()
                            .map(|(branch, _)| IntExpr::constant(*branch))
                            .collect::<Vec<_>>();
                        let selector_index = IntExpr::Select {
                            selector: Box::new(branch_index.clone()),
                            branches: branch_indices,
                        };
                        let (_, first_selected) =
                            branches.first().ok_or(ProgramValidationError::InvalidFamilyRange)?;
                        let mut leaves = Vec::with_capacity(leaf_count);
                        if first_selected.is_empty() {
                            // An all-zero transform denotes an empty product,
                            // whose public exponent encoding is the identity.
                            leaves.push(Operand::Scalar(identity));
                        } else {
                            for column in 0..leaf_count {
                                // At each product position, select the D0 index
                                // belonging to the current group-local branch.
                                // This preserves one common family body while
                                // allowing each branch to use different columns.
                                let indices = branches
                                    .iter()
                                    .map(|(_, selected)| IntExpr::constant(selected[column]))
                                    .collect::<Vec<_>>();
                                leaves.push(Operand::Member {
                                    family: d0,
                                    index: IntExpr::Select {
                                        selector: Box::new(branch_index.clone()),
                                        branches: indices,
                                    },
                                });
                            }
                        }

                        // Multiply the previous bucket state by the public D0
                        // factors selected by T_(label, branch). In exponent
                        // representation their product adds the selected D0
                        // exponents, yielding state + D0*T_(label, branch).
                        let multiplied = apply_public_product(builder, state, leaves)?;

                        // Weight the candidate by its private selector. With a
                        // one-hot selector, exactly one branch survives in each
                        // bucket without revealing which branch was selected.
                        let weighted = builder.mul_private(
                            Operand::Scalar(multiplied),
                            Operand::Member { family: selector, index: selector_index },
                        )?;
                        Ok(Operand::Scalar(weighted))
                    })?;

                    // Sum all branches with this product width, then merge the
                    // result with the other width groups in the same bucket.
                    let reduced = builder.sum(family)?;
                    bucket_sum = Some(match bucket_sum {
                        None => reduced,
                        Some(previous) => {
                            builder.add(Operand::Scalar(previous), Operand::Scalar(reduced))?
                        }
                    });
                }

                // Feed the selected branch sum into the next bucket, making
                // buckets sequential even though branches within one bucket
                // are represented by parallel families.
                state = bucket_sum.ok_or(PrfError::InvalidSparseLwrBlock)?;
                if label_index == 0 {
                    // All labels reuse the same D0 and selector declarations.
                    // Record bucket-facing metadata only once rather than
                    // duplicating identical declarations for every label.
                    selection_families.push(PrfSelectionFamily {
                        selector,
                        d0,
                        capacity: bucket_width,
                    });
                }
            }

            // Read the final exponent accumulator into the scalar value
            // associated with this output label.
            outputs.push(builder.lut(Operand::Scalar(state), lut)?);
        }
        let output = *outputs.first().ok_or(PrfError::InvalidSparseLwrBlock)?;

        // Freeze all ordered label roots into one immutable Program. The first
        // root is also retained for the scalar `output` accessor, while
        // `outputs` preserves the complete label order.
        let accumulation =
            builder.finish(outputs.iter().copied().map(Operand::Scalar).collect())?;
        Ok(Self {
            accumulation,
            readout,
            initial_input,
            output,
            profile,
            bucket_width,
            block_count,
            selection_families,
            d0,
            output_labels,
            outputs,
            label_transforms: transforms,
        })
    }

    /// Returns the structural selector/value family declarations in bucket
    /// order. No runtime family members are materialized by this accessor.
    pub fn selection_families(&self) -> &[PrfSelectionFamily] {
        &self.selection_families
    }

    /// Returns the encryptor-supplied D0 family input used by every bucket.
    pub fn d0_input(&self) -> WireId {
        self.d0
    }

    /// Ordered scalar roots and their opaque application labels. The order is
    /// the immutable Program root order; callers must not infer meaning from
    /// map iteration order.
    pub fn ordered_outputs(&self) -> impl Iterator<Item = (&PrfLabel, WireId)> {
        self.output_labels.iter().zip(self.outputs.iter().copied())
    }

    /// Returns the canonical label-major transforms embedded in the program.
    /// The slice is ordered as `label * (block_count * bucket_width) + branch`.
    /// It is immutable program input material and does not expose graph
    /// internals or runtime selector values.
    pub fn label_transforms(&self) -> &[BinaryTransformMatrix] {
        &self.label_transforms
    }

    /// Returns the balanced reduction shape required by one transform row.
    /// Empty rows use the evaluator's identity encoding, singleton rows reuse
    /// their input, and odd levels carry the final singleton unchanged.
    pub fn transform_product_schedule(
        &self,
        transform: &BinaryTransformMatrix,
    ) -> Result<BalancedProductSchedule, PrfError> {
        let selected = transform_columns(transform).ok_or(PrfError::InvalidSparseLwrBlock)?;
        BalancedProductSchedule::new(selected.len()).map_err(|_| PrfError::InvalidSparseLwrBlock)
    }

    /// Derives the setup identity for the structural PRF declaration.
    ///
    /// `D0` is intentionally absent; it is an encrypt-stage input and must
    /// not force a new preprocessing graph or selector namespace.
    pub fn setup_identity(
        &self,
        layout: &PbcPublicLayout,
    ) -> Result<SparseLwrPrfSetupIdentity, PrfError> {
        self.setup_identity_for_key(layout, [0; 32])
    }

    /// Derives setup identity while binding the public selector key namespace.
    pub fn setup_identity_for_key(
        &self,
        layout: &PbcPublicLayout,
        selector_key_instance: [u8; 32],
    ) -> Result<SparseLwrPrfSetupIdentity, PrfError> {
        if !self.is_structural() || self.block_count != layout.parameters.bucket_count {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let schedule = self.product_schedule()?;
        let label_schedule = self.label_schedule()?;
        let t_l_commitment = *self.id().as_bytes();
        Ok(SparseLwrPrfSetupIdentity {
            program: self.id(),
            layout: layout.layout_id,
            t_l_commitment,
            input_key_identity: selector_key_instance,
            label_schedule,
            schedule,
        })
    }

    /// Binds encryptor-supplied public D0 encodings and selector runtime data
    /// to this setup.  The returned binding contains no secret input and can
    /// be used to authenticate the online stage's structural inputs.
    pub fn encryption_binding(
        &self,
        layout: &PbcPublicLayout,
        d0: &PbcPublicVectorFamilyBinding,
        selector: &PbcSelectorRuntimeBinding,
    ) -> Result<SparseLwrPrfEncryptionBinding, PrfError> {
        let setup = self.setup_identity_for_key(layout, selector.key_instance_id())?;
        let expected_count = self.profile.transform_column_count()?;
        d0.validate_d0(layout, expected_count).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        if selector.layout_id() != layout.layout_id {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(SparseLwrPrfEncryptionBinding {
            setup,
            d0_identity: d0.identity(),
            d0_modulus: d0.modulus,
            d0_family_count: d0.family_count,
            selector_input_name: selector.input_name().to_owned(),
            input_key_identity: selector.key_instance_id(),
        })
    }

    /// Returns whether this declaration uses structural selector/value
    /// families rather than pre-combined RHS gates.
    pub const fn is_structural(&self) -> bool {
        !self.selection_families.is_empty()
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

    pub const fn bucket_width(&self) -> usize {
        self.bucket_width
    }

    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Returns the canonical balanced reduction shape for the block product.
    ///
    /// The schedule is metadata only: it can be captured by a structural
    /// family lowerer without allocating one host graph node per bucket
    /// member. Empty and singleton products are represented explicitly so
    /// evaluators can reuse an identity or the sole input, respectively.
    pub fn product_schedule(&self) -> Result<BalancedProductSchedule, PrfError> {
        BalancedProductSchedule::new(self.block_count).map_err(|_| PrfError::InvalidSparseLwrBlock)
    }

    /// Returns the balanced schedule for the labels in one bucket.  This is
    /// the schedule used for the public `D0*T_l` tree; it is separate from
    /// the outer bucket aggregation schedule returned by `product_schedule`.
    pub fn label_schedule(&self) -> Result<BalancedProductSchedule, PrfError> {
        BalancedProductSchedule::new(self.bucket_width).map_err(|_| PrfError::InvalidSparseLwrBlock)
    }

    /// Returns the setup commitment for the public label transforms without
    /// materializing a matrix or expanding a family on the host.
    pub fn label_transform_commitment(&self) -> [u8; 32] {
        *self.id().as_bytes()
    }

    pub fn id(&self) -> ExponentLutProgramId {
        self.accumulation.id()
    }

    /// Returns the complete immutable declaration, including the terminal
    /// coefficient LUT.  Evaluators should lower this value directly instead
    /// of constructing a separate readout graph.
    pub fn program(&self) -> &ExponentLutProgram {
        &self.accumulation
    }

    /// The native accumulator has no reduction or terminal LUT stage.
    pub fn has_only_native_accumulation(&self) -> bool {
        false
    }

    /// Returns whether the program has exactly one terminal coefficient LUT.
    pub fn has_terminal_coefficient_lut(&self) -> bool {
        true
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

/// Structural PRF preparation descriptor.  Cryptographic helper caches are
/// owned by the concrete evaluator; this type records only the Program,
/// and setup identity needed to reopen them.
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
        digest.update(b"mxx-exponent-lut/sparse-lwr/structural-preparation");
        digest.update(bytes);
        Ok(digest.finalize().into())
    }
}

/// Structural PRF preparation state shared by preprocessing and online
/// evaluation.  It carries no D0 values and no secret material.
#[derive(Clone, Debug)]
pub struct PreparedSparseLwrPrf {
    program: SparseLwrPrfProgram,
    setup: SparseLwrPrfSetupIdentity,
}

impl PartialEq for PreparedSparseLwrPrf {
    fn eq(&self, other: &Self) -> bool {
        // The structural declaration is compared through its canonical
        // descriptor; the embedded graph handles are not value-comparable.
        self.descriptor() == other.descriptor()
    }
}

impl Eq for PreparedSparseLwrPrf {}

impl PreparedSparseLwrPrf {
    pub fn new(
        program: SparseLwrPrfProgram,
        layout: &PbcPublicLayout,
        selector_key_instance: [u8; 32],
    ) -> Result<Self, PrfError> {
        validate_profile_program(program.profile(), &program)?;
        let setup = program.setup_identity_for_key(layout, selector_key_instance)?;
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
            return Err(PrfError::Artifact("structural PRF identity mismatch".into()));
        }
        if descriptor.setup.program != program.id() || !program.is_structural() {
            return Err(PrfError::Artifact("structural PRF setup mismatch".into()));
        }
        if descriptor.setup.schedule != program.product_schedule()? {
            return Err(PrfError::Artifact("structural PRF schedule mismatch".into()));
        }
        let expected_label_schedule = BalancedProductSchedule::new(program.bucket_width())
            .map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        let expected_t_l_commitment = program.label_transform_commitment();
        if descriptor.setup.label_schedule != expected_label_schedule ||
            descriptor.setup.t_l_commitment != expected_t_l_commitment
        {
            return Err(PrfError::Artifact("structural PRF label schedule mismatch".into()));
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
    use num_bigint::BigInt;

    #[test]
    fn profile_derives_q_and_p_for_nonbinary_digit_bases() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        assert_eq!(profile.native_width(), 32);
        assert_eq!(profile.q().unwrap(), 64);
        assert_eq!(profile.p().unwrap(), 16);
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 6 }, 16).unwrap();
        assert_eq!(profile.q().unwrap(), 128);
        assert_eq!(profile.p().unwrap(), 32);
    }

    #[test]
    fn profile_rejects_nondividing_digit_base() {
        assert!(SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 12).is_err());
    }

    #[test]
    fn profile_preserves_zero_and_nonzero_stride_input_layouts() {
        let zero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 0, width_log2: 4 }, 2).unwrap();
        let nonzero_stride =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 2).unwrap();

        assert_eq!(zero_stride.ring_dimension(), nonzero_stride.ring_dimension());
        assert_ne!(
            zero_stride.q(),
            nonzero_stride.q(),
            "the exponent modulus is selected by input width"
        );
        assert_ne!(
            zero_stride.transform_column_count(),
            nonzero_stride.transform_column_count(),
            "the input layout, not only the native ring, selects the transform width"
        );
        assert_eq!(zero_stride.canonical_readout().unwrap().len(), 16);
        assert_eq!(nonzero_stride.canonical_readout().unwrap().len(), 8);
    }

    #[test]
    fn canonical_labels_and_transforms_are_stable_and_single_column() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let first_slot = 7_u64.to_le_bytes();
        let second_slot = 8_u64.to_le_bytes();
        let first =
            PrfLabel::from_components(&[b"mask".as_slice(), first_slot.as_slice()]).unwrap();
        let second =
            PrfLabel::from_components(&[b"mask".as_slice(), second_slot.as_slice()]).unwrap();
        assert_ne!(first, second);
        let columns = profile.transform_column_count().unwrap();
        let a = profile.derive_label_transform(b"hash-key", &first, columns).unwrap();
        let b = profile.derive_label_transform(b"hash-key", &first, columns).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, profile.derive_label_transform(b"hash-key", &second, columns).unwrap());
        assert_eq!((a.rows(), a.columns()), (columns, 1));
        assert!(profile.derive_label_transform(b"hash-key", &first, 0).is_err());
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
            let profile =
                SparseLwrPrfProfile::new(ExpLayout { stride_log2, width_log2 }, p_prime).unwrap();
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

    #[test]
    fn public_product_uses_balanced_depth_and_keeps_public_factors_on_the_left() {
        for leaf_count in 1..=17 {
            let schedule = BalancedProductSchedule::new(leaf_count).unwrap();
            let mut multiplication_count = 0;
            let depth = reduce_balanced(vec![0_usize; leaf_count], |lhs, rhs| {
                multiplication_count += 1;
                Ok::<_, std::convert::Infallible>(lhs.max(rhs) + 1)
            })
            .unwrap()
            .unwrap();
            assert_eq!(depth, schedule.level_count() - 1);
            assert_eq!(multiplication_count, leaf_count - 1);
        }

        let mut builder = ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = (0..8)
            .map(|_| builder.input(AttributeKnowledge::PublicPayload).unwrap())
            .map(Operand::Scalar)
            .collect();
        let output = apply_public_product(&mut builder, private, public).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        assert_eq!(program.wire_knowledge(output).unwrap(), AttributeKnowledge::PrivatePayload);
    }

    #[test]
    fn structural_program_uses_one_shared_d_family_and_sequential_buckets() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new_structural(
            profile,
            4,
            3,
            b"test-hash-key",
            &[PrfLabel::from_components(&[b"test-label"]).unwrap()],
        )
        .unwrap();
        assert!(program.is_structural());
        assert_eq!(program.selection_families().len(), 3);
        assert_eq!(program.d0_input(), program.selection_families()[0].d0);
        assert!(program.selection_families().windows(2).all(|families| {
            families[0].selector == families[1].selector && families[0].d0 == families[1].d0
        }));
        assert!(program.has_terminal_coefficient_lut());
        assert!(!program.has_only_native_accumulation());
        assert_eq!(program.product_schedule().unwrap().leaf_count(), 3);
        let other = SparseLwrPrfProgram::new_structural(
            profile,
            4,
            3,
            b"test-hash-key",
            &[PrfLabel::from_components(&[b"other-label"]).unwrap()],
        )
        .unwrap();
        assert_ne!(program.id(), other.id());
        let other_hash = SparseLwrPrfProgram::new_structural(
            profile,
            4,
            3,
            b"other-hash-key",
            &[PrfLabel::from_components(&[b"test-label"]).unwrap()],
        )
        .unwrap();
        assert_ne!(program.id(), other_hash.id());
    }

    #[test]
    fn fixed_transform_shapes_are_checked_before_ast_construction() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 3 }, 2).unwrap();
        let columns = profile.transform_column_count().unwrap();
        let transform = BinaryTransformMatrix::new(1, columns, vec![1; columns]).unwrap();
        let label = PrfLabel::from_components(&[b"test-label"]).unwrap();
        assert!(
            SparseLwrPrfProgram::new_structural_with_labels(
                profile,
                2,
                2,
                vec![label.clone()],
                vec![transform.clone()]
            )
            .is_err()
        );
        let transforms = vec![transform; 4];
        let program =
            SparseLwrPrfProgram::new_structural_with_labels(profile, 2, 2, vec![label], transforms)
                .unwrap();
        assert_eq!(
            program
                .transform_product_schedule(
                    &BinaryTransformMatrix::new(1, columns, vec![1; columns]).unwrap(),
                )
                .unwrap()
                .leaf_count(),
            columns
        );
    }

    #[test]
    fn setup_identity_is_shared_across_encryptor_d0_values() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let parameters = crate::noise_refresh::pbc::PbcParameters::custom(3, 2, 2, 2, 32, None);
        let generated = (0..32_u8)
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    &[0, 1],
                )
                .ok()
            })
            .expect("toy setup must find a PBC schedule");
        let layout = &generated.public_layout;
        let program = SparseLwrPrfProgram::new_structural(
            profile,
            layout.bucket_width,
            layout.parameters.bucket_count,
            b"toy-hash-key",
            &[PrfLabel::from_components(&[b"toy-label"]).unwrap()],
        )
        .unwrap();
        let ring =
            crate::test_support::explicit_ring(&[257], profile.ring_dimension().unwrap() as u32);
        let selector = crate::noise_refresh::pbc::PbcTrustedSelectorBits::from_schedule(
            &generated, &ring, [9; 32],
        )
        .unwrap()
        .runtime_binding(&ring)
        .unwrap();
        let columns = profile.transform_column_count().unwrap();
        let modulus = profile.q().unwrap();
        let a = crate::noise_refresh::pbc::PbcPublicVectorFamilyBinding::from_d0_values(
            layout,
            (0..columns).map(|index| (index % modulus) as u64).collect(),
            modulus,
            columns,
        )
        .unwrap();
        let b = crate::noise_refresh::pbc::PbcPublicVectorFamilyBinding::from_d0_values(
            layout,
            (0..columns).map(|index| ((index + 1) % modulus) as u64).collect(),
            modulus,
            columns,
        )
        .unwrap();
        assert!(
            crate::noise_refresh::pbc::PbcPublicVectorFamilyBinding::from_d0_values(
                layout,
                vec![0; columns.saturating_sub(1)],
                modulus,
                columns,
            )
            .is_err()
        );
        let mut forged = serde_json::to_value(&a).unwrap();
        forged["family_count"] = serde_json::json!(columns.saturating_sub(1));
        let forged: crate::noise_refresh::pbc::PbcPublicVectorFamilyBinding =
            serde_json::from_value(forged).unwrap();
        assert!(forged.validate_d0(layout, columns).is_err());
        let a_binding = program.encryption_binding(layout, &a, &selector).unwrap();
        let b_binding = program.encryption_binding(layout, &b, &selector).unwrap();
        assert_eq!(a_binding.setup, b_binding.setup);
        assert_eq!(a_binding.setup.label_schedule.leaf_count(), program.bucket_width());
        assert_eq!(a_binding.setup.t_l_commitment, program.label_transform_commitment());
        assert_eq!(a_binding.setup.input_key_identity, [9; 32]);
        assert_eq!(a_binding.selector_input_name, b_binding.selector_input_name);
        assert_ne!(a_binding.d0_identity, b_binding.d0_identity);
        assert_ne!(a_binding.identity().unwrap(), b_binding.identity().unwrap());
    }

    #[test]
    fn prepared_descriptor_requires_an_external_trusted_identity() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let parameters = crate::noise_refresh::pbc::PbcParameters::custom(3, 2, 2, 2, 32, None);
        let generated = (0..32_u8)
            .find_map(|seed| {
                crate::noise_refresh::pbc::generate_key_layout(
                    &parameters,
                    crate::noise_refresh::pbc::PbcRootSeed([seed; 32]),
                    &[0, 1],
                )
                .ok()
            })
            .expect("toy setup must find a PBC schedule");
        let layout = generated.public_layout;
        let program = SparseLwrPrfProgram::new_structural(
            profile,
            layout.bucket_width,
            layout.parameters.bucket_count,
            b"toy-hash-key",
            &[PrfLabel::from_components(&[b"toy-label"]).unwrap()],
        )
        .unwrap();
        let prepared = PreparedSparseLwrPrf::new(program.clone(), &layout, [7; 32]).unwrap();
        let descriptor = prepared.descriptor();
        assert!(PreparedSparseLwrPrf::import(program.clone(), &descriptor, [0; 32]).is_err());
        assert!(
            PreparedSparseLwrPrf::import(program, &descriptor, descriptor.identity().unwrap(),)
                .is_ok()
        );
    }

    #[test]
    fn profile_program_mismatch_is_rejected() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new_structural(
            profile,
            2,
            1,
            b"test-hash-key",
            &[PrfLabel::from_components(&[b"test-label"]).unwrap()],
        )
        .unwrap();
        let other =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4).unwrap();
        assert!(validate_profile_program(other, &program).is_err());

        let mut noncanonical = program.clone();
        noncanonical.readout[0] += 1;
        assert!(validate_profile_program(profile, &noncanonical).is_err());
    }

    #[test]
    fn tiny_native_reference_oracle_covers_every_z_mod_q() {
        let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
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
