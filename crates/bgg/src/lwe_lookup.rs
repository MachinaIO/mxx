//! LWE-based public lookup-table evaluation expressed with the declarative DSL.

use crate::{
    BggEncodingWire, BggPolyEncodingWire, BggPublicKeyWire, CircuitCompileError,
    NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
};
use mxx_dsl::{Bytes, DslContext, DslError, Family, HashTag, Mat, Ring, Trapdoor, parallel_zip};
use mxx_gadgets::{
    Poly, PolyElem,
    circuit::{CircuitLoweringTypes, GateInstance, PolyCircuit, PublicLookupLowering},
};
use mxx_ir_core::{
    IntExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    types::MatrixType,
};
use num_bigint::{BigInt, Sign};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// Stable structural identity of one public-lookup invocation.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LweLookupIdentity {
    pub call_path: Vec<usize>,
    pub gate: usize,
    pub occurrence: usize,
    pub lookup: usize,
    pub slot: Option<usize>,
}

impl LweLookupIdentity {
    fn gate_token(&self) -> String {
        if self.call_path.is_empty() && self.occurrence == 0 {
            return self.gate.to_string();
        }
        let mut token = self.call_path.iter().map(usize::to_string).collect::<Vec<_>>().join("_");
        if !token.is_empty() {
            token.push('_');
        }
        token.push_str(&format!("g{}_o{}", self.gate, self.occurrence));
        token
    }

    fn slot_index(&self) -> usize {
        self.slot.unwrap_or(0)
    }

    fn output_public_key_tag(&self) -> Vec<u8> {
        format!("A_LT_{}_slot{}", self.gate_token(), self.slot_index()).into_bytes()
    }

    fn low_matrix_tag(&self, row: usize) -> Vec<u8> {
        format!("LWE_R_G_{}_{}_{}_slot{}", self.gate_token(), self.lookup, row, self.slot_index())
            .into_bytes()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LweLookupEntry {
    row: usize,
    output: BigInt,
}

/// A dense public LUT materialized in input order.
///
/// Rows must be a permutation of `0..len`; row identity determines the
/// historical `K_low` hash tag while artifact families remain in input order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupTable {
    entries: Vec<LweLookupEntry>,
}

impl LweLookupTable {
    pub fn new(
        rows_and_outputs_by_input: impl IntoIterator<Item = (usize, BigInt)>,
    ) -> Result<Self, LweLookupCompileError> {
        let entries = rows_and_outputs_by_input
            .into_iter()
            .map(|(row, output)| LweLookupEntry { row, output })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(LweLookupCompileError::EmptyTable);
        }
        let mut seen = vec![false; entries.len()];
        for entry in &entries {
            if entry.row >= entries.len() {
                return Err(LweLookupCompileError::RowOutOfRange {
                    row: entry.row,
                    length: entries.len(),
                });
            }
            if std::mem::replace(&mut seen[entry.row], true) {
                return Err(LweLookupCompileError::DuplicateRow(entry.row));
            }
        }
        if let Some(row) = seen.iter().position(|present| !present) {
            return Err(LweLookupCompileError::MissingRow(row));
        }
        Ok(Self { entries })
    }

    pub fn from_public_lut<P: Poly>(
        parameters: &P::Params,
        table: &mxx_gadgets::circuit::PublicLut<P>,
    ) -> Result<Self, LweLookupCompileError> {
        let mut entries = vec![None; table.len()];
        for (input, (row, output)) in table.entries(parameters) {
            let input = usize::try_from(input)
                .map_err(|_| LweLookupCompileError::InputOutOfRange(input))?;
            let row =
                usize::try_from(row).map_err(|_| LweLookupCompileError::RowIndexTooLarge(row))?;
            if input >= entries.len() {
                return Err(LweLookupCompileError::InputOutOfRange(input as u64));
            }
            entries[input] = Some((row, BigInt::from_biguint(Sign::Plus, output.value().clone())));
        }
        Self::new(
            entries
                .into_iter()
                .enumerate()
                .map(|(input, entry)| entry.ok_or(LweLookupCompileError::MissingInput(input)))
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn commitment(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-bgg/lwe-lookup-table/v1");
        update_commitment_usize(&mut hasher, self.entries.len());
        for (input, entry) in self.entries.iter().enumerate() {
            update_commitment_usize(&mut hasher, input);
            update_commitment_usize(&mut hasher, entry.row);
            let (sign, magnitude) = entry.output.to_bytes_le();
            hasher.update([match sign {
                Sign::NoSign => 0,
                Sign::Plus => 1,
                Sign::Minus => 2,
            }]);
            update_commitment_bytes(&mut hasher, &magnitude);
        }
        hasher.finalize().into()
    }
}

fn update_commitment_usize(hasher: &mut Sha256, value: usize) {
    update_commitment_bytes(hasher, value.to_string().as_bytes());
}

fn update_commitment_bytes(hasher: &mut Sha256, value: &[u8]) {
    let length = u64::try_from(value.len()).expect("lookup commitment component exceeds u64");
    hasher.update(length.to_le_bytes());
    hasher.update(value);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupArtifactNames {
    pub output_public_key: String,
    pub low_matrices: String,
    pub high_matrices: String,
}

impl LweLookupArtifactNames {
    pub fn for_compiler(compiler: &LweLookupCompiler) -> Self {
        let identity = &compiler.identity;
        let prefix = format!(
            "lwe_lookup_{}_{}_slot{}",
            identity.gate_token(),
            identity.lookup,
            identity.slot_index()
        );
        Self {
            output_public_key: format!("{prefix}_output_public_key"),
            low_matrices: format!("{prefix}_low_matrices"),
            high_matrices: format!("{prefix}_high_matrices"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupArtifacts {
    production_id: ProductionId,
    table_length: usize,
    table_commitment: [u8; 32],
}

impl LweLookupArtifacts {
    pub fn for_compiler(production_id: ProductionId, compiler: &LweLookupCompiler) -> Self {
        Self {
            production_id,
            table_length: compiler.table.len(),
            table_commitment: compiler.table.commitment(),
        }
    }

    pub fn production_id(&self) -> &ProductionId {
        &self.production_id
    }

    pub fn table_length(&self) -> usize {
        self.table_length
    }
}

#[derive(Clone)]
pub struct LweLookupPreprocessingWires {
    pub output_public_key: Mat,
    pub low_matrices: Family<Mat>,
    pub high_matrices: Family<Mat>,
}

#[derive(Clone)]
pub struct LweLookupArtifactWires {
    pub output_public_key: Mat,
    pub low_matrices: Family<Mat>,
    pub high_matrices: Family<Mat>,
}

#[derive(Clone, Debug)]
pub struct LweLookupInvocation {
    compiler: LweLookupCompiler,
    artifacts: LweLookupArtifacts,
    circuit_table_identity: usize,
}

#[derive(Clone, Debug)]
pub struct NaiveLweLookupInvocation {
    identity: LweLookupIdentity,
    slots: Vec<LweLookupInvocation>,
    circuit_table_identity: usize,
}

#[derive(Clone, Debug)]
pub struct LweLookupCompiler {
    pub identity: LweLookupIdentity,
    pub table: LweLookupTable,
    pub public_key_type: MatrixType,
    pub low_matrix_type: MatrixType,
    pub high_matrix_type: MatrixType,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
}

#[derive(Debug, Error)]
pub enum LweLookupCompileError {
    #[error("an LWE lookup table must contain at least one entry")]
    EmptyTable,
    #[error("LWE lookup input {0} does not fit in the host index type")]
    InputOutOfRange(u64),
    #[error("LWE lookup row {0} does not fit in the host index type")]
    RowIndexTooLarge(u64),
    #[error("LWE lookup table is missing input {0}")]
    MissingInput(usize),
    #[error("LWE lookup row {row} is out of range for table length {length}")]
    RowOutOfRange { row: usize, length: usize },
    #[error("LWE lookup row {0} occurs more than once")]
    DuplicateRow(usize),
    #[error("LWE lookup row {0} is missing")]
    MissingRow(usize),
    #[error("LWE lookup matrix types do not satisfy the public LWE equation")]
    MatrixTypeMismatch,
    #[error("LWE BGG+ encoding lookup requires a revealed input plaintext")]
    MissingPlaintext,
    #[error("LWE BGG+ lookup families must have matching slot counts")]
    SlotCountMismatch,
    #[error("LWE lookup artifact families have an incompatible count or matrix type")]
    ArtifactFamilyLayout,
    #[error("lookup artifact table length {actual} does not match compiler length {expected}")]
    ArtifactLength { expected: usize, actual: usize },
    #[error("the materialized circuit lookup table differs from the preprocessing table")]
    CircuitTableMismatch,
    #[error("the lookup lowering was bound to a different circuit lookup registration")]
    CircuitTableBinding,
    #[error("lookup artifacts were produced for a different lookup table")]
    ArtifactTableCommitment,
    #[error("more than one LWE lookup invocation has the same structural identity")]
    DuplicateInvocation,
    #[error("the circuit gate has no bound LWE lookup invocation")]
    MissingInvocation,
    #[error("a naive LWE lookup invocation requires at least one slot")]
    EmptySlotInvocations,
    #[error("naive LWE lookup slot {slot} has an inconsistent structural identity")]
    SlotIdentity { slot: usize },
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl LweLookupInvocation {
    pub fn bind<P: Poly>(
        compiler: LweLookupCompiler,
        artifacts: LweLookupArtifacts,
        parameters: &P::Params,
        circuit: &PolyCircuit<P>,
    ) -> Result<Self, LweLookupCompileError> {
        let circuit_table = circuit.lookup_table(compiler.identity.lookup);
        let materialized = LweLookupTable::from_public_lut(parameters, circuit_table.as_ref())?;
        if materialized != compiler.table {
            return Err(LweLookupCompileError::CircuitTableMismatch);
        }
        compiler.validate_artifacts(&artifacts)?;
        Ok(Self {
            compiler,
            artifacts,
            circuit_table_identity: std::sync::Arc::as_ptr(&circuit_table) as usize,
        })
    }
}

impl NaiveLweLookupInvocation {
    pub fn new(
        slots: impl IntoIterator<Item = LweLookupInvocation>,
    ) -> Result<Self, LweLookupCompileError> {
        let slots = slots.into_iter().collect::<Vec<_>>();
        let Some(first) = slots.first() else {
            return Err(LweLookupCompileError::EmptySlotInvocations);
        };
        let mut identity = first.compiler.identity.clone();
        identity.slot = None;
        let circuit_table_identity = first.circuit_table_identity;
        for (slot, invocation) in slots.iter().enumerate() {
            let mut expected = identity.clone();
            expected.slot = Some(slot);
            if invocation.compiler.identity != expected ||
                invocation.circuit_table_identity != circuit_table_identity
            {
                return Err(LweLookupCompileError::SlotIdentity { slot });
            }
        }
        Ok(Self { identity, slots, circuit_table_identity })
    }
}

impl LweLookupCompiler {
    /// Builds `K_high = Preimage_B(A_lt - yG - (A_z - xG)K_low)`.
    pub fn preprocess(
        &self,
        hash_key: Bytes,
        input_public_key: &BggPublicKeyWire,
        trapdoor: &Trapdoor,
    ) -> Result<LweLookupPreprocessingWires, LweLookupCompileError> {
        self.validate_layout()?;
        if !same_matrix_type(input_public_key.matrix.matrix_type(), &self.public_key_type) ||
            !same_matrix_type(
                trapdoor.public_matrix().matrix_type(),
                &MatrixType {
                    columns: self.high_matrix_type.rows.clone(),
                    ..self.public_key_type.clone()
                },
            )
        {
            return Err(LweLookupCompileError::MatrixTypeMismatch);
        }
        let ring = self.ring();
        let output_public_key = ring.hash_matrix(
            hash_key.clone(),
            HashTag::from(self.identity.output_public_key_tag()),
            shape(&self.public_key_type),
        );
        let input_scalars = Family::pack(
            (0..self.table.len())
                .map(|input| ring.polynomial([IntExpr::constant(input)]))
                .collect(),
        )?;
        let output_scalars = Family::pack(
            self.table
                .entries
                .iter()
                .map(|entry| ring.polynomial([IntExpr::constant(entry.output.clone())]))
                .collect(),
        )?;
        let low_inputs = Family::pack(
            self.table
                .entries
                .iter()
                .map(|entry| {
                    ring.hash_decomposed(
                        hash_key.clone(),
                        HashTag::from(self.identity.low_matrix_tag(entry.row)),
                        shape(&self.low_matrix_type),
                        self.gadget_base.clone(),
                        self.digit_count.clone(),
                    )
                })
                .collect(),
        )?;
        let gadget = ring.gadget(
            self.public_key_type.rows.clone(),
            self.gadget_base.clone(),
            self.digit_count.clone(),
        );
        let input_public_key = input_public_key.matrix.clone();
        let output_for_loop = output_public_key.clone();
        let trapdoor = trapdoor.clone();
        let high_shape = shape(&self.high_matrix_type);
        let (low_matrices, high_matrices) = parallel_zip(
            (input_scalars, output_scalars, low_inputs),
            move |_, (input_scalar, output_scalar, low)| {
                let extended_input = input_public_key.clone() - gadget.clone() * input_scalar;
                let target = output_for_loop.clone() - gadget.clone() * output_scalar;
                let adjusted_target = target - extended_input * low.clone();
                let high = trapdoor.sample_preimage(adjusted_target, high_shape.clone()).as_mat();
                (low, high)
            },
        )?;
        Ok(LweLookupPreprocessingWires { output_public_key, low_matrices, high_matrices })
    }

    pub fn export_preprocessing(
        &self,
        context: DslContext,
        wires: LweLookupPreprocessingWires,
        names: &LweLookupArtifactNames,
    ) -> Result<DslContext, LweLookupCompileError> {
        Ok(context
            .public_output(names.output_public_key.clone(), wires.output_public_key)?
            .public_family_output(names.low_matrices.clone(), wires.low_matrices)?
            .public_family_output(names.high_matrices.clone(), wires.high_matrices)?)
    }

    pub fn import_artifacts(
        &self,
        artifacts: &LweLookupArtifacts,
    ) -> Result<LweLookupArtifactWires, LweLookupCompileError> {
        self.validate_layout()?;
        self.validate_artifacts(artifacts)?;
        let ring = self.ring();
        let names = LweLookupArtifactNames::for_compiler(self);
        Ok(LweLookupArtifactWires {
            output_public_key: ring.artifact_input(
                artifacts.production_id.clone(),
                names.output_public_key,
                shape(&self.public_key_type),
                ArtifactConfidentiality::Public,
            ),
            low_matrices: ring.family_artifact_input(
                artifacts.production_id.clone(),
                names.low_matrices,
                self.table.len(),
                shape(&self.low_matrix_type),
                ArtifactConfidentiality::Public,
            ),
            high_matrices: ring.family_artifact_input(
                artifacts.production_id.clone(),
                names.high_matrices,
                self.table.len(),
                shape(&self.high_matrix_type),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn public_key(&self, artifacts: &LweLookupArtifactWires) -> BggPublicKeyWire {
        BggPublicKeyWire { matrix: artifacts.output_public_key.clone(), reveal_plaintext: true }
    }

    /// Builds `c_out = C_B K_high[z] + c_z K_low[z]`.
    pub fn encoding(
        &self,
        input: &BggEncodingWire,
        c_b: &Mat,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<BggEncodingWire, LweLookupCompileError> {
        let plaintext = input.plaintext.clone().ok_or(LweLookupCompileError::MissingPlaintext)?;
        self.validate_artifact_wires(artifacts)?;
        let input_index = plaintext.extract_coefficient(0);
        let low = artifacts.low_matrices.get(input_index.clone());
        let high = artifacts.high_matrices.get(input_index.clone());
        let output_plaintext = input_index.select(
            self.table
                .entries
                .iter()
                .map(|entry| self.ring().polynomial([IntExpr::constant(entry.output.clone())]))
                .collect(),
        )?;
        Ok(BggEncodingWire {
            vector: c_b.clone() * high + input.vector.clone() * low,
            pubkey: self.public_key(artifacts),
            plaintext: Some(output_plaintext),
        })
    }

    pub fn poly_encoding(
        &self,
        input: &BggPolyEncodingWire,
        c_b_by_slot: &Family<Mat>,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<BggPolyEncodingWire, LweLookupCompileError> {
        let plaintexts = input.plaintexts.clone().ok_or(LweLookupCompileError::MissingPlaintext)?;
        if input.vectors.count() != plaintexts.count() ||
            input.vectors.count() != c_b_by_slot.count()
        {
            return Err(LweLookupCompileError::SlotCountMismatch);
        }
        let compiler = self.clone();
        let output_public_key = self.public_key(artifacts);
        let artifacts = artifacts.clone();
        let (vectors, plaintexts) = parallel_zip(
            (input.vectors.clone(), plaintexts, c_b_by_slot.clone()),
            move |_, (vector, plaintext, c_b)| {
                let output = compiler
                    .encoding(
                        &BggEncodingWire {
                            vector,
                            pubkey: compiler.public_key(&artifacts),
                            plaintext: Some(plaintext),
                        },
                        &c_b,
                        &artifacts,
                    )
                    .expect("validated lookup families");
                (output.vector, output.plaintext.expect("lookup reveals its output"))
            },
        )?;
        Ok(BggPolyEncodingWire { vectors, pubkey: output_public_key, plaintexts: Some(plaintexts) })
    }

    fn validate_artifacts(
        &self,
        artifacts: &LweLookupArtifacts,
    ) -> Result<(), LweLookupCompileError> {
        if artifacts.table_length != self.table.len() {
            return Err(LweLookupCompileError::ArtifactLength {
                expected: self.table.len(),
                actual: artifacts.table_length,
            });
        }
        if artifacts.table_commitment != self.table.commitment() {
            return Err(LweLookupCompileError::ArtifactTableCommitment);
        }
        Ok(())
    }

    fn validate_layout(&self) -> Result<(), LweLookupCompileError> {
        let expected_public_columns = IntExpr::Mul(
            Box::new(self.public_key_type.rows.clone()),
            Box::new(self.digit_count.clone()),
        )
        .canonicalize();
        let expected_high_rows = IntExpr::Mul(
            Box::new(self.public_key_type.rows.clone()),
            Box::new(IntExpr::Add(
                Box::new(self.digit_count.clone()),
                Box::new(IntExpr::constant(2)),
            )),
        )
        .canonicalize();
        let types = [&self.low_matrix_type, &self.high_matrix_type];
        if types.iter().any(|ty| {
            ty.modulus.canonicalize() != self.public_key_type.modulus.canonicalize() ||
                ty.ring_dimension.canonicalize() !=
                    self.public_key_type.ring_dimension.canonicalize()
        }) || self.public_key_type.columns.canonicalize() != expected_public_columns ||
            self.low_matrix_type.rows.canonicalize() !=
                self.public_key_type.columns.canonicalize() ||
            self.low_matrix_type.columns.canonicalize() !=
                self.public_key_type.columns.canonicalize() ||
            self.high_matrix_type.columns.canonicalize() !=
                self.public_key_type.columns.canonicalize() ||
            self.high_matrix_type.rows.canonicalize() != expected_high_rows
        {
            return Err(LweLookupCompileError::MatrixTypeMismatch);
        }
        Ok(())
    }

    fn validate_artifact_wires(
        &self,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<(), LweLookupCompileError> {
        let count = IntExpr::constant(self.table.len());
        if !same_matrix_type(artifacts.output_public_key.matrix_type(), &self.public_key_type) ||
            artifacts.low_matrices.count() != &count ||
            !same_matrix_type(artifacts.low_matrices.element_type(), &self.low_matrix_type) ||
            artifacts.high_matrices.count() != &count ||
            !same_matrix_type(artifacts.high_matrices.element_type(), &self.high_matrix_type)
        {
            return Err(LweLookupCompileError::ArtifactFamilyLayout);
        }
        Ok(())
    }

    fn ring(&self) -> Ring {
        Ring::new(self.public_key_type.modulus.clone(), self.public_key_type.ring_dimension.clone())
    }
}

fn shape(matrix_type: &MatrixType) -> (IntExpr, IntExpr) {
    (matrix_type.rows.clone(), matrix_type.columns.clone())
}

fn same_matrix_type(lhs: &MatrixType, rhs: &MatrixType) -> bool {
    lhs.modulus.canonicalize() == rhs.modulus.canonicalize() &&
        lhs.ring_dimension.canonicalize() == rhs.ring_dimension.canonicalize() &&
        lhs.rows.canonicalize() == rhs.rows.canonicalize() &&
        lhs.columns.canonicalize() == rhs.columns.canonicalize()
}

#[derive(Clone)]
pub struct LweLookupPublicKeyLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
}

#[derive(Clone)]
pub struct LweLookupEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    c_b: Mat,
}

#[derive(Clone)]
pub struct LweLookupPolyEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    c_b_by_slot: Family<Mat>,
}

#[derive(Clone)]
pub struct NaiveLweLookupPublicKeyLowering {
    invocations: BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
}

#[derive(Clone)]
pub struct NaiveLweLookupEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
    c_b_by_slot: Family<Mat>,
}

impl LweLookupPublicKeyLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = LweLookupInvocation>,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)? })
    }
}

impl LweLookupEncodingLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = LweLookupInvocation>,
        c_b: Mat,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)?, c_b })
    }
}

impl LweLookupPolyEncodingLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = LweLookupInvocation>,
        c_b_by_slot: Family<Mat>,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)?, c_b_by_slot })
    }
}

impl NaiveLweLookupPublicKeyLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = NaiveLweLookupInvocation>,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_naive_invocations(invocations)? })
    }
}

impl NaiveLweLookupEncodingLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = NaiveLweLookupInvocation>,
        c_b_by_slot: Family<Mat>,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_naive_invocations(invocations)?, c_b_by_slot })
    }
}

fn collect_invocations(
    invocations: impl IntoIterator<Item = LweLookupInvocation>,
) -> Result<BTreeMap<LweLookupIdentity, LweLookupInvocation>, LweLookupCompileError> {
    let mut collected = BTreeMap::new();
    for invocation in invocations {
        let identity = invocation.compiler.identity.clone();
        if collected.insert(identity, invocation).is_some() {
            return Err(LweLookupCompileError::DuplicateInvocation);
        }
    }
    Ok(collected)
}

fn collect_naive_invocations(
    invocations: impl IntoIterator<Item = NaiveLweLookupInvocation>,
) -> Result<BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>, LweLookupCompileError> {
    let mut collected = BTreeMap::new();
    for invocation in invocations {
        if collected.insert(invocation.identity.clone(), invocation).is_some() {
            return Err(LweLookupCompileError::DuplicateInvocation);
        }
    }
    Ok(collected)
}

impl CircuitLoweringTypes for LweLookupPublicKeyLowering {
    type Wire = BggPublicKeyWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupPublicKeyLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        _input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let invocation = invocation_for_gate(&self.invocations, circuit, lookup_id, gate)?;
        let artifacts = invocation
            .compiler
            .import_artifacts(&invocation.artifacts)
            .map_err(|source| lookup_error(gate, source))?;
        Ok(invocation.compiler.public_key(&artifacts))
    }
}

impl CircuitLoweringTypes for LweLookupEncodingLowering {
    type Wire = BggEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupEncodingLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let invocation = invocation_for_gate(&self.invocations, circuit, lookup_id, gate)?;
        let artifacts = invocation
            .compiler
            .import_artifacts(&invocation.artifacts)
            .map_err(|source| lookup_error(gate, source))?;
        invocation
            .compiler
            .encoding(input, &self.c_b, &artifacts)
            .map_err(|source| lookup_error(gate, source))
    }
}

impl CircuitLoweringTypes for LweLookupPolyEncodingLowering {
    type Wire = BggPolyEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupPolyEncodingLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let invocation = invocation_for_gate(&self.invocations, circuit, lookup_id, gate)?;
        let artifacts = invocation
            .compiler
            .import_artifacts(&invocation.artifacts)
            .map_err(|source| lookup_error(gate, source))?;
        invocation
            .compiler
            .poly_encoding(input, &self.c_b_by_slot, &artifacts)
            .map_err(|source| lookup_error(gate, source))
    }
}

impl CircuitLoweringTypes for NaiveLweLookupPublicKeyLowering {
    type Wire = NaiveBggPublicKeyVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for NaiveLweLookupPublicKeyLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let invocation = naive_invocation_for_gate(&self.invocations, circuit, lookup_id, gate)?;
        require_slot_count(input.matrices.count(), invocation.slots.len(), gate)?;
        let matrices = Family::pack(
            invocation
                .slots
                .par_iter()
                .map(|slot| {
                    slot.compiler
                        .import_artifacts(&slot.artifacts)
                        .map(|artifacts| artifacts.output_public_key)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| lookup_error(gate, source))?,
        )?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true })
    }
}

impl CircuitLoweringTypes for NaiveLweLookupEncodingLowering {
    type Wire = NaiveBggEncodingVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for NaiveLweLookupEncodingLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let invocation = naive_invocation_for_gate(&self.invocations, circuit, lookup_id, gate)?;
        let plaintexts = input
            .plaintexts
            .as_ref()
            .ok_or_else(|| lookup_error(gate, LweLookupCompileError::MissingPlaintext))?;
        for count in [
            input.vectors.count(),
            input.pubkeys.count(),
            plaintexts.count(),
            self.c_b_by_slot.count(),
        ] {
            require_slot_count(count, invocation.slots.len(), gate)?;
        }

        let outputs = invocation
            .slots
            .par_iter()
            .enumerate()
            .map(|(slot_index, slot)| {
                let artifacts = slot
                    .compiler
                    .import_artifacts(&slot.artifacts)
                    .map_err(|source| lookup_error(gate, source))?;
                slot.compiler
                    .encoding(
                        &BggEncodingWire {
                            vector: input.vectors.get_static(slot_index),
                            pubkey: BggPublicKeyWire {
                                matrix: input.pubkeys.get_static(slot_index),
                                reveal_plaintext: input.pubkey_reveal_plaintext,
                            },
                            plaintext: Some(plaintexts.get_static(slot_index)),
                        },
                        &self.c_b_by_slot.get_static(slot_index),
                        &artifacts,
                    )
                    .map_err(|source| lookup_error(gate, source))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut vectors = Vec::with_capacity(outputs.len());
        let mut public_keys = Vec::with_capacity(outputs.len());
        let mut output_plaintexts = Vec::with_capacity(outputs.len());
        for output in outputs {
            vectors.push(output.vector);
            public_keys.push(output.pubkey.matrix);
            output_plaintexts.push(output.plaintext.expect("lookup reveals its output"));
        }
        Ok(NaiveBggEncodingVecWire {
            vectors: Family::pack(vectors)?,
            pubkeys: Family::pack(public_keys)?,
            pubkey_reveal_plaintext: true,
            plaintexts: Some(Family::pack(output_plaintexts)?),
        })
    }
}

fn identity_for_gate(gate: GateInstance<'_>, lookup: usize) -> LweLookupIdentity {
    LweLookupIdentity {
        call_path: gate.call_path().to_vec(),
        gate: gate.local_gate().index(),
        occurrence: gate.operation_occurrence(),
        lookup,
        slot: None,
    }
}

fn invocation_for_gate<'a, P: Poly>(
    invocations: &'a BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    circuit: &PolyCircuit<P>,
    lookup_id: usize,
    gate: GateInstance<'_>,
) -> Result<&'a LweLookupInvocation, CircuitCompileError> {
    let identity = identity_for_gate(gate, lookup_id);
    let invocation = invocations
        .get(&identity)
        .ok_or_else(|| lookup_error(gate, LweLookupCompileError::MissingInvocation))?;
    let circuit_table = circuit.lookup_table(lookup_id);
    if invocation.circuit_table_identity != std::sync::Arc::as_ptr(&circuit_table) as usize {
        return Err(lookup_error(gate, LweLookupCompileError::CircuitTableBinding));
    }
    Ok(invocation)
}

fn naive_invocation_for_gate<'a, P: Poly>(
    invocations: &'a BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
    circuit: &PolyCircuit<P>,
    lookup_id: usize,
    gate: GateInstance<'_>,
) -> Result<&'a NaiveLweLookupInvocation, CircuitCompileError> {
    let identity = identity_for_gate(gate, lookup_id);
    let invocation = invocations
        .get(&identity)
        .ok_or_else(|| lookup_error(gate, LweLookupCompileError::MissingInvocation))?;
    let circuit_table = circuit.lookup_table(lookup_id);
    if invocation.circuit_table_identity != std::sync::Arc::as_ptr(&circuit_table) as usize {
        return Err(lookup_error(gate, LweLookupCompileError::CircuitTableBinding));
    }
    Ok(invocation)
}

fn require_slot_count(
    count: &IntExpr,
    expected: usize,
    gate: GateInstance<'_>,
) -> Result<(), CircuitCompileError> {
    if count.canonicalize() != IntExpr::constant(expected) {
        return Err(lookup_error(gate, LweLookupCompileError::SlotCountMismatch));
    }
    Ok(())
}

fn lookup_error(gate: GateInstance<'_>, source: LweLookupCompileError) -> CircuitCompileError {
    CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BggEncodingCompiler, BggPublicKeyCompiler};
    use mxx_dsl::DslContext;
    use mxx_gadgets::circuit::{PolyCircuit, PublicLut};
    use mxx_ir_core::{
        ParamEnv,
        artifact::{ArtifactType, SpecHash},
        node::NodeKind,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly as ConcretePoly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::{ArtifactPayload, ArtifactStore, MemoryArtifactStore},
        backend::{Backend, poly::cpu_backend},
        execute,
        transcript::SamplingMode,
    };

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn compiler(
        parameters: &DCRTPolyParams,
        identity: LweLookupIdentity,
        table: LweLookupTable,
    ) -> LweLookupCompiler {
        let digit_count = parameters.modulus_digits();
        LweLookupCompiler {
            identity,
            table,
            public_key_type: matrix_type(parameters, 1, digit_count),
            low_matrix_type: matrix_type(parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        }
    }

    #[test]
    fn tags_and_dense_row_validation_match_the_existing_lwe_layout() {
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 9,
            occurrence: 0,
            lookup: 4,
            slot: Some(2),
        };
        assert_eq!(identity.output_public_key_tag(), b"A_LT_9_slot2");
        assert_eq!(identity.low_matrix_tag(3), b"LWE_R_G_9_4_3_slot2");
        assert!(matches!(
            LweLookupTable::new([(0, BigInt::from(1)), (0, BigInt::from(2))]),
            Err(LweLookupCompileError::DuplicateRow(0))
        ));
        assert!(matches!(
            LweLookupTable::new([(0, BigInt::from(1)), (2, BigInt::from(2))]),
            Err(LweLookupCompileError::RowOutOfRange { row: 2, length: 2 })
        ));
    }

    #[test]
    fn preprocessing_artifacts_drive_the_online_lwe_equation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                let (row, output) = if input == 0 { (1, 11) } else { (0, 13) };
                Some((
                    row,
                    <DCRTPoly as ConcretePoly>::Elem::constant(&parameters.modulus(), output),
                ))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: output_gate.as_single_wire().index(),
                occurrence: 0,
                lookup: lookup_id,
                slot: None,
            },
            LweLookupTable::from_public_lut(&parameters, circuit.lookup_table(lookup_id).as_ref())
                .expect("permutation table"),
        );
        let names = LweLookupArtifactNames::for_compiler(&lookup);
        let ring = lookup.ring();
        let trapdoor =
            ring.sample_trapdoor(1, 5, lookup.gadget_base.clone(), lookup.digit_count.clone());
        let wires = lookup
            .preprocess(
                ring.bytes_input("hash-key", 32),
                &BggPublicKeyWire {
                    matrix: ring.input("input-public-key", (1, digit_count)),
                    reveal_plaintext: true,
                },
                &trapdoor,
            )
            .expect("preprocessing graph");
        let producer = lookup
            .export_preprocessing(DslContext::new("lwe-lookup-producer"), wires, &names)
            .expect("preprocessing outputs")
            .build()
            .expect("producer graph");
        assert_eq!(
            producer
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            1
        );
        let producer_symbolic = producer.elaborate(&ParamEnv::default()).expect("producer IR");
        let producer = producer.validate(&ParamEnv::default()).expect("validated producer");
        let mut store = MemoryArtifactStore::default();
        let mut producer_backend = cpu_backend([parameters.clone()]);
        let produced = execute(
            &producer,
            &mut producer_backend,
            BTreeMap::from([
                ("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
                (
                    "input-public-key".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, 1, digit_count)),
                ),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("preprocessing execution");
        assert_eq!(producer_backend.preimage_batch_calls(), 1);
        let production_id = produced.production_id.clone().expect("artifact production");
        let manifest = store.manifest(&production_id).expect("manifest").clone();
        let symbolic_exports = producer_symbolic
            .outputs
            .iter()
            .map(|(name, reference)| {
                let wire = producer_symbolic.wire(reference).expect("symbolic artifact output");
                (
                    name.clone(),
                    mxx_ir_symbolic::manifest::ExportArtifact {
                        wire_type: wire.wire_type.clone(),
                        expression: wire.expression,
                        family: wire.family.clone(),
                        content_hash: None,
                        layout: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let symbolic_manifest = mxx_ir_symbolic::manifest::export_manifest(
            production_id.clone(),
            &symbolic_exports,
            &producer_symbolic.atoms,
            &producer_symbolic.expressions,
            &producer_symbolic.preimage_relations,
            producer_symbolic.assumption_digest,
        )
        .expect("symbolic manifest");
        assert_eq!(manifest.artifacts[&names.low_matrices].family_count, Some(lookup.table.len()));
        let selected_low_handle = &produced.artifact_handles[&names.low_matrices][1];
        let selected_low_descriptor = &manifest.artifacts[&names.low_matrices];
        let ArtifactPayload::Matrix(selected_low_bytes) = store
            .load(&selected_low_handle.key, selected_low_descriptor)
            .expect("selected low matrix")
        else {
            panic!("matrix artifact")
        };
        let ArtifactType::Matrix(selected_low_type) = &selected_low_handle.artifact_type else {
            panic!("matrix artifact type")
        };
        let selected_low = producer_backend
            .matrix_from_bytes(selected_low_type, &selected_low_bytes)
            .expect("decode selected low matrix");
        let selected_high_handle = &produced.artifact_handles[&names.high_matrices][1];
        let selected_high_descriptor = &manifest.artifacts[&names.high_matrices];
        let ArtifactPayload::Matrix(selected_high_bytes) = store
            .load(&selected_high_handle.key, selected_high_descriptor)
            .expect("selected high matrix")
        else {
            panic!("matrix artifact")
        };
        let ArtifactType::Matrix(selected_high_type) = &selected_high_handle.artifact_type else {
            panic!("matrix artifact type")
        };
        let selected_high = producer_backend
            .matrix_from_bytes(selected_high_type, &selected_high_bytes)
            .expect("decode selected high matrix");

        let artifacts = LweLookupArtifacts::for_compiler(production_id.clone(), &lookup);
        let invocation =
            LweLookupInvocation::bind(lookup.clone(), artifacts, &parameters, &circuit)
                .expect("circuit lookup invocation");
        let input_vector = ring.input("input-vector", (1, digit_count));
        let input = BggEncodingWire {
            vector: input_vector.clone(),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("consumer-public-key", (1, digit_count)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.input("input-plaintext", (1, 1))),
        };
        let one = BggEncodingWire {
            vector: ring.zero((1, digit_count)),
            pubkey: BggPublicKeyWire {
                matrix: ring.zero((1, digit_count)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.polynomial([IntExpr::constant(1)])),
        };
        let standard = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: lookup.gadget_base.clone(),
            digit_count: lookup.digit_count.clone(),
        };
        let mut lookup_lowering =
            LweLookupEncodingLowering::new([invocation], ring.input("c-b", (1, digit_count + 2)))
                .expect("lookup lowering");
        let mut slots = crate::NoSlotOperations::default();
        let output = crate::PolyCircuitCompiler { public_key: standard }
            .compile_encodings_with_lowerings(
                &circuit,
                one,
                [input],
                &mut lookup_lowering,
                &mut slots,
            )
            .expect("circuit online graph")
            .remove(0);
        let consumer = DslContext::new("lwe-lookup-consumer")
            .output("vector", output.vector)
            .expect("vector")
            .output("plaintext", output.plaintext.expect("revealed output"))
            .expect("plaintext")
            .build()
            .expect("consumer graph");
        let consumer_symbolic = consumer
            .elaborate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id.clone(), manifest.clone())]),
                &[symbolic_manifest],
            )
            .expect("consumer symbolic IR");
        let output_expression = consumer_symbolic
            .wire(&consumer_symbolic.outputs["vector"])
            .and_then(|wire| wire.expression)
            .expect("lookup output expression");
        assert!(matches!(
            consumer_symbolic
                .expressions
                .get(output_expression)
                .expect("lookup output expression")
                .node,
            mxx_ir_symbolic::SymbolicExprNode::Add(_)
        ));
        let consumer = consumer
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .expect("validated consumer");
        let input_vector_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            (0..digit_count)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        &parameters,
                        index % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        );
        let c_b_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            (0..digit_count + 2)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        &parameters,
                        index % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        );
        let expected_vector =
            c_b_value.clone() * &selected_high + input_vector_value.clone() * &selected_low;
        let consumed = execute(
            &consumer,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([
                ("input-vector".to_owned(), RuntimeValue::matrix(input_vector_value)),
                (
                    "consumer-public-key".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, 1, digit_count)),
                ),
                (
                    "input-plaintext".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                        &parameters,
                        vec![DCRTPoly::from_usize_to_constant(&parameters, 1)],
                    )),
                ),
                ("c-b".to_owned(), RuntimeValue::matrix(c_b_value)),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("online execution");
        let RuntimeValue::Matrix(actual_vector) = &consumed.outputs["vector"] else {
            panic!("vector output")
        };
        assert_eq!(actual_vector.as_ref(), &expected_vector);
        let RuntimeValue::Matrix(actual_plaintext) = &consumed.outputs["plaintext"] else {
            panic!("plaintext output")
        };
        assert_eq!(
            actual_plaintext.as_ref(),
            &DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_usize_to_constant(&parameters, 13)],
            )
        );
    }

    #[test]
    fn circuit_public_lookup_lowers_to_lazy_artifact_selection() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((
                    input,
                    <DCRTPoly as ConcretePoly>::Elem::constant(&parameters.modulus(), input),
                ))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: output_gate.as_single_wire().index(),
                occurrence: 0,
                lookup: lookup_id,
                slot: None,
            },
            LweLookupTable::from_public_lut(&parameters, circuit.lookup_table(lookup_id).as_ref())
                .expect("table"),
        );
        let invocation = LweLookupInvocation::bind(
            lookup.clone(),
            LweLookupArtifacts::for_compiler(
                ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
                &lookup,
            ),
            &parameters,
            &circuit,
        )
        .expect("invocation");
        let ring = lookup.ring();
        let standard = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: lookup.gadget_base.clone(),
                digit_count: lookup.digit_count.clone(),
            },
        };
        let circuit_compiler =
            crate::PolyCircuitCompiler { public_key: standard.public_key.clone() };
        let public_key = |prefix: &str| BggPublicKeyWire {
            matrix: ring.input(format!("{prefix}-public"), (1, digit_count)),
            reveal_plaintext: true,
        };
        let mut public_key_lowering =
            LweLookupPublicKeyLowering::new([invocation.clone()]).expect("public-key lowering");
        let mut public_key_slots = crate::NoSlotOperations::default();
        let public_key_outputs = circuit_compiler
            .compile_public_keys_with_lowerings(
                &circuit,
                public_key("one"),
                [public_key("input")],
                &mut public_key_lowering,
                &mut public_key_slots,
            )
            .expect("public-key lookup lowering");
        let public_key_graph = DslContext::new("lwe-lookup-public-key-lowering")
            .output("public-key", public_key_outputs[0].matrix.clone())
            .expect("public key")
            .build()
            .expect("public-key graph");
        assert_eq!(
            public_key_graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            1
        );
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, digit_count)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (1, digit_count)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let mut lowering =
            LweLookupEncodingLowering::new([invocation], ring.input("c-b", (1, digit_count + 2)))
                .expect("lowering");
        let mut encoding_slots = crate::NoSlotOperations::default();
        let outputs = circuit_compiler
            .compile_encodings_with_lowerings(
                &circuit,
                encoding("one"),
                [encoding("input")],
                &mut lowering,
                &mut encoding_slots,
            )
            .expect("lookup lowering");
        let built = DslContext::new("lwe-lookup-lowering")
            .output("vector", outputs[0].vector.clone())
            .expect("output")
            .output("public-key", outputs[0].pubkey.matrix.clone())
            .expect("public key")
            .output("plaintext", outputs[0].plaintext.clone().expect("lookup output plaintext"))
            .expect("plaintext")
            .build()
            .expect("graph");
        let nodes = built.graph.root_scope().nodes();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            3
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic)).count(),
            2
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::Select { .. })).count(),
            1
        );
    }

    #[test]
    fn poly_and_naive_lookup_lowerings_build_structural_family_graphs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((
                    input,
                    <DCRTPoly as ConcretePoly>::Elem::constant(&parameters.modulus(), input),
                ))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: output_gate.as_single_wire().index(),
            occurrence: 0,
            lookup: lookup_id,
            slot: None,
        };
        let table =
            LweLookupTable::from_public_lut(&parameters, circuit.lookup_table(lookup_id).as_ref())
                .expect("table");
        let lookup = compiler(&parameters, identity.clone(), table.clone());
        let invocation = LweLookupInvocation::bind(
            lookup.clone(),
            LweLookupArtifacts::for_compiler(
                ProductionId { spec_hash: SpecHash([30; 32]), execution_nonce: [31; 32] },
                &lookup,
            ),
            &parameters,
            &circuit,
        )
        .expect("poly invocation");
        let ring = lookup.ring();
        let public_key_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: lookup.gadget_base.clone(),
            digit_count: lookup.digit_count.clone(),
        };
        let poly_encoding = |prefix: &str| BggPolyEncodingWire {
            vectors: ring.input_family(format!("{prefix}-vectors"), 2, (1, digit_count)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (1, digit_count)),
                reveal_plaintext: true,
            },
            plaintexts: Some(ring.input_family(format!("{prefix}-plaintexts"), 2, (1, 1))),
        };
        let mut poly_lowering = LweLookupPolyEncodingLowering::new(
            [invocation],
            ring.input_family("poly-c-b", 2, (1, digit_count + 2)),
        )
        .expect("poly lowering");
        let mut poly_slots = crate::NoSlotOperations::default();
        let circuit_compiler =
            crate::PolyCircuitCompiler { public_key: public_key_compiler.clone() };
        let poly_outputs = circuit_compiler
            .compile_poly_encodings_with_lowerings(
                &circuit,
                poly_encoding("poly-one"),
                [poly_encoding("poly-input")],
                &mut poly_lowering,
                &mut poly_slots,
            )
            .expect("poly lookup lowering");
        let poly_graph = DslContext::new("poly-lwe-lookup")
            .family_output("vectors", poly_outputs[0].vectors.clone())
            .expect("vectors")
            .family_output("plaintexts", poly_outputs[0].plaintexts.clone().expect("plaintexts"))
            .expect("plaintexts")
            .output("public", poly_outputs[0].pubkey.matrix.clone())
            .expect("public")
            .build()
            .expect("poly graph");
        assert!(poly_graph.graph.scopes().values().any(|scope| {
            scope.nodes().iter().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic))
        }));

        let slots = (0..2)
            .map(|slot| {
                let slot_lookup = compiler(
                    &parameters,
                    LweLookupIdentity { slot: Some(slot), ..identity.clone() },
                    table.clone(),
                );
                LweLookupInvocation::bind(
                    slot_lookup.clone(),
                    LweLookupArtifacts::for_compiler(
                        ProductionId {
                            spec_hash: SpecHash([40 + slot as u8; 32]),
                            execution_nonce: [50 + slot as u8; 32],
                        },
                        &slot_lookup,
                    ),
                    &parameters,
                    &circuit,
                )
                .expect("slot invocation")
            })
            .collect::<Vec<_>>();
        let naive_invocation = NaiveLweLookupInvocation::new(slots).expect("naive invocation");
        let naive_public_key = |prefix: &str| NaiveBggPublicKeyVecWire {
            matrices: ring.input_family(format!("{prefix}-public"), 2, (1, digit_count)),
            reveal_plaintext: true,
        };
        let mut naive_public_key_lowering =
            NaiveLweLookupPublicKeyLowering::new([naive_invocation.clone()])
                .expect("naive public-key lowering");
        let mut naive_public_key_slots = crate::NoSlotOperations::default();
        let naive_public_keys = circuit_compiler
            .compile_naive_public_keys_with_lowerings(
                &circuit,
                naive_public_key("naive-one"),
                [naive_public_key("naive-input")],
                &mut naive_public_key_lowering,
                &mut naive_public_key_slots,
            )
            .expect("naive public-key lookup lowering");
        let naive_public_key_graph = DslContext::new("naive-lwe-public-key-lookup")
            .family_output("public", naive_public_keys[0].matrices.clone())
            .expect("public")
            .build()
            .expect("naive public-key graph");
        assert_eq!(
            naive_public_key_graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            2
        );

        let naive = |prefix: &str| NaiveBggEncodingVecWire {
            vectors: ring.input_family(format!("{prefix}-vectors"), 2, (1, digit_count)),
            pubkeys: ring.input_family(format!("{prefix}-public"), 2, (1, digit_count)),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(ring.input_family(format!("{prefix}-plaintexts"), 2, (1, 1))),
        };
        let mut naive_lowering = NaiveLweLookupEncodingLowering::new(
            [naive_invocation],
            ring.input_family("naive-c-b", 2, (1, digit_count + 2)),
        )
        .expect("naive lowering");
        let mut naive_slots = crate::NoSlotOperations::default();
        let naive_outputs = circuit_compiler
            .compile_naive_encodings_with_lowerings(
                &circuit,
                naive("naive-one"),
                [naive("naive-input")],
                &mut naive_lowering,
                &mut naive_slots,
            )
            .expect("naive lookup lowering");
        let naive_graph = DslContext::new("naive-lwe-lookup")
            .family_output("vectors", naive_outputs[0].vectors.clone())
            .expect("vectors")
            .family_output("public", naive_outputs[0].pubkeys.clone())
            .expect("public")
            .family_output("plaintexts", naive_outputs[0].plaintexts.clone().expect("plaintexts"))
            .expect("plaintexts")
            .build()
            .expect("naive graph");
        assert_eq!(
            naive_graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            6
        );
    }
}
