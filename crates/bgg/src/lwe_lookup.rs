//! LWE-based public lookup-table evaluation expressed with the declarative DSL.

use crate::{
    BggEncodingWire, BggPublicKeyWire, CircuitCompileError, NaiveBggEncodingVecWire,
    NaiveBggPublicKeyVecWire,
    tall_encoding::{BggTallEncodingWire, BggTallPlaintext},
};
use mxx_dsl::{
    Bytes, DslContext, DslError, Family, HashTag, Int, Mat, Preimage, Ring, Trapdoor, parallel_zip,
};
#[cfg(test)]
use mxx_dsl::{MatFamilyType, Subgraph};
use mxx_gadgets::{
    Poly,
    circuit::{
        ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
        PolyCircuit, PolyGateKind, PublicLookupLowering, PublicLutProgram, SlotOperationLowering,
        lower_circuit,
    },
};
use mxx_ir_core::{
    IntExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    types::MatrixType,
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

const LWE_LOOKUP_ARTIFACT_SEMANTIC_VERSION: u32 = 2;

struct LweLookupIdentityCollector {
    identities: Vec<LweLookupIdentity>,
}

impl CircuitLoweringTypes for LweLookupIdentityCollector {
    type Wire = ();
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for LweLookupIdentityCollector {
    fn binary(
        &mut self,
        _operation: PolyGateKind,
        _lhs: &Self::Wire,
        _rhs: &Self::Wire,
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }

    fn small_scalar_mul(
        &mut self,
        _input: &Self::Wire,
        _scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }

    fn large_scalar_mul(
        &mut self,
        _input: &Self::Wire,
        _scalar: &[num_bigint::BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupIdentityCollector {
    fn public_lookup(
        &mut self,
        _circuit: &PolyCircuit<P>,
        lookup_id: usize,
        _input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.identities.push(identity_for_gate(gate, lookup_id));
        Ok(())
    }
}

impl<P: Poly> SlotOperationLowering<P> for LweLookupIdentityCollector {
    fn slot_transfer(
        &mut self,
        _input: &Self::Wire,
        _source_slots: &[(u32, Option<u32>)],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }

    fn slot_reduce(
        &mut self,
        _inputs: &[Self::Wire],
        _slot_count: usize,
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }

    fn slot_identity_repeated_lanes(
        &mut self,
        _input: &Self::Wire,
        _num_blocks: u32,
        _lane_scalars: &[Option<u32>],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }

    fn slot_anchor_reduce(
        &mut self,
        _input: &Self::Wire,
        _num_blocks: u32,
        _lane_scalars: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(())
    }
}

/// Enumerates every concrete public-lookup invocation after recursively
/// expanding direct, summed, and repeated sub-circuit calls. The identities
/// are exactly those consumed by the LWE lookup lowerings, including call
/// paths and operation occurrences.
pub fn collect_lwe_lookup_identities<P: Poly>(
    circuit: &PolyCircuit<P>,
) -> Result<Vec<LweLookupIdentity>, CircuitCompileError> {
    collect_lwe_lookup_identities_with_prefix(circuit, &[])
}

pub fn collect_lwe_lookup_identities_with_prefix<P: Poly>(
    circuit: &PolyCircuit<P>,
    call_path_prefix: &[usize],
) -> Result<Vec<LweLookupIdentity>, CircuitCompileError> {
    let mut collector = LweLookupIdentityCollector { identities: Vec::new() };
    lower_circuit(circuit, (), std::iter::repeat_n((), circuit.num_input()), &mut collector)
        .map_err(|error| match error {
            CircuitLowerError::Operation { source, .. } => source,
            other => CircuitCompileError::Structure(other.to_string()),
        })?;
    Ok(collector
        .identities
        .into_iter()
        .map(|mut identity| {
            let mut call_path = call_path_prefix.to_vec();
            call_path.extend(identity.call_path);
            identity.call_path = call_path;
            identity
        })
        .collect())
}

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

    fn low_matrix_tag_prefix(&self) -> Vec<u8> {
        format!(
            "mxx-bgg/lwe-lookup-low/v2:{}:{}:slot{}:row:",
            self.gate_token(),
            self.lookup,
            self.slot_index()
        )
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
/// Rows must be a permutation of `0..len`; row identity determines the `K_low` hash tag while
/// artifact families remain in input order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupTable {
    entries: Arc<[LweLookupEntry]>,
    /// Exclusive canonical upper bound for every output, when every static
    /// table value is nonnegative. This is table metadata, not a new IR value.
    canonical_output_exclusive_upper: Option<BigUint>,
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
        let mut canonical_output_maximum = Some(BigUint::zero());
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
            match (canonical_output_maximum.as_mut(), entry.output.to_biguint()) {
                (Some(maximum), Some(output)) => *maximum = maximum.clone().max(output),
                (_, None) => canonical_output_maximum = None,
                (None, Some(_)) => {}
            }
        }
        if let Some(row) = seen.iter().position(|present| !present) {
            return Err(LweLookupCompileError::MissingRow(row));
        }
        let canonical_output_exclusive_upper =
            canonical_output_maximum.map(|maximum| maximum + BigUint::from(1u8));
        Ok(Self { entries: entries.into(), canonical_output_exclusive_upper })
    }

    pub fn from_public_lut(table: &PublicLutProgram) -> Result<Self, LweLookupCompileError> {
        let mut entries = vec![None; table.len()];
        for (input, (row, output)) in table.entries() {
            let input = usize::try_from(input)
                .map_err(|_| LweLookupCompileError::InputOutOfRange(input))?;
            let row =
                usize::try_from(row).map_err(|_| LweLookupCompileError::RowIndexTooLarge(row))?;
            if input >= entries.len() {
                return Err(LweLookupCompileError::InputOutOfRange(input as u64));
            }
            entries[input] = Some((row, output));
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

    fn canonical_output_exclusive_upper_for_modulus(&self, modulus: &IntExpr) -> Option<BigUint> {
        let upper = self.canonical_output_exclusive_upper.as_ref()?;
        let IntExpr::Const(modulus) = modulus else {
            return None;
        };
        (upper <= &modulus.to_biguint()?).then(|| upper.clone())
    }

    fn commitment(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-bgg/lwe-lookup-table/v2");
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
    pub output_plaintexts: String,
}

impl LweLookupArtifactNames {
    pub fn for_compiler(compiler: &LweLookupCompiler) -> Self {
        let identity = &compiler.identity;
        let prefix = format!(
            "lwe_lookup_v2_{}_{}_slot{}",
            identity.gate_token(),
            identity.lookup,
            identity.slot_index()
        );
        Self {
            output_public_key: format!("{prefix}_output_public_key"),
            low_matrices: format!("{prefix}_low_matrices"),
            high_matrices: format!("{prefix}_high_matrices"),
            output_plaintexts: format!("{prefix}_output_plaintexts"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupArtifacts {
    production_id: ProductionId,
    semantic_version: u32,
    table_length: usize,
    table_commitment: [u8; 32],
}

impl LweLookupArtifacts {
    pub fn for_compiler(production_id: ProductionId, compiler: &LweLookupCompiler) -> Self {
        Self {
            production_id,
            semantic_version: LWE_LOOKUP_ARTIFACT_SEMANTIC_VERSION,
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
    pub low_matrices: Family<Preimage>,
    pub high_matrices: Family<Preimage>,
    /// Public LUT outputs indexed by the logical input value.
    pub output_plaintexts: Family<Mat>,
}

#[derive(Clone)]
pub struct LweLookupArtifactWires {
    pub output_public_key: Mat,
    /// Logical lookup family indexed directly by the table input `x`.
    ///
    /// The producer and evaluator share this table-length domain; no physical chunk coordinate is
    /// part of the checker-visible graph identity.
    pub low_matrices: Family<Preimage>,
    pub high_matrices: Family<Preimage>,
    /// The authoritative public output plaintext family, indexed by the lookup input.
    pub output_plaintexts: Family<Mat>,
}

#[derive(Clone)]
pub struct LweLookupPreprocessingEntry {
    pub compiler: LweLookupCompiler,
    pub wires: LweLookupPreprocessingWires,
}

impl LweLookupPreprocessingEntry {
    pub fn export(&self, context: DslContext) -> Result<DslContext, LweLookupCompileError> {
        self.compiler.export_preprocessing(
            context,
            self.wires.clone(),
            &LweLookupArtifactNames::for_compiler(&self.compiler),
        )
    }
}

#[derive(Clone)]
pub struct NaiveLweLookupPreprocessingEntry {
    pub compilers: Vec<LweLookupCompiler>,
    pub wires: Vec<LweLookupPreprocessingWires>,
}

impl NaiveLweLookupPreprocessingEntry {
    pub fn export(&self, mut context: DslContext) -> Result<DslContext, LweLookupCompileError> {
        for (compiler, wires) in self.compilers.iter().zip(&self.wires) {
            context = compiler.export_preprocessing(
                context,
                wires.clone(),
                &LweLookupArtifactNames::for_compiler(compiler),
            )?;
        }
        Ok(context)
    }
}

pub struct LweLookupPreprocessingLowering<P: Poly> {
    parameters: P::Params,
    hash_key: Bytes,
    trapdoor: Trapdoor,
    gadget_base: IntExpr,
    digit_count: IntExpr,
    call_path_prefix: Vec<usize>,
    registry_tables: BTreeMap<LookupRegistryKey, LweLookupTable>,
    tables: BTreeMap<[u8; 32], LweLookupTable>,
    table_families: BTreeMap<[u8; 32], (Family<Int>, Family<Mat>)>,
    entries: Vec<LweLookupPreprocessingEntry>,
}

pub struct NaiveLweLookupPreprocessingLowering<P: Poly> {
    parameters: P::Params,
    hash_key: Bytes,
    trapdoors: Vec<Trapdoor>,
    gadget_base: IntExpr,
    digit_count: IntExpr,
    slot_count: usize,
    call_path_prefix: Vec<usize>,
    registry_tables: BTreeMap<LookupRegistryKey, LweLookupTable>,
    tables: BTreeMap<[u8; 32], LweLookupTable>,
    table_families: BTreeMap<[u8; 32], (Family<Int>, Family<Mat>)>,
    entries: Vec<NaiveLweLookupPreprocessingEntry>,
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
    #[error("lookup artifact semantic version {actual} does not match required version {expected}")]
    ArtifactSemanticVersion { expected: u32, actual: u32 },
    #[error("more than one LWE lookup invocation has the same structural identity")]
    DuplicateInvocation,
    #[error("the circuit gate has no bound LWE lookup invocation")]
    MissingInvocation,
    #[error("a naive LWE lookup invocation requires at least one slot")]
    EmptySlotInvocations,
    #[error("naive LWE lookup slot {slot} has an inconsistent structural identity")]
    SlotIdentity { slot: usize },
    #[error("LWE lookup circuit traversal failed: {0}")]
    CircuitStructure(String),
    #[error("tall lookup kernel requires exactly six family inputs")]
    TallKernelSchema,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl LweLookupInvocation {
    pub fn bind<P: Poly>(
        compiler: LweLookupCompiler,
        artifacts: LweLookupArtifacts,
        _parameters: &P::Params,
        circuit: &PolyCircuit<P>,
    ) -> Result<Self, LweLookupCompileError> {
        let circuit_table = circuit.lookup_table(compiler.identity.lookup);
        let materialized = LweLookupTable::from_public_lut(circuit_table.as_ref())?;
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
    /// Number of rows in this public LUT, and therefore the number of high preimages required
    /// when its preprocessing artifacts are generated.
    pub fn table_length(&self) -> usize {
        self.table.len()
    }

    /// Builds `K_high = Preimage_B(A_lt - yG - (A_z - xG)K_low)`.
    pub fn preprocess(
        &self,
        hash_key: Bytes,
        input_public_key: &BggPublicKeyWire,
        trapdoor: &Trapdoor,
    ) -> Result<LweLookupPreprocessingWires, LweLookupCompileError> {
        let table_families = self.table_families()?;
        self.preprocess_with_table_families(hash_key, input_public_key, trapdoor, &table_families)
    }

    fn preprocess_with_table_families(
        &self,
        hash_key: Bytes,
        input_public_key: &BggPublicKeyWire,
        trapdoor: &Trapdoor,
        table_families: &(Family<Int>, Family<Mat>),
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
        let gadget = ring.gadget(
            self.public_key_type.rows.clone(),
            self.gadget_base.clone(),
            self.digit_count.clone(),
        );
        let (rows, output_plaintexts) = (table_families.0.clone(), table_families.1.clone());
        let input_public_key = input_public_key.matrix.clone();
        let output_for_loop = output_public_key.clone();
        let trapdoor = trapdoor.clone();
        let gadget = gadget.clone();
        let high_shape = shape(&self.high_matrix_type);
        let low_shape = shape(&self.low_matrix_type);
        let scalar_type = ring.matrix_type((1, 1));
        let low_tag_prefix = self.identity.low_matrix_tag_prefix();
        let gadget_base = self.gadget_base.clone();
        let digit_count = self.digit_count.clone();
        let (low_matrices, high_matrices) =
            parallel_zip((rows, output_plaintexts.clone()), move |index, (row, output)| {
                let input_scalar = index
                    .as_int()
                    .add(Int::constant(0))
                    .lift_to_constant_polynomial(scalar_type.clone());
                let mut low_tag = HashTag::from(low_tag_prefix.clone());
                low_tag.push(row);
                let low_plain_shape = (
                    IntExpr::Div(Box::new(low_shape.0.clone()), Box::new(digit_count.clone())),
                    low_shape.1.clone(),
                );
                let low = ring
                    .hash_matrix(hash_key.clone(), low_tag, low_plain_shape)
                    .decompose(gadget_base.clone(), digit_count.clone())
                    .into_preimage_relation();
                let extended_input = input_public_key.clone() - gadget.clone() * input_scalar;
                let target = output_for_loop.clone() - gadget.clone() * output;
                let adjusted_target = target - extended_input * low.clone().materialize_exact();
                let high = trapdoor.sample_preimage(adjusted_target, high_shape.clone());
                (low, high)
            })?;
        Ok(LweLookupPreprocessingWires {
            output_public_key,
            low_matrices,
            high_matrices,
            output_plaintexts,
        })
    }

    fn table_families(&self) -> Result<(Family<Int>, Family<Mat>), LweLookupCompileError> {
        let ring = self.ring();
        Ok((
            Family::pack(
                self.table.entries.iter().map(|entry| Int::constant(entry.row)).collect(),
            )?,
            Family::pack(
                self.table
                    .entries
                    .iter()
                    .map(|entry| ring.polynomial([IntExpr::constant(entry.output.clone())]))
                    .collect(),
            )?,
        ))
    }

    pub fn export_preprocessing(
        &self,
        context: DslContext,
        wires: LweLookupPreprocessingWires,
        names: &LweLookupArtifactNames,
    ) -> Result<DslContext, LweLookupCompileError> {
        Ok(context
            .public_output(names.output_public_key.clone(), wires.output_public_key)?
            .public_preimage_family_output(names.low_matrices.clone(), wires.low_matrices)?
            .public_preimage_family_output(names.high_matrices.clone(), wires.high_matrices)?
            .public_family_output(names.output_plaintexts.clone(), wires.output_plaintexts)?)
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
            output_public_key: self.import_output_public_key(artifacts),
            low_matrices: ring.preimage_family_artifact_input(
                artifacts.production_id.clone(),
                names.low_matrices,
                vec![IntExpr::constant(artifacts.table_length)],
                shape(&self.low_matrix_type),
                ArtifactConfidentiality::Public,
            ),
            high_matrices: ring.preimage_family_artifact_input(
                artifacts.production_id.clone(),
                names.high_matrices,
                vec![IntExpr::constant(artifacts.table_length)],
                shape(&self.high_matrix_type),
                ArtifactConfidentiality::Public,
            ),
            output_plaintexts: ring.family_artifact_input(
                artifacts.production_id.clone(),
                names.output_plaintexts,
                artifacts.table_length,
                shape(&ring.matrix_type((1, 1))),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    fn import_output_public_key(&self, artifacts: &LweLookupArtifacts) -> Mat {
        self.ring().artifact_input(
            artifacts.production_id.clone(),
            LweLookupArtifactNames::for_compiler(self).output_public_key,
            shape(&self.public_key_type),
            ArtifactConfidentiality::Public,
        )
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
        // The lookup is only defined for inputs inside the table, so the table length is the
        // canonical upper bound of the selector coefficient.
        let input_index = plaintext.extract_coefficient_with_canonical_input_exclusive_upper(
            0,
            Some(BigUint::from(self.table.len())),
        );
        let low = artifacts.low_matrices.get(input_index.clone());
        let high = artifacts.high_matrices.get(input_index.clone());
        let output_plaintext = artifacts.output_plaintexts.get(input_index.clone());
        Ok(BggEncodingWire {
            vector: c_b.clone().apply_preimage(high) + input.vector.clone().apply_preimage(low),
            pubkey: self.public_key(artifacts),
            plaintext: Some(output_plaintext),
        })
    }

    /// Evaluates one shared public LUT helper family over every tall encoding row.
    pub fn tall_encoding(
        &self,
        input: &BggTallEncodingWire,
        c_b_rows: &Family<Mat>,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<BggTallEncodingWire, LweLookupCompileError> {
        let BggTallPlaintext::Diagonal(input_plaintexts) = &input.plaintext else {
            return Err(LweLookupCompileError::MissingPlaintext);
        };
        let expected_c_b_type = MatrixType {
            modulus: self.high_matrix_type.modulus.clone(),
            ring_dimension: self.high_matrix_type.ring_dimension.clone(),
            rows: IntExpr::constant(1),
            columns: self.high_matrix_type.rows.clone(),
        };
        if input.rows.count() != input_plaintexts.count() || input.rows.count() != c_b_rows.count()
        {
            return Err(LweLookupCompileError::SlotCountMismatch);
        }
        if !same_matrix_type(c_b_rows.element_type(), &expected_c_b_type) {
            return Err(LweLookupCompileError::MatrixTypeMismatch);
        }
        self.validate_artifact_wires(artifacts)?;

        let output_public_key = self.public_key(artifacts);
        let artifact_rows = artifacts.clone();
        let (rows, plaintexts) = parallel_zip(
            (input.rows.clone(), input_plaintexts.clone(), c_b_rows.clone()),
            move |_, (input_row, plaintext, c_b)| {
                let input_index = plaintext
                    .extract_coefficient_with_canonical_input_exclusive_upper(
                        0,
                        input.canonical_input_exclusive_upper.clone(),
                    );
                let low = artifact_rows.low_matrices.get(input_index.clone());
                let high = artifact_rows.high_matrices.get(input_index.clone());
                let output_plaintext = artifact_rows.output_plaintexts.get(input_index.clone());
                (c_b.apply_preimage(high) + input_row.apply_preimage(low), output_plaintext)
            },
        )?;
        Ok(BggTallEncodingWire {
            rows,
            pubkey: output_public_key,
            plaintext: BggTallPlaintext::Diagonal(plaintexts),
            canonical_input_exclusive_upper: self
                .table
                .canonical_output_exclusive_upper_for_modulus(&self.high_matrix_type.modulus),
        })
    }

    fn validate_artifacts(
        &self,
        artifacts: &LweLookupArtifacts,
    ) -> Result<(), LweLookupCompileError> {
        if artifacts.semantic_version != LWE_LOOKUP_ARTIFACT_SEMANTIC_VERSION {
            return Err(LweLookupCompileError::ArtifactSemanticVersion {
                expected: LWE_LOOKUP_ARTIFACT_SEMANTIC_VERSION,
                actual: artifacts.semantic_version,
            });
        }
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
            !same_matrix_type(artifacts.high_matrices.element_type(), &self.high_matrix_type) ||
            artifacts.output_plaintexts.count() != &count ||
            !same_matrix_type(
                artifacts.output_plaintexts.element_type(),
                &self.ring().matrix_type((1, 1)),
            )
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

fn lookup_compiler_for_circuit<P: Poly>(
    _parameters: &P::Params,
    circuit: &PolyCircuit<P>,
    identity: LweLookupIdentity,
    public_key_type: MatrixType,
    gadget_base: IntExpr,
    digit_count: IntExpr,
) -> Result<LweLookupCompiler, LweLookupCompileError> {
    let table = LweLookupTable::from_public_lut(circuit.lookup_table(identity.lookup).as_ref())?;
    lookup_compiler_for_table::<P>(
        _parameters,
        identity,
        table,
        public_key_type,
        gadget_base,
        digit_count,
    )
}

fn lookup_compiler_for_table<P: Poly>(
    _parameters: &P::Params,
    identity: LweLookupIdentity,
    table: LweLookupTable,
    public_key_type: MatrixType,
    gadget_base: IntExpr,
    digit_count: IntExpr,
) -> Result<LweLookupCompiler, LweLookupCompileError> {
    let public_columns = public_key_type.columns.clone();
    let high_rows = IntExpr::Mul(
        Box::new(public_key_type.rows.clone()),
        Box::new(IntExpr::Add(Box::new(digit_count.clone()), Box::new(IntExpr::constant(2)))),
    )
    .canonicalize();
    let low_matrix_type = MatrixType {
        rows: public_columns.clone(),
        columns: public_columns.clone(),
        ..public_key_type.clone()
    };
    let high_matrix_type =
        MatrixType { rows: high_rows, columns: public_columns, ..public_key_type.clone() };
    Ok(LweLookupCompiler {
        identity,
        table,
        public_key_type,
        low_matrix_type,
        high_matrix_type,
        gadget_base,
        digit_count,
    })
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LookupRegistryKey(usize);

fn cached_lookup_table<P: Poly>(
    cache: &mut BTreeMap<LookupRegistryKey, LweLookupTable>,
    circuit: &PolyCircuit<P>,
    lookup_id: usize,
) -> Result<LweLookupTable, LweLookupCompileError> {
    let table = circuit.lookup_table(lookup_id);
    let key = LookupRegistryKey(Arc::as_ptr(&table) as usize);
    if let Some(table) = cache.get(&key) {
        return Ok(table.clone());
    }
    let table = LweLookupTable::from_public_lut(table.as_ref())?;
    cache.insert(key, table.clone());
    Ok(table)
}

impl<P: Poly> LweLookupPreprocessingLowering<P> {
    pub fn new(
        parameters: P::Params,
        hash_key: Bytes,
        trapdoor: Trapdoor,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        call_path_prefix: Vec<usize>,
    ) -> Self {
        Self {
            parameters,
            hash_key,
            trapdoor,
            gadget_base,
            digit_count,
            call_path_prefix,
            registry_tables: BTreeMap::new(),
            tables: BTreeMap::new(),
            table_families: BTreeMap::new(),
            entries: Vec::new(),
        }
    }

    pub fn entries(&self) -> &[LweLookupPreprocessingEntry] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<LweLookupPreprocessingEntry> {
        self.entries
    }
}

impl<P: Poly> CircuitLoweringTypes for LweLookupPreprocessingLowering<P> {
    type Wire = BggPublicKeyWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupPreprocessingLowering<P> {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let mut identity = identity_for_gate(gate, lookup_id);
        let mut call_path = self.call_path_prefix.clone();
        call_path.extend(identity.call_path);
        identity.call_path = call_path;
        let table = cached_lookup_table(&mut self.registry_tables, circuit, lookup_id).map_err(
            |source| CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source },
        )?;
        let mut compiler = lookup_compiler_for_table::<P>(
            &self.parameters,
            identity,
            table,
            input.matrix.matrix_type().clone(),
            self.gadget_base.clone(),
            self.digit_count.clone(),
        )
        .map_err(|source| CircuitCompileError::LweLookup {
            gate: gate.local_gate().index(),
            source,
        })?;
        let table_commitment = compiler.table.commitment();
        match self.tables.get(&table_commitment) {
            Some(table) => compiler.table = table.clone(),
            None => {
                self.tables.insert(table_commitment, compiler.table.clone());
            }
        }
        let table_families = match self.table_families.get(&table_commitment) {
            Some(table_families) => table_families.clone(),
            None => {
                let table_families = compiler.table_families().map_err(|source| {
                    CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
                })?;
                self.table_families.insert(table_commitment, table_families.clone());
                table_families
            }
        };
        let wires = compiler
            .preprocess_with_table_families(
                self.hash_key.clone(),
                input,
                &self.trapdoor,
                &table_families,
            )
            .map_err(|source| CircuitCompileError::LweLookup {
                gate: gate.local_gate().index(),
                source,
            })?;
        let output =
            BggPublicKeyWire { matrix: wires.output_public_key.clone(), reveal_plaintext: true };
        self.entries.push(LweLookupPreprocessingEntry { compiler, wires });
        Ok(output)
    }
}

impl<P: Poly> NaiveLweLookupPreprocessingLowering<P> {
    pub fn new(
        parameters: P::Params,
        hash_key: Bytes,
        trapdoors: Vec<Trapdoor>,
        gadget_base: IntExpr,
        digit_count: IntExpr,
        call_path_prefix: Vec<usize>,
    ) -> Result<Self, LweLookupCompileError> {
        let slot_count = trapdoors.len();
        if slot_count == 0 {
            return Err(LweLookupCompileError::EmptySlotInvocations);
        }
        Ok(Self {
            parameters,
            hash_key,
            trapdoors,
            gadget_base,
            digit_count,
            slot_count,
            call_path_prefix,
            registry_tables: BTreeMap::new(),
            tables: BTreeMap::new(),
            table_families: BTreeMap::new(),
            entries: Vec::new(),
        })
    }

    pub fn into_entries(self) -> Vec<NaiveLweLookupPreprocessingEntry> {
        self.entries
    }
}

impl<P: Poly> CircuitLoweringTypes for NaiveLweLookupPreprocessingLowering<P> {
    type Wire = NaiveBggPublicKeyVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for NaiveLweLookupPreprocessingLowering<P> {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        if input.matrices.count() != &IntExpr::constant(self.slot_count) {
            return Err(CircuitCompileError::LweLookup {
                gate: gate.local_gate().index(),
                source: LweLookupCompileError::SlotCountMismatch,
            });
        }
        let mut base_identity = identity_for_gate(gate, lookup_id);
        let mut call_path = self.call_path_prefix.clone();
        call_path.extend(base_identity.call_path);
        base_identity.call_path = call_path;
        let mut compilers = Vec::with_capacity(self.slot_count);
        let mut wires = Vec::with_capacity(self.slot_count);
        let mut outputs = Vec::with_capacity(self.slot_count);
        for slot in 0..self.slot_count {
            let table = cached_lookup_table(&mut self.registry_tables, circuit, lookup_id)
                .map_err(|source| CircuitCompileError::LweLookup {
                    gate: gate.local_gate().index(),
                    source,
                })?;
            let mut compiler = lookup_compiler_for_table::<P>(
                &self.parameters,
                LweLookupIdentity { slot: Some(slot), ..base_identity.clone() },
                table,
                input.matrices.element_type().clone(),
                self.gadget_base.clone(),
                self.digit_count.clone(),
            )
            .map_err(|source| CircuitCompileError::LweLookup {
                gate: gate.local_gate().index(),
                source,
            })?;
            let table_commitment = compiler.table.commitment();
            match self.tables.get(&table_commitment) {
                Some(table) => compiler.table = table.clone(),
                None => {
                    self.tables.insert(table_commitment, compiler.table.clone());
                }
            }
            let table_families = match self.table_families.get(&table_commitment) {
                Some(table_families) => table_families.clone(),
                None => {
                    let table_families = compiler.table_families().map_err(|source| {
                        CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
                    })?;
                    self.table_families.insert(table_commitment, table_families.clone());
                    table_families
                }
            };
            let preprocessing = compiler
                .preprocess_with_table_families(
                    self.hash_key.clone(),
                    &BggPublicKeyWire {
                        matrix: input.matrices.get_static(slot),
                        reveal_plaintext: input.reveal_plaintext,
                    },
                    &self.trapdoors[slot],
                    &table_families,
                )
                .map_err(|source| CircuitCompileError::LweLookup {
                    gate: gate.local_gate().index(),
                    source,
                })?;
            outputs.push(preprocessing.output_public_key.clone());
            compilers.push(compiler);
            wires.push(preprocessing);
        }
        let matrices = Family::pack(outputs).map_err(|source| CircuitCompileError::LweLookup {
            gate: gate.local_gate().index(),
            source: source.into(),
        })?;
        self.entries.push(NaiveLweLookupPreprocessingEntry { compilers, wires });
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true })
    }
}

pub fn bind_lwe_lookup_invocations<P: Poly>(
    parameters: &P::Params,
    circuit: &PolyCircuit<P>,
    production: ProductionId,
    compilers: impl IntoIterator<Item = LweLookupCompiler>,
) -> Result<Vec<LweLookupInvocation>, LweLookupCompileError> {
    compilers
        .into_iter()
        .map(|compiler| {
            LweLookupInvocation::bind(
                compiler.clone(),
                LweLookupArtifacts::for_compiler(production.clone(), &compiler),
                parameters,
                circuit,
            )
        })
        .collect()
}

pub fn bind_naive_lwe_lookup_invocations<P: Poly>(
    parameters: &P::Params,
    circuit: &PolyCircuit<P>,
    production: ProductionId,
    public_key_type: MatrixType,
    gadget_base: IntExpr,
    digit_count: IntExpr,
    slot_count: usize,
    call_path_prefix: &[usize],
) -> Result<Vec<NaiveLweLookupInvocation>, LweLookupCompileError> {
    if slot_count == 0 {
        return Err(LweLookupCompileError::EmptySlotInvocations);
    }
    collect_lwe_lookup_identities_with_prefix(circuit, call_path_prefix)
        .map_err(|error| LweLookupCompileError::CircuitStructure(error.to_string()))?
        .into_iter()
        .map(|identity| {
            let slots = (0..slot_count)
                .map(|slot| {
                    let compiler = lookup_compiler_for_circuit(
                        parameters,
                        circuit,
                        LweLookupIdentity { slot: Some(slot), ..identity.clone() },
                        public_key_type.clone(),
                        gadget_base.clone(),
                        digit_count.clone(),
                    )?;
                    LweLookupInvocation::bind(
                        compiler.clone(),
                        LweLookupArtifacts::for_compiler(production.clone(), &compiler),
                        parameters,
                        circuit,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            NaiveLweLookupInvocation::new(slots)
        })
        .collect()
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

/// Public-LUT lowering for tall BGG+ encodings sharing one helper family.
#[derive(Clone)]
pub struct LweLookupTallEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    c_b_rows: Family<Mat>,
}

#[cfg(test)]
type TallLookupKernel = Subgraph<Vec<Family<Mat>>, (Family<Mat>, Family<Mat>)>;

#[cfg(test)]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TallLookupKernelKey {
    input_types: Vec<MatFamilyType>,
    canonical_input_exclusive_upper: Option<BigUint>,
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

impl LweLookupTallEncodingLowering {
    /// Creates a tall lowering from gate invocations and caller-supplied `C_B` rows.
    pub fn new(
        invocations: impl IntoIterator<Item = LweLookupInvocation>,
        c_b_rows: Family<Mat>,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)?, c_b_rows })
    }
}

#[cfg(test)]
fn tall_lookup_kernel_for(
    kernels: &mut BTreeMap<TallLookupKernelKey, TallLookupKernel>,
    input_families: &[Family<Mat>],
    canonical_input_exclusive_upper: Option<BigUint>,
    kernel_name: String,
) -> Result<TallLookupKernel, LweLookupCompileError> {
    if input_families.len() != 6 {
        return Err(LweLookupCompileError::TallKernelSchema);
    }
    let key = TallLookupKernelKey {
        input_types: input_families
            .iter()
            .map(|family| MatFamilyType {
                element: MatrixType {
                    modulus: family.element_type().modulus.canonicalize(),
                    ring_dimension: family.element_type().ring_dimension.canonicalize(),
                    rows: family.element_type().rows.canonicalize(),
                    columns: family.element_type().columns.canonicalize(),
                },
                shape: vec![family.count().canonicalize()],
            })
            .collect(),
        canonical_input_exclusive_upper,
    };
    if let Some(kernel) = kernels.get(&key) {
        return Ok(kernel.clone());
    }
    let schemas = key.input_types.clone();
    let canonical_upper = key.canonical_input_exclusive_upper.clone();
    let kernel = Subgraph::try_define(kernel_name, schemas, move |families: Vec<Family<Mat>>| {
        let [
            input_rows,
            input_plaintexts,
            c_b_rows,
            low_matrices,
            high_matrices,
            output_plaintexts,
        ] = families.as_slice()
        else {
            return Err(DslError::Schema);
        };
        let (rows, plaintexts) = Family::<Mat>::parallel_zip_many_with_broadcast_values(
            vec![input_rows.clone(), input_plaintexts.clone(), c_b_rows.clone()],
            vec![low_matrices.clone(), high_matrices.clone(), output_plaintexts.clone()],
            move |_, values, artifacts| {
                let [input_row, input_plaintext, c_b] = values.as_slice() else {
                    return Err(DslError::Schema);
                };
                let [low_matrices, high_matrices, output_plaintexts] = artifacts.as_slice() else {
                    return Err(DslError::Schema);
                };
                let input_index = input_plaintext
                    .clone()
                    .extract_coefficient_with_canonical_input_exclusive_upper(
                        0,
                        canonical_upper.clone(),
                    );
                let low = low_matrices.get(input_index.clone());
                let high = high_matrices.get(input_index.clone());
                let output_plaintext = output_plaintexts.get(input_index);
                Ok((c_b.clone() * high + input_row.clone() * low, output_plaintext))
            },
        )?;
        Ok((rows, plaintexts))
    })?;
    kernels.insert(key, kernel.clone());
    Ok(kernel)
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
        invocation
            .compiler
            .validate_layout()
            .and_then(|_| invocation.compiler.validate_artifacts(&invocation.artifacts))
            .map_err(|source| lookup_error(gate, source))?;
        Ok(BggPublicKeyWire {
            matrix: invocation.compiler.import_output_public_key(&invocation.artifacts),
            reveal_plaintext: true,
        })
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

impl CircuitLoweringTypes for LweLookupTallEncodingLowering {
    type Wire = BggTallEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> PublicLookupLowering<P> for LweLookupTallEncodingLowering {
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
            .tall_encoding(input, &self.c_b_rows, &artifacts)
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
    use crate::{
        BggEncodingCompiler, BggPublicKeyCompiler, BggTallEncodingCompiler,
        test_utils::{matrix_output, row},
    };
    use mxx_dsl::DslContext;
    use mxx_gadgets::circuit::{LutExpr, PolyCircuit, PublicLutProgram};
    use mxx_ir_core::{
        FrozenGraphScopeId, ParamEnv,
        artifact::{ArtifactConfidentiality, SpecHash},
        node::{GridInputMode, IntBinaryOp, MatrixBinaryOp, NodeKind},
        types::WireType,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        ExecutionConfig, RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend,
        execute, execute_with_config, transcript::SamplingMode,
    };
    use num_bigint::BigUint;
    use std::num::NonZeroUsize;

    fn identity_lut(length: u64) -> PublicLutProgram {
        PublicLutProgram::new(length, LutExpr::input()).expect("identity LUT")
    }

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
        assert_eq!(identity.low_matrix_tag_prefix(), b"mxx-bgg/lwe-lookup-low/v2:9:4:slot2:row:");
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
    fn v1_lookup_artifact_metadata_is_rejected() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: 9,
                occurrence: 0,
                lookup: 4,
                slot: Some(2),
            },
            LweLookupTable::new([(0, BigInt::from(1)), (1, BigInt::from(0))]).unwrap(),
        );
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let mut artifacts = LweLookupArtifacts::for_compiler(production_id, &lookup);
        artifacts.semantic_version = 1;

        assert!(matches!(
            lookup.import_artifacts(&artifacts),
            Err(LweLookupCompileError::ArtifactSemanticVersion { expected: 2, actual: 1 })
        ));
        let names = LweLookupArtifactNames::for_compiler(&lookup);
        assert!(names.output_public_key.starts_with("lwe_lookup_v2_"));
        assert!(names.low_matrices.starts_with("lwe_lookup_v2_"));
        assert!(names.high_matrices.starts_with("lwe_lookup_v2_"));
    }

    #[test]
    fn preprocessing_is_one_logical_table_length_parallel_producer() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digits = parameters.modulus_digits();
        let lookup = compiler(
            &parameters,
            LweLookupIdentity { call_path: vec![3], gate: 7, occurrence: 1, lookup: 2, slot: None },
            LweLookupTable::new([(2, BigInt::from(5)), (0, BigInt::from(4)), (1, BigInt::from(6))])
                .unwrap(),
        );
        let ring = lookup.ring();
        let trapdoor = ring.sample_trapdoor(
            1,
            5,
            lookup.gadget_base.clone(),
            lookup.digit_count.clone(),
            1_000_000,
        );
        let wires = lookup
            .preprocess(
                ring.bytes_input("hash-key", 32),
                &BggPublicKeyWire { matrix: ring.zero((1, digits)), reveal_plaintext: true },
                &trapdoor,
            )
            .unwrap();
        let names = LweLookupArtifactNames::for_compiler(&lookup);
        let built = lookup
            .export_preprocessing(
                DslContext::new("logical-lwe-lookup-preprocessing"),
                wires,
                &names,
            )
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let grids = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
            .collect::<Vec<_>>();
        assert_eq!(grids.len(), 1);
        let NodeKind::ParallelGrid(spec) = grids[0].kind() else { unreachable!() };
        assert_eq!(spec.shape, vec![IntExpr::constant(3)]);
        assert_eq!(grids[0].output_types().len(), 2);
        assert!(grids[0].output_types().iter().all(|output| {
            matches!(
                output,
                WireType::Family { element, shape }
                    if matches!(element.as_ref(), WireType::Preimage(_)) &&
                        shape == &vec![IntExpr::constant(3)]
            )
        }));

        for node in built.graph.scopes().values().flat_map(|scope| scope.nodes()) {
            assert!(!matches!(node.kind(), NodeKind::FamilyGetStatic { .. }));
            assert!(!matches!(node.kind(), NodeKind::Select { .. }));
            assert!(!matches!(
                node.kind(),
                NodeKind::IntBinary(IntBinaryOp::Divide | IntBinaryOp::Remainder)
            ));
            if matches!(node.kind(), NodeKind::FamilyPack { .. }) {
                let Some(output) = node.output_types().first() else {
                    panic!("family pack output")
                };
                assert!(match output {
                    WireType::Family { element, .. } => {
                        matches!(element.as_ref(), WireType::Int) ||
                            matches!(element.as_ref(), WireType::Matrix(matrix)
                                if matrix.rows == IntExpr::constant(1) &&
                                    matrix.columns == IntExpr::constant(1))
                    }
                    _ => false,
                });
            }
        }
        let dynamic_hashes = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), NodeKind::HashSample { tag_prefix, .. }
                    if tag_prefix.starts_with(b"mxx-bgg/lwe-lookup-low/v2:"))
            })
            .collect::<Vec<_>>();
        assert_eq!(dynamic_hashes.len(), 1);
        assert_eq!(dynamic_hashes[0].arguments().len(), 2);
        assert!(matches!(dynamic_hashes[0].arguments()[1].wire_type(), WireType::Int));
    }

    #[test]
    fn shuffled_preprocessing_rows_have_distinct_v2_hashes_and_satisfy_the_lwe_equation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digits = parameters.modulus_digits();
        let table =
            LweLookupTable::new([(2, BigInt::from(5)), (0, BigInt::from(4)), (1, BigInt::from(6))])
                .unwrap();
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: 5,
                occurrence: 0,
                lookup: 1,
                slot: None,
            },
            table.clone(),
        );
        let ring = lookup.ring();
        let trapdoor = ring.sample_trapdoor(
            1,
            5,
            lookup.gadget_base.clone(),
            lookup.digit_count.clone(),
            1_000_000,
        );
        let input_public_key =
            BggPublicKeyWire { matrix: ring.zero((1, digits)), reveal_plaintext: true };
        let wires = lookup
            .preprocess(ring.bytes_input("hash-key", 32), &input_public_key, &trapdoor)
            .unwrap();
        let outputs = Family::pack(
            table.entries.iter().map(|entry| Int::constant(entry.output.clone())).collect(),
        )
        .unwrap();
        let scalar_type = ring.matrix_type((1, 1));
        let gadget = ring.gadget(1, lookup.gadget_base.clone(), lookup.digit_count.clone());
        let public_b = trapdoor.public_matrix();
        let input_a = input_public_key.matrix;
        let output_a = wires.output_public_key.clone();
        let residuals = parallel_zip(
            (wires.low_matrices.clone(), wires.high_matrices.clone(), outputs),
            move |index, (low, high, output)| {
                let x = index
                    .as_int()
                    .add(Int::constant(0))
                    .lift_to_constant_polynomial(scalar_type.clone());
                let y = output.lift_to_constant_polynomial(scalar_type.clone());
                public_b.clone().apply_preimage(high) -
                    (output_a.clone() -
                        gadget.clone() * y -
                        (input_a.clone() - gadget.clone() * x).apply_preimage(low))
            },
        )
        .unwrap();
        let validated = DslContext::new("shuffled-logical-lwe-preprocessing")
            .preimage_family_output("low", wires.low_matrices)
            .unwrap()
            .family_output("residual", residuals)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let execute_once = |parameters: &DCRTPolyParams| {
            let mut backend = cpu_backend([parameters.clone()]);
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &validated,
                &mut backend,
                BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x6d; 32]))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig {
                    max_parallel_instances: NonZeroUsize::new(2).unwrap(),
                    ..ExecutionConfig::default()
                },
            )
            .unwrap();
            let low = {
                let RuntimeValue::Family(values) =
                    result.materialize_output("low", &backend, &mut store).unwrap()
                else {
                    panic!("low output must be a family")
                };
                values
                    .iter()
                    .map(|value| {
                        let RuntimeValue::Matrix(matrix) = value else {
                            panic!("low family member must be a matrix")
                        };
                        matrix.as_ref().clone()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(low.len(), 3);
            let RuntimeValue::Family(residuals) =
                result.materialize_output("residual", &backend, &mut store).unwrap()
            else {
                panic!("residual output must be a family")
            };
            let zero = DCRTPolyMatrix::zero(parameters, 1, parameters.modulus_digits());
            assert_eq!(residuals.len(), 3);
            assert!(residuals.iter().all(|value| {
                matches!(value, RuntimeValue::Matrix(matrix) if matrix.as_ref() == &zero)
            }));
            result.cleanup_staged(&mut store).unwrap();
            low
        };

        let first = execute_once(&parameters);
        let second = execute_once(&parameters);
        assert_eq!(first, second);
        assert_ne!(first[0], first[1]);
        assert_ne!(first[0], first[2]);
        assert_ne!(first[1], first[2]);
    }

    #[test]
    fn static_table_records_only_nonnegative_output_bounds() {
        let nonnegative =
            LweLookupTable::new([(0, BigInt::from(0)), (1, BigInt::from(9))]).unwrap();
        assert_eq!(nonnegative.canonical_output_exclusive_upper, Some(BigUint::from(10u8)));
        assert_eq!(
            nonnegative.canonical_output_exclusive_upper_for_modulus(&IntExpr::constant(9)),
            None,
            "a canonical output range must fit the matrix modulus"
        );
        assert_eq!(
            LweLookupTable::new([(0, BigInt::from(-1)), (1, BigInt::from(9))])
                .unwrap()
                .canonical_output_exclusive_upper,
            None
        );
    }

    #[test]
    fn tall_lookup_shares_one_helper_family_and_matches_every_runtime_row() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digits = parameters.modulus_digits();
        let slots = 4;
        let table = LweLookupTable::new([
            (0, BigInt::from(1)),
            (1, BigInt::from(1)),
            (2, BigInt::from(1)),
            (3, BigInt::from(0)),
        ])
        .unwrap();
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: 1,
                occurrence: 0,
                lookup: 0,
                slot: None,
            },
            table,
        );
        let ring = lookup.ring();
        let input = BggTallEncodingWire {
            rows: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("input-row-{slot}"), (1, digits)))
                    .collect(),
            )
            .unwrap(),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("input-public", (1, digits)),
                reveal_plaintext: true,
            },
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack(
                    (0..slots)
                        .map(|slot| ring.input(format!("input-plain-{slot}"), (1, 1)))
                        .collect(),
                )
                .unwrap(),
            ),
            canonical_input_exclusive_upper: Some(BigUint::from(4u8)),
        };
        let c_b_rows = Family::pack(
            (0..slots).map(|slot| ring.input(format!("c-b-{slot}"), (1, digits + 2))).collect(),
        )
        .unwrap();
        let low_values =
            (0..4).map(|_| DCRTPolyMatrix::zero(&parameters, digits, digits)).collect::<Vec<_>>();
        let names = LweLookupArtifactNames::for_compiler(&lookup);
        let low_inputs = (0..4)
            .map(|_| {
                ring.zero((1, digits))
                    .decompose(lookup.gadget_base.clone(), lookup.digit_count.clone())
                    .into_preimage_relation()
            })
            .collect::<Vec<_>>();
        let helper_trapdoor = ring.sample_trapdoor(
            1,
            5,
            lookup.gadget_base.clone(),
            lookup.digit_count.clone(),
            1_000_000,
        );
        let high_inputs = (0..4)
            .map(|_| helper_trapdoor.sample_preimage(ring.zero((1, digits)), (digits + 2, digits)))
            .collect::<Vec<_>>();
        let output_plaintext_inputs = (0..4)
            .map(|index| ring.input(format!("output-plaintext-{index}"), (1, 1)))
            .collect::<Vec<_>>();
        let producer_wires = LweLookupPreprocessingWires {
            output_public_key: ring.input("output-public", (1, digits)),
            low_matrices: Family::pack(low_inputs).unwrap(),
            high_matrices: Family::pack(high_inputs).unwrap(),
            output_plaintexts: Family::pack(output_plaintext_inputs).unwrap(),
        };
        let producer = lookup
            .export_preprocessing(DslContext::new("tall-lookup-helpers"), producer_wires, &names)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut producer_inputs = BTreeMap::from([(
            "output-public".to_owned(),
            RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, 1, digits)),
        )]);
        let output_values = [1usize, 1, 1, 0];
        for index in 0..4 {
            producer_inputs.insert(
                format!("output-plaintext-{index}"),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, output_values[index])],
                )),
            );
        }
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced =
            execute(&producer, &mut backend, producer_inputs, &mut store, SamplingMode::Fresh)
                .unwrap();
        let RuntimeValue::Family(high_outputs) = &produced.outputs[&names.high_matrices] else {
            panic!("high preimage output must be a matrix family")
        };
        let high_values = high_outputs
            .iter()
            .map(|value| {
                let RuntimeValue::Matrix(matrix) = value else {
                    panic!("high preimage family member must be a matrix")
                };
                matrix.as_ref().clone()
            })
            .collect::<Vec<_>>();
        let production_id = produced.production_id.expect("helper artifact production");
        let manifest = store.manifest(&production_id).unwrap().clone();
        let mut v1_manifest = manifest.clone();
        for v2_name in [
            &names.output_public_key,
            &names.low_matrices,
            &names.high_matrices,
            &names.output_plaintexts,
        ] {
            let artifact = v1_manifest.artifacts.remove(v2_name).expect("v2 manifest artifact");
            let v1_name = v2_name.replacen("lwe_lookup_v2_", "lwe_lookup_", 1);
            v1_manifest.artifacts.insert(v1_name, artifact);
        }
        let artifacts = lookup
            .import_artifacts(&LweLookupArtifacts::for_compiler(production_id.clone(), &lookup))
            .unwrap();
        let output = lookup.tall_encoding(&input, &c_b_rows, &artifacts).unwrap();
        assert_eq!(output.canonical_input_exclusive_upper, Some(BigUint::from(2u8)));
        let tall_compiler = BggTallEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                digit_count: IntExpr::constant(digits),
            },
        };
        let accumulated = tall_compiler.add(&output, &output).unwrap();
        assert_eq!(accumulated.canonical_input_exclusive_upper, Some(BigUint::from(3u8)));
        let final_output = lookup.tall_encoding(&accumulated, &c_b_rows, &artifacts).unwrap();
        let BggTallPlaintext::Diagonal(output_plaintexts) = output.plaintext else {
            panic!("public lookup must reveal its output")
        };
        let BggTallPlaintext::Diagonal(final_plaintexts) = final_output.plaintext else {
            panic!("the accumulated public lookup must reveal its output")
        };
        let mut context = DslContext::new("tall-lookup-runtime");
        for slot in 0..slots {
            context = context
                .output(format!("row-{slot}"), output.rows.get_static(slot))
                .unwrap()
                .output(format!("plain-{slot}"), output_plaintexts.get_static(slot))
                .unwrap()
                .output(format!("final-plain-{slot}"), final_plaintexts.get_static(slot))
                .unwrap();
        }
        let graph = context.build().unwrap();
        assert!(graph.graph.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
            matches!(
                node.kind(),
                NodeKind::ExtractCoefficient {
                    canonical_input_exclusive_upper: Some(upper),
                    ..
                } if upper == &BigUint::from(4u8)
            )
        }));
        assert!(graph.graph.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
            matches!(
                node.kind(),
                NodeKind::ExtractCoefficient {
                    canonical_input_exclusive_upper: Some(upper),
                    ..
                } if upper == &BigUint::from(3u8)
            )
        }));
        assert_eq!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic { .. }))
                .count(),
            6,
            "each of the two lookup stages uses one shared low/high/output helper selection"
        );
        assert_eq!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| {
                    matches!(node.kind(), NodeKind::Select { .. }) &&
                        matches!(node.output_types().first(), Some(WireType::Family { .. }))
                })
                .count(),
            0,
            "Tall lookup must not expose a chunk selector or switch"
        );
        assert_eq!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .filter(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::IntBinary(IntBinaryOp::Divide | IntBinaryOp::Remainder)
                    )
                })
                .count(),
            0,
            "Tall lookup must not derive a chunk or offset from x"
        );

        let indices = [0usize, 1, 1, 3];
        let input_rows =
            (0..slots).map(|slot| row(&parameters, digits, 2 + slot)).collect::<Vec<_>>();
        let c_b_values =
            (0..slots).map(|slot| row(&parameters, digits + 2, 7 + slot)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::new();
        for slot in 0..slots {
            inputs.insert(
                format!("input-row-{slot}"),
                RuntimeValue::matrix(input_rows[slot].clone()),
            );
            inputs.insert(
                format!("input-plain-{slot}"),
                RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, indices[slot])],
                )),
            );
            inputs.insert(format!("c-b-{slot}"), RuntimeValue::matrix(c_b_values[slot].clone()));
        }
        assert!(
            graph
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(production_id.clone(), v1_manifest)]),
                )
                .is_err(),
            "a v1 artifact namespace must not satisfy the v2 evaluator graph"
        );
        let graph = graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .unwrap();
        let result =
            execute(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh).unwrap();
        let outputs = [1usize, 1, 1, 0];
        for slot in 0..slots {
            let index = indices[slot];
            let expected = c_b_values[slot].clone() * high_values[index].clone() +
                input_rows[slot].clone() * low_values[index].clone();
            assert_eq!(matrix_output(&result, &format!("row-{slot}")), &expected);
            assert_eq!(
                matrix_output(&result, &format!("plain-{slot}")),
                &DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, outputs[index])],
                )
            );
            assert_eq!(
                matrix_output(&result, &format!("final-plain-{slot}")),
                &DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, 1)],
                )
            );
        }
    }

    #[test]
    fn tall_lookup_kernel_cache_keys_canonical_upper_and_reuses_definitions() {
        let ring = Ring::new(17, 8);
        let family = |prefix: &str| {
            Family::pack(
                (0..2).map(|index| ring.input(format!("{prefix}-{index}"), (1, 1))).collect(),
            )
            .unwrap()
        };
        let first_inputs = vec![
            family("row"),
            family("plaintext"),
            family("c-b"),
            family("low"),
            family("high"),
            family("output"),
        ];
        let second_inputs = vec![
            family("row-second"),
            family("plaintext-second"),
            family("c-b-second"),
            family("low-second"),
            family("high-second"),
            family("output-second"),
        ];
        let third_inputs = vec![
            family("row-third"),
            family("plaintext-third"),
            family("c-b-third"),
            family("low-third"),
            family("high-third"),
            family("output-third"),
        ];
        let mut kernels = BTreeMap::new();
        let first = tall_lookup_kernel_for(
            &mut kernels,
            &first_inputs,
            Some(BigUint::from(4u8)),
            "tall-lookup-kernel-test-0".to_owned(),
        )
        .unwrap();
        let second = tall_lookup_kernel_for(
            &mut kernels,
            &first_inputs,
            Some(BigUint::from(4u8)),
            "tall-lookup-kernel-test-0".to_owned(),
        )
        .unwrap();
        assert_eq!(kernels.len(), 1);
        let third = tall_lookup_kernel_for(
            &mut kernels,
            &first_inputs,
            Some(BigUint::from(5u8)),
            "tall-lookup-kernel-test-1".to_owned(),
        )
        .unwrap();
        assert_eq!(kernels.len(), 2);
        let (rows, plaintexts) = first.call(first_inputs).unwrap();
        let (second_rows, second_plaintexts) = second.call(second_inputs).unwrap();
        let (other_rows, other_plaintexts) = third.call(third_inputs).unwrap();
        let context = DslContext::new("tall-lookup-kernel-cache");
        let built = context
            .public_family_output("rows", rows)
            .unwrap()
            .public_family_output("plaintexts", plaintexts)
            .unwrap()
            .public_family_output("second-rows", second_rows)
            .unwrap()
            .public_family_output("second-plaintexts", second_plaintexts)
            .unwrap()
            .public_family_output("other-rows", other_rows)
            .unwrap()
            .public_family_output("other-plaintexts", other_plaintexts)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let root_calls = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::SubgraphCall(_)))
            .collect::<Vec<_>>();
        assert_eq!(root_calls.len(), 3);
        assert!(root_calls.iter().all(|node| node.arguments().len() == 6));
        assert_ne!(root_calls[0].arguments()[0], root_calls[1].arguments()[0]);
        assert_ne!(root_calls[1].arguments()[0], root_calls[2].arguments()[0]);
        let named_definitions = built
            .graph
            .scopes()
            .keys()
            .filter(|id| matches!(id, FrozenGraphScopeId::Subgraph { .. }))
            .count();
        assert_eq!(named_definitions, 2);
        let named_scopes = built
            .graph
            .scopes()
            .iter()
            .filter_map(|(id, scope)| {
                matches!(id, FrozenGraphScopeId::Subgraph { .. }).then_some(scope)
            })
            .collect::<Vec<_>>();
        assert!(named_scopes.iter().all(|scope| {
            scope.nodes().iter().any(|node| {
                matches!(
                    node.kind(),
                    NodeKind::ParallelGrid(spec)
                        if spec.input_modes.len() == 6 &&
                            spec.input_modes[..3]
                                .iter()
                                .all(|mode| matches!(mode, GridInputMode::Reindex { .. })) &&
                            spec.input_modes[3..]
                                .iter()
                                .all(|mode| matches!(mode, GridInputMode::Broadcast))
                )
            })
        }));
        let parallel_bodies = built
            .graph
            .scopes()
            .iter()
            .filter_map(|(id, scope)| {
                matches!(id, FrozenGraphScopeId::ParallelBody { .. }).then_some(scope)
            })
            .collect::<Vec<_>>();
        assert_eq!(parallel_bodies.len(), 2);
        let uppers = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter_map(|node| match node.kind() {
                NodeKind::ExtractCoefficient { canonical_input_exclusive_upper, .. } => {
                    canonical_input_exclusive_upper.clone()
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(uppers.contains(&BigUint::from(4u8)));
        assert!(uppers.contains(&BigUint::from(5u8)));
    }

    #[test]
    fn tall_lookup_kernel_rejects_non_six_family_schema() {
        let ring = Ring::new(17, 8);
        let family = Family::pack(vec![ring.input("schema-0", (1, 1))]).unwrap();
        let mut kernels = BTreeMap::new();
        assert!(matches!(
            tall_lookup_kernel_for(
                &mut kernels,
                &[family.clone(), family],
                None,
                "tall-lookup-kernel-invalid".to_owned(),
            ),
            Err(LweLookupCompileError::TallKernelSchema)
        ));
        assert!(kernels.is_empty());
    }

    #[test]
    fn tall_lookup_rejects_hidden_mismatched_and_wrong_width_inputs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digits = parameters.modulus_digits();
        let lookup = compiler(
            &parameters,
            LweLookupIdentity {
                call_path: Vec::new(),
                gate: 1,
                occurrence: 0,
                lookup: 0,
                slot: None,
            },
            LweLookupTable::new([(0, BigInt::from(1)), (1, BigInt::from(0))]).unwrap(),
        );
        let ring = lookup.ring();
        let rows = Family::pack(vec![ring.zero((1, digits)), ring.zero((1, digits))]).unwrap();
        let diagonal = Family::pack(vec![ring.zero((1, 1)), ring.zero((1, 1))]).unwrap();
        let artifacts = LweLookupArtifactWires {
            output_public_key: ring.zero((1, digits)),
            low_matrices: Family::pack(
                (0..2)
                    .map(|_| {
                        ring.zero((digits, digits))
                            .decompose(lookup.gadget_base.clone(), 1)
                            .into_preimage_relation()
                    })
                    .collect(),
            )
            .unwrap(),
            high_matrices: Family::pack(
                (0..2)
                    .map(|_| {
                        ring.zero((digits + 2, digits))
                            .decompose(lookup.gadget_base.clone(), 1)
                            .into_preimage_relation()
                    })
                    .collect(),
            )
            .unwrap(),
            output_plaintexts: Family::pack((0..2).map(|_| ring.zero((1, 1))).collect()).unwrap(),
        };
        let wire = |plaintext| BggTallEncodingWire {
            rows: rows.clone(),
            pubkey: BggPublicKeyWire { matrix: ring.zero((1, digits)), reveal_plaintext: true },
            plaintext,
            canonical_input_exclusive_upper: None,
        };
        assert!(matches!(
            lookup.tall_encoding(
                &wire(BggTallPlaintext::Hidden),
                &Family::pack(vec![ring.zero((1, digits + 2)), ring.zero((1, digits + 2)),])
                    .unwrap(),
                &artifacts,
            ),
            Err(LweLookupCompileError::MissingPlaintext)
        ));
        assert!(matches!(
            lookup.tall_encoding(
                &wire(BggTallPlaintext::Diagonal(diagonal.clone())),
                &Family::pack(vec![ring.zero((1, digits + 2))]).unwrap(),
                &artifacts,
            ),
            Err(LweLookupCompileError::SlotCountMismatch)
        ));
        assert!(matches!(
            lookup.tall_encoding(
                &wire(BggTallPlaintext::Diagonal(diagonal)),
                &Family::pack(vec![ring.zero((1, digits + 1)), ring.zero((1, digits + 1)),])
                    .unwrap(),
                &artifacts,
            ),
            Err(LweLookupCompileError::MatrixTypeMismatch)
        ));
    }

    #[test]
    fn lookup_identity_collection_matches_the_lowering_identity() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
        let output = circuit.public_lookup_gate(input, lookup_id).as_single_wire();
        circuit.output([output]);
        assert_eq!(
            collect_lwe_lookup_identities(&circuit).unwrap(),
            vec![LweLookupIdentity {
                call_path: Vec::new(),
                gate: output.index(),
                occurrence: 0,
                lookup: lookup_id,
                slot: None,
            }]
        );
    }

    #[test]
    fn lookup_identity_collection_does_not_materialize_compact_slot_transfers() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_identity_repeated_lanes_gate(
            input,
            500_000_000,
            vec![Some(0), None, Some(2), None],
        );
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
        let output = circuit.public_lookup_gate(transferred, lookup_id).as_single_wire();
        circuit.output([output]);

        assert_eq!(
            collect_lwe_lookup_identities(&circuit).unwrap(),
            vec![LweLookupIdentity {
                call_path: Vec::new(),
                gate: output.index(),
                occurrence: 0,
                lookup: lookup_id,
                slot: None,
            }]
        );
    }

    #[test]
    fn lookup_registry_conversion_cache_reuses_same_program_arc() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
        let mut cache = BTreeMap::new();
        let first = cached_lookup_table(&mut cache, &circuit, lookup_id).unwrap();
        let second = cached_lookup_table(&mut cache, &circuit, lookup_id).unwrap();
        assert_eq!(cache.len(), 1);
        let cached = cache.values().next().unwrap();
        assert!(Arc::ptr_eq(&first.entries, &cached.entries));
        assert!(Arc::ptr_eq(&second.entries, &cached.entries));
    }

    #[test]
    fn preprocessing_lowering_reuses_public_table_families_across_lookup_invocations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let first_lookup_id = circuit.register_public_lookup(identity_lut(2));
        let second_lookup_id = circuit.register_public_lookup(identity_lut(2));
        let first = circuit.public_lookup_gate(input, first_lookup_id);
        let second = circuit.public_lookup_gate(input, second_lookup_id);
        circuit.output([first, second]);
        let trapdoor = ring.sample_trapdoor(
            1,
            5,
            BigInt::from(1u64 << parameters.base_bits()),
            digit_count,
            1_000_000,
        );
        let mut lookup = LweLookupPreprocessingLowering::new(
            parameters.clone(),
            ring.bytes_input("hash-key", 32),
            trapdoor,
            BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count.into(),
            Vec::new(),
        );
        let mut slots = crate::NoSlotOperations::default();
        let public_key = |name: &str| BggPublicKeyWire {
            matrix: ring.input(name, (1, digit_count)),
            reveal_plaintext: true,
        };
        let outputs = crate::PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        }
        .compile_public_keys_with_lowerings(
            &circuit,
            public_key("one"),
            [public_key("input")],
            &mut lookup,
            &mut slots,
        )
        .unwrap();
        assert_eq!(lookup.entries().len(), 2);
        assert_eq!(
            lookup.entries().iter().map(|entry| entry.compiler.table_length()).sum::<usize>(),
            4
        );
        assert!(Arc::ptr_eq(
            &lookup.entries()[0].compiler.table.entries,
            &lookup.entries()[1].compiler.table.entries,
        ));
        assert_ne!(
            LweLookupArtifactNames::for_compiler(&lookup.entries()[0].compiler).output_public_key,
            LweLookupArtifactNames::for_compiler(&lookup.entries()[1].compiler).output_public_key,
        );
        let mut context = DslContext::new("shared-lwe-lookup-public-table-families");
        for (index, output) in outputs.into_iter().enumerate() {
            context = context.output(format!("output-{index}"), output.matrix).unwrap();
        }
        for entry in lookup.into_entries() {
            context = entry.export(context).unwrap();
        }
        let built = context.build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();

        let table_family_packs = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), NodeKind::FamilyPack { shape, .. }
                    if shape == &vec![IntExpr::constant(2)]) &&
                    matches!(
                        node.output_types().first(),
                        Some(WireType::Family { element, .. })
                            if matches!(element.as_ref(), WireType::Int) ||
                                matches!(element.as_ref(), WireType::Matrix(matrix)
                                    if matrix.rows == IntExpr::constant(1) &&
                                        matrix.columns == IntExpr::constant(1))
                    )
            })
            .count();
        assert_eq!(table_family_packs, 2, "each logical table remains a rank-N family pack");
    }

    #[test]
    fn naive_preprocessing_lowering_reuses_shared_trapdoors_and_namespaces_artifacts() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let ring = Ring::new(modulus, parameters.ring_dimension() as usize);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
        let output = circuit.public_lookup_gate(input, lookup_id);
        circuit.output([output]);
        let public_key = |name: &str| NaiveBggPublicKeyVecWire {
            matrices: ring.input_family(name, 2, (1, digit_count)),
            reveal_plaintext: true,
        };
        let trapdoor = ring.sample_trapdoor(
            1,
            5,
            BigInt::from(1u64 << parameters.base_bits()),
            digit_count,
            1_000_000,
        );
        let mut lookup = NaiveLweLookupPreprocessingLowering::new(
            parameters.clone(),
            ring.bytes_input("hash-key", 32),
            vec![trapdoor.clone(), trapdoor],
            BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count.into(),
            vec![17],
        )
        .unwrap();
        let mut slots = crate::NoSlotOperations::default();
        let outputs = crate::PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        }
        .compile_naive_public_keys_with_lowerings(
            &circuit,
            public_key("one"),
            [public_key("input")],
            &mut lookup,
            &mut slots,
        )
        .unwrap();
        let entries = lookup.into_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].compilers.len(), 2);
        assert!(
            entries[0].compilers.iter().all(|compiler| compiler.identity.call_path == vec![17])
        );
        let mut context = DslContext::new("naive-lookup-preprocessing")
            .family_output("output", outputs[0].matrices.clone())
            .unwrap();
        for entry in entries {
            context = entry.export(context).unwrap();
        }
        context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn circuit_public_lookup_lowers_to_logical_artifact_family_access() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
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
            LweLookupTable::from_public_lut(circuit.lookup_table(lookup_id).as_ref())
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
            4
        );
        for suffix in ["_low_matrices", "_high_matrices", "_output_plaintexts"] {
            assert!(
                nodes.iter().any(|node| {
                    let NodeKind::Input { artifact: Some(artifact), wire_type, .. } = node.kind()
                    else {
                        return false;
                    };
                    artifact.artifact_name.ends_with(suffix) &&
                        matches!(
                            wire_type,
                            WireType::Family { shape, .. }
                                if shape == &vec![IntExpr::constant(2)]
                        )
                }),
                "logical lookup artifact {suffix} must have table-length domain"
            );
        }
        let dynamic_gets = nodes
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic { .. }))
            .collect::<Vec<_>>();
        assert_eq!(dynamic_gets.len(), 3);
        assert!(dynamic_gets.iter().all(|node| {
            matches!(
                node.arguments()[1].node().kind(),
                NodeKind::ExtractCoefficient { position, .. }
                    if position == &IntExpr::constant(0)
            )
        }));
        assert!(
            dynamic_gets
                .iter()
                .skip(1)
                .all(|node| node.arguments()[1] == dynamic_gets[0].arguments()[1])
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| {
                    matches!(node.kind(), NodeKind::Select { .. }) &&
                        matches!(node.output_types().first(), Some(WireType::Family { .. }))
                })
                .count(),
            0,
            "lookup artifacts are one logical family; chunk selection is not checker-visible"
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::IntBinary(IntBinaryOp::Divide | IntBinaryOp::Remainder)
                    )
                })
                .count(),
            0,
            "lookup access must use x directly, without chunk arithmetic"
        );
    }

    #[test]
    fn single_public_lookup_keeps_the_output_signal_in_secret_public_key_gadget_order() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
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
            LweLookupTable::from_public_lut(circuit.lookup_table(lookup_id).as_ref())
                .expect("table"),
        );
        let production_id = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] };
        let output_public_key_artifact =
            LweLookupArtifactNames::for_compiler(&lookup).output_public_key;
        let invocation = LweLookupInvocation::bind(
            lookup.clone(),
            LweLookupArtifacts::for_compiler(production_id.clone(), &lookup),
            &parameters,
            &circuit,
        )
        .expect("invocation");
        let ring = lookup.ring();
        let circuit_compiler = crate::PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: lookup.gadget_base.clone(),
                digit_count: lookup.digit_count.clone(),
            },
        };
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, digit_count)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (1, digit_count)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let input_encoding = encoding("input");
        let mut lowering =
            LweLookupEncodingLowering::new([invocation], ring.input("c-b", (1, digit_count + 2)))
                .expect("lowering");
        let mut slots = crate::NoSlotOperations::default();
        let output = circuit_compiler
            .compile_encodings_with_lowerings(
                &circuit,
                encoding("one"),
                [input_encoding],
                &mut lowering,
                &mut slots,
            )
            .expect("lookup lowering")
            .into_iter()
            .next()
            .expect("one LUT output");
        let output_plaintext = output.plaintext.clone().expect("lookup output plaintext");
        let secret = ring.input("lookup-secret", (1, 1));
        let gadget = ring.gadget(1, lookup.gadget_base.clone(), lookup.digit_count.clone());
        let signal = secret.clone() * output.pubkey.matrix.clone() -
            output_plaintext.clone() * secret.clone() * gadget.clone();

        let NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) = signal.value_handle().node().kind()
        else {
            panic!("lookup signal must be a subtraction")
        };
        let [left, right] = signal.value_handle().node().arguments() else {
            panic!("lookup signal subtraction must have two operands")
        };
        let NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) = left.node().kind() else {
            panic!("lookup signal left side must be secret times output public key")
        };
        assert_eq!(
            left.node().arguments(),
            [secret.value_handle().clone(), output.pubkey.matrix.value_handle().clone()]
        );
        let NodeKind::Input { artifact: Some(artifact), .. } =
            output.pubkey.matrix.value_handle().node().kind()
        else {
            panic!("lookup output public key must be imported from its exact artifact")
        };
        assert_eq!(artifact.production_id, production_id);
        assert_eq!(artifact.artifact_name, output_public_key_artifact);
        assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);

        let NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) = right.node().kind() else {
            panic!("lookup signal right side must end with a gadget multiplication")
        };
        let [plaintext_times_secret, right_gadget] = right.node().arguments() else {
            panic!("lookup signal gadget product must have two operands")
        };
        assert_eq!(right_gadget, gadget.value_handle());
        let NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) = plaintext_times_secret.node().kind()
        else {
            panic!("lookup signal must multiply its plaintext by the secret before the gadget")
        };
        assert_eq!(
            plaintext_times_secret.node().arguments(),
            [output_plaintext.value_handle().clone(), secret.value_handle().clone()]
        );
        let NodeKind::FamilyGetDynamic { .. } = output_plaintext.value_handle().node().kind()
        else {
            panic!("lookup output plaintext must use dynamic access to the artifact family")
        };
        let [output_family, output_selector] = output_plaintext.value_handle().node().arguments()
        else {
            panic!("lookup output plaintext family access has the wrong arity")
        };
        let NodeKind::Input { artifact: Some(artifact), .. } = output_family.node().kind() else {
            panic!("lookup output plaintext must come from the authoritative artifact family")
        };
        assert_eq!(artifact.production_id, production_id);
        assert_eq!(
            artifact.artifact_name,
            LweLookupArtifactNames::for_compiler(&lookup).output_plaintexts
        );
        assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);
        let NodeKind::ExtractCoefficient { position, .. } = output_selector.node().kind() else {
            panic!("lookup output plaintext selector must be the input coefficient")
        };
        assert_eq!(position, &IntExpr::constant(0));

        DslContext::new("single-lwe-public-lookup-signal")
            .output("signal", signal)
            .expect("signal output")
            .build()
            .expect("signal graph");
    }

    #[test]
    fn naive_lookup_lowerings_build_structural_family_graphs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(identity_lut(2));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: output_gate.as_single_wire().index(),
            occurrence: 0,
            lookup: lookup_id,
            slot: None,
        };
        let table = LweLookupTable::from_public_lut(circuit.lookup_table(lookup_id).as_ref())
            .expect("table");
        let lookup = compiler(&parameters, identity.clone(), table.clone());
        let ring = lookup.ring();
        let public_key_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: lookup.gadget_base.clone(),
            digit_count: lookup.digit_count.clone(),
        };
        let circuit_compiler =
            crate::PolyCircuitCompiler { public_key: public_key_compiler.clone() };

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
            8
        );
    }
}
