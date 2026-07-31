use crate::{
    AdvancedGateLowering, BggEncodingWire, BggPolyEncodingWire, BggPublicKeyWire,
    CircuitCompileError, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
};
use mxx_gadgets::{
    Poly, PolyElem,
    circuit::{GateInstance, PolyCircuit, PublicLut},
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, OutputFamilyError, SubgraphBuildError,
    TrapdoorWire, WireRef,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConstantMatrix, HashVariant, LoopInputMode, MatrixBinaryOp},
    types::MatrixType,
};
use num_bigint::{BigInt, Sign};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

/// Stable structural identity of one LWE public-lookup invocation.
///
/// Top-level ordinary gates keep the historical tag spelling. Nested and
/// repeated invocations add their call path and occurrence so two distinct
/// Graph IR productions cannot accidentally reuse one auxiliary matrix.
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

/// Materialized, deterministic form of a BGG-independent [`PublicLut`].
///
/// Rows must be a permutation of `0..len`. This is the invariant assumed by
/// the historical row-indexed `K_high` storage layout.
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
        table: &PublicLut<P>,
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
    let length =
        u64::try_from(value.len()).expect("lookup commitment component length exceeds u64");
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupPreprocessingWire {
    pub output_public_key: MatrixWire,
    pub low_matrices: MatrixFamilyWire,
    pub high_matrices: MatrixFamilyWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LweLookupArtifactWires {
    pub output_public_key: MatrixWire,
    pub low_matrices: MatrixFamilyWire,
    pub high_matrices: MatrixFamilyWire,
}

#[derive(Clone, Debug)]
pub struct LweLookupInvocation {
    compiler: LweLookupCompiler,
    artifacts: LweLookupArtifacts,
    circuit_table_identity: usize,
}

/// Scalar public-key advanced lowering backed by per-gate LWE artifacts.
#[derive(Clone, Debug, Default)]
pub struct LweLookupPublicKeyLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
}

/// Scalar encoding advanced lowering backed by the same per-gate LWE
/// artifacts and the shared `C_B = sB` matrix.
#[derive(Clone, Debug)]
pub struct LweLookupEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    c_b: MatrixWire,
}

/// Polynomial-encoding advanced lowering that evaluates the same scalar LWE
/// lookup independently for every slot while sharing one artifact production.
#[derive(Clone, Debug)]
pub struct LweLookupPolyEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    c_b_by_slot: MatrixFamilyWire,
}

#[derive(Clone, Debug)]
pub struct NaiveLweLookupInvocation {
    identity: LweLookupIdentity,
    slots: Vec<LweLookupInvocation>,
    circuit_table_identity: usize,
}

#[derive(Clone, Debug, Default)]
pub struct NaiveLweLookupPublicKeyLowering {
    invocations: BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
}

#[derive(Clone, Debug)]
pub struct NaiveLweLookupEncodingLowering {
    invocations: BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
    c_b_by_slot: MatrixFamilyWire,
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

#[derive(Debug, Error, Eq, PartialEq)]
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
    #[error("LWE BGG+ encoding lookup requires a revealed input plaintext")]
    MissingPlaintext,
    #[error("LWE BGG+ poly-encoding lookup families must have matching slot counts")]
    SlotCountMismatch,
    #[error("lookup artifact table length {actual} does not match compiler length {expected}")]
    ArtifactLength { expected: usize, actual: usize },
    #[error("the materialized circuit lookup table differs from the preprocessing table")]
    CircuitTableMismatch,
    #[error("the advanced lowering was bound to a different circuit lookup registration")]
    CircuitTableBinding,
    #[error("lookup artifacts were produced for a different lookup table")]
    ArtifactTableCommitment,
    #[error("more than one LWE lookup invocation has the same structural identity")]
    DuplicateInvocation,
    #[error("a naive LWE lookup invocation requires at least one slot")]
    EmptySlotInvocations,
    #[error("naive LWE lookup slot {slot} has an inconsistent structural identity")]
    SlotIdentity { slot: usize },
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
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
        if artifacts.table_length != compiler.table.len() {
            return Err(LweLookupCompileError::ArtifactLength {
                expected: compiler.table.len(),
                actual: artifacts.table_length,
            });
        }
        if artifacts.table_commitment != compiler.table.commitment() {
            return Err(LweLookupCompileError::ArtifactTableCommitment);
        }
        Ok(Self {
            compiler,
            artifacts,
            circuit_table_identity: std::sync::Arc::as_ptr(&circuit_table) as usize,
        })
    }
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
        c_b: MatrixWire,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)?, c_b })
    }
}

impl LweLookupPolyEncodingLowering {
    pub fn new(
        invocations: impl IntoIterator<Item = LweLookupInvocation>,
        c_b_by_slot: MatrixFamilyWire,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_invocations(invocations)?, c_b_by_slot })
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
        c_b_by_slot: MatrixFamilyWire,
    ) -> Result<Self, LweLookupCompileError> {
        Ok(Self { invocations: collect_naive_invocations(invocations)?, c_b_by_slot })
    }
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

impl LweLookupCompiler {
    /// Builds the exact historical LWE auxiliary-matrix equation:
    ///
    /// `K_high = Preimage_B(A_lt - yG - (A_z - xG)K_low)`.
    ///
    /// `A_lt`, every `K_low`, and every `K_high` are exported from this one
    /// production and imported by evaluation. This preserves atom identity
    /// across graphs instead of merely recomputing numerically equal hashes.
    pub fn preprocess(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        input_public_key: &BggPublicKeyWire,
        trapdoor: &TrapdoorWire,
    ) -> Result<LweLookupPreprocessingWire, LweLookupCompileError> {
        let output_public_key = self.sample_output_public_key(builder, hash_key);
        let scalar_type = self.scalar_type();
        let mut input_scalars = Vec::with_capacity(self.table.len());
        let mut output_scalars = Vec::with_capacity(self.table.len());
        let mut low_matrices = Vec::with_capacity(self.table.len());
        for (input, entry) in self.table.entries.iter().enumerate() {
            input_scalars
                .push(builder.constant_polynomial(scalar_type.clone(), [BigInt::from(input)]));
            output_scalars
                .push(builder.constant_polynomial(scalar_type.clone(), [entry.output.clone()]));
            low_matrices.push(self.sample_low_matrix(builder, hash_key, entry.row));
        }
        let input_scalars = builder.family_pack(&input_scalars)?;
        let output_scalars = builder.family_pack(&output_scalars)?;
        let low_inputs = builder.family_pack(&low_matrices)?;

        let mut body = GraphBuilder::new(
            format!(
                "lwe-lookup-preimage-{}-{}-slot{}",
                self.identity.gate_token(),
                self.identity.lookup,
                self.identity.slot_index()
            ),
            Vec::new(),
        );
        let body_input_public_key = body.input("0_input_public_key", self.public_key_type.clone());
        let body_output_public_key =
            body.input("1_output_public_key", self.public_key_type.clone());
        let body_trapdoor = body.trapdoor_input(
            "2_trapdoor",
            trapdoor.public.matrix_type.clone(),
            trapdoor.sigma.clone(),
            trapdoor.gadget_base.clone(),
            trapdoor.digit_count.clone(),
        );
        let body_input_scalar = body.input("3_input_scalar", scalar_type.clone());
        let body_output_scalar = body.input("4_output_scalar", scalar_type);
        let body_low = body.input("5_low", self.low_matrix_type.clone());
        let gadget = body.constant_matrix(
            self.public_key_type.clone(),
            ConstantMatrix::Gadget { base: self.gadget_base.clone(), small: false },
        );
        let input_gadget = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &gadget,
            &body_input_scalar,
            self.public_key_type.clone(),
        );
        let extended_input = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &body_input_public_key,
            &input_gadget,
            self.public_key_type.clone(),
        );
        let output_gadget = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &gadget,
            &body_output_scalar,
            self.public_key_type.clone(),
        );
        let target = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &body_output_public_key,
            &output_gadget,
            self.public_key_type.clone(),
        );
        let low_term = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &extended_input,
            &body_low,
            self.public_key_type.clone(),
        );
        let adjusted_target = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &target,
            &low_term,
            self.public_key_type.clone(),
        );
        let high =
            body.preimage_sample(&body_trapdoor, &adjusted_target, self.high_matrix_type.clone());
        body.value_output_wire("0_low", body_low.wire);
        body.value_output_wire("1_high", high.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            IntExpr::constant(self.table.len()),
            "entry",
            Vec::new(),
            vec![
                input_public_key.matrix.wire,
                output_public_key.wire,
                trapdoor.wire,
                input_scalars.wire,
                output_scalars.wire,
                low_inputs.wire,
            ],
            vec![
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
            ],
            &[self.low_matrix_type.clone(), self.high_matrix_type.clone()],
        )?;
        let low_matrices = outputs.remove(0);
        let high_matrices = outputs.remove(0);
        Ok(LweLookupPreprocessingWire { output_public_key, low_matrices, high_matrices })
    }

    pub fn export_preprocessing(
        &self,
        builder: &mut GraphBuilder,
        wires: &LweLookupPreprocessingWire,
        names: &LweLookupArtifactNames,
    ) {
        builder.output(
            names.output_public_key.clone(),
            &wires.output_public_key,
            ArtifactConfidentiality::Public,
        );
        builder.output_family_wire(
            names.low_matrices.clone(),
            &wires.low_matrices,
            ArtifactConfidentiality::Public,
        );
        builder.output_family_wire(
            names.high_matrices.clone(),
            &wires.high_matrices,
            ArtifactConfidentiality::Public,
        );
    }

    pub fn import_artifacts(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &LweLookupArtifacts,
    ) -> Result<LweLookupArtifactWires, LweLookupCompileError> {
        if artifacts.table_length != self.table.len() {
            return Err(LweLookupCompileError::ArtifactLength {
                expected: self.table.len(),
                actual: artifacts.table_length,
            });
        }
        if artifacts.table_commitment != self.table.commitment() {
            return Err(LweLookupCompileError::ArtifactTableCommitment);
        }
        let names = LweLookupArtifactNames::for_compiler(self);
        let count = IntExpr::constant(self.table.len());
        Ok(LweLookupArtifactWires {
            output_public_key: builder.artifact_input(
                format!("{}_input", names.output_public_key),
                self.public_key_type.clone(),
                artifacts.production_id.clone(),
                names.output_public_key,
                ArtifactConfidentiality::Public,
            ),
            low_matrices: builder.artifact_family_input(
                format!("{}_input", names.low_matrices),
                self.low_matrix_type.clone(),
                artifacts.production_id.clone(),
                names.low_matrices,
                count.clone(),
                ArtifactConfidentiality::Public,
            ),
            high_matrices: builder.artifact_family_input(
                format!("{}_input", names.high_matrices),
                self.high_matrix_type.clone(),
                artifacts.production_id.clone(),
                names.high_matrices,
                count,
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn public_key(&self, artifacts: &LweLookupArtifactWires) -> BggPublicKeyWire {
        BggPublicKeyWire { matrix: artifacts.output_public_key.clone(), reveal_plaintext: true }
    }

    /// Evaluates the exact historical online equation:
    ///
    /// `c_out = C_B K_high[row(z)] + c_z K_low[row(z)]`.
    pub fn encoding(
        &self,
        builder: &mut GraphBuilder,
        input: &BggEncodingWire,
        c_b: &MatrixWire,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<BggEncodingWire, LweLookupCompileError> {
        let plaintext = input.plaintext.as_ref().ok_or(LweLookupCompileError::MissingPlaintext)?;
        let input_index = builder.extract_coefficient(plaintext, IntExpr::constant(0));
        let mut plaintext_branches = Vec::with_capacity(self.table.len());
        for entry in &self.table.entries {
            plaintext_branches
                .push(builder.constant_polynomial(self.scalar_type(), [entry.output.clone()]));
        }
        // Artifact families are stored in input order. The LUT row
        // permutation still determines each K_low hash tag in preprocessing,
        // while online evaluation performs one lazy load per large family.
        let low = builder.family_get_dynamic(&artifacts.low_matrices, input_index);
        let high = builder.family_get_dynamic(&artifacts.high_matrices, input_index);
        let output_plaintext = builder.select(input_index, &plaintext_branches);
        let high_term = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            c_b,
            &high,
            input.vector.matrix_type.clone(),
        );
        let low_term = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &input.vector,
            &low,
            input.vector.matrix_type.clone(),
        );
        let vector = builder.matrix_binary(
            MatrixBinaryOp::Add,
            &high_term,
            &low_term,
            input.vector.matrix_type.clone(),
        );
        Ok(BggEncodingWire {
            vector,
            pubkey: self.public_key(artifacts),
            plaintext: Some(output_plaintext),
        })
    }

    /// Evaluates the scalar LWE lookup formula independently for each
    /// polynomial-encoding slot. Artifact families remain lazy broadcast loop
    /// inputs, so each slot imports only its selected `K_low` and `K_high`.
    pub fn poly_encoding(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        c_b_by_slot: &MatrixFamilyWire,
        artifacts: &LweLookupArtifactWires,
    ) -> Result<BggPolyEncodingWire, LweLookupCompileError> {
        let plaintexts =
            input.plaintexts.as_ref().ok_or(LweLookupCompileError::MissingPlaintext)?;
        if input.vectors.count != plaintexts.count || input.vectors.count != c_b_by_slot.count {
            return Err(LweLookupCompileError::SlotCountMismatch);
        }

        let mut body = GraphBuilder::new(
            format!("lwe-poly-lookup-{}-{}", self.identity.gate_token(), self.identity.lookup),
            Vec::new(),
        );
        let body_vector = body.input("0_vector", input.vectors.matrix_type.clone());
        let body_plaintext = body.input("1_plaintext", plaintexts.matrix_type.clone());
        let body_c_b = body.input("2_c_b", c_b_by_slot.matrix_type.clone());
        let body_output_public_key =
            body.input("3_output_public_key", artifacts.output_public_key.matrix_type.clone());
        let body_low = body.family_input(
            "4_low_matrices",
            artifacts.low_matrices.matrix_type.clone(),
            artifacts.low_matrices.count.clone(),
        );
        let body_high = body.family_input(
            "5_high_matrices",
            artifacts.high_matrices.matrix_type.clone(),
            artifacts.high_matrices.count.clone(),
        );
        let output = self.encoding(
            &mut body,
            &BggEncodingWire {
                vector: body_vector,
                pubkey: BggPublicKeyWire {
                    matrix: body_output_public_key.clone(),
                    reveal_plaintext: true,
                },
                plaintext: Some(body_plaintext),
            },
            &body_c_b,
            &LweLookupArtifactWires {
                output_public_key: body_output_public_key,
                low_matrices: body_low,
                high_matrices: body_high,
            },
        )?;
        body.value_output_wire("0_vector", output.vector.wire);
        body.value_output_wire(
            "1_plaintext",
            output.plaintext.as_ref().expect("revealed lookup output").wire,
        );
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.vectors.count.clone(),
            "slot",
            Vec::new(),
            vec![
                input.vectors.wire,
                plaintexts.wire,
                c_b_by_slot.wire,
                artifacts.output_public_key.wire,
                artifacts.low_matrices.wire,
                artifacts.high_matrices.wire,
            ],
            vec![
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Zip,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
                LoopInputMode::Broadcast,
            ],
            &[output.vector.matrix_type, output.plaintext.expect("revealed output").matrix_type],
        )?;
        Ok(BggPolyEncodingWire {
            vectors: outputs.remove(0),
            pubkey: self.public_key(artifacts),
            plaintexts: Some(outputs.remove(0)),
        })
    }

    fn sample_output_public_key(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
    ) -> MatrixWire {
        builder.hash_sample(
            hash_key,
            self.public_key_type.clone(),
            HashVariant::Plain,
            self.identity.output_public_key_tag(),
            Vec::new(),
            None,
            None,
        )
    }

    fn sample_low_matrix(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        row: usize,
    ) -> MatrixWire {
        builder.hash_sample(
            hash_key,
            self.low_matrix_type.clone(),
            HashVariant::Decomposed,
            self.identity.low_matrix_tag(row),
            Vec::new(),
            Some(self.gadget_base.clone()),
            Some(self.digit_count.clone()),
        )
    }

    fn scalar_type(&self) -> MatrixType {
        MatrixType {
            modulus: self.public_key_type.modulus.clone(),
            ring_dimension: self.public_key_type.ring_dimension.clone(),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        }
    }
}

impl<P: Poly> AdvancedGateLowering<P, BggPublicKeyWire> for LweLookupPublicKeyLowering {
    fn slot_transfer(
        &mut self,
        _builder: &mut GraphBuilder,
        _input: &BggPublicKeyWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot transfer",
        })
    }

    fn slot_reduce(
        &mut self,
        _builder: &mut GraphBuilder,
        _inputs: &[BggPublicKeyWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot reduction",
        })
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        _input: &BggPublicKeyWire,
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        let identity = identity_for_gate(gate, lookup_id);
        let circuit_table = circuit.lookup_table(lookup_id);
        let invocation = invocation_for_gate(
            &self.invocations,
            &identity,
            std::sync::Arc::as_ptr(&circuit_table) as usize,
            gate,
        )?;
        let artifacts =
            invocation.compiler.import_artifacts(builder, &invocation.artifacts).map_err(
                |source| CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source },
            )?;
        Ok(invocation.compiler.public_key(&artifacts))
    }
}

impl<P: Poly> AdvancedGateLowering<P, BggEncodingWire> for LweLookupEncodingLowering {
    fn slot_transfer(
        &mut self,
        _builder: &mut GraphBuilder,
        _input: &BggEncodingWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggEncodingWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot transfer",
        })
    }

    fn slot_reduce(
        &mut self,
        _builder: &mut GraphBuilder,
        _inputs: &[BggEncodingWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggEncodingWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot reduction",
        })
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &BggEncodingWire,
        gate: GateInstance<'_>,
    ) -> Result<BggEncodingWire, CircuitCompileError> {
        let identity = identity_for_gate(gate, lookup_id);
        let circuit_table = circuit.lookup_table(lookup_id);
        let invocation = invocation_for_gate(
            &self.invocations,
            &identity,
            std::sync::Arc::as_ptr(&circuit_table) as usize,
            gate,
        )?;
        let artifacts =
            invocation.compiler.import_artifacts(builder, &invocation.artifacts).map_err(
                |source| CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source },
            )?;
        invocation.compiler.encoding(builder, input, &self.c_b, &artifacts).map_err(|source| {
            CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
        })
    }
}

impl<P: Poly> AdvancedGateLowering<P, BggPolyEncodingWire> for LweLookupPolyEncodingLowering {
    fn slot_transfer(
        &mut self,
        _builder: &mut GraphBuilder,
        _input: &BggPolyEncodingWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot transfer",
        })
    }

    fn slot_reduce(
        &mut self,
        _builder: &mut GraphBuilder,
        _inputs: &[BggPolyEncodingWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot reduction",
        })
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &BggPolyEncodingWire,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        let identity = identity_for_gate(gate, lookup_id);
        let circuit_table = circuit.lookup_table(lookup_id);
        let invocation = invocation_for_gate(
            &self.invocations,
            &identity,
            std::sync::Arc::as_ptr(&circuit_table) as usize,
            gate,
        )?;
        let artifacts =
            invocation.compiler.import_artifacts(builder, &invocation.artifacts).map_err(
                |source| CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source },
            )?;
        invocation.compiler.poly_encoding(builder, input, &self.c_b_by_slot, &artifacts).map_err(
            |source| CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source },
        )
    }
}

impl<P: Poly> AdvancedGateLowering<P, NaiveBggPublicKeyVecWire>
    for NaiveLweLookupPublicKeyLowering
{
    fn slot_transfer(
        &mut self,
        _builder: &mut GraphBuilder,
        _input: &NaiveBggPublicKeyVecWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot transfer",
        })
    }

    fn slot_reduce(
        &mut self,
        _builder: &mut GraphBuilder,
        _inputs: &[NaiveBggPublicKeyVecWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot reduction",
        })
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &NaiveBggPublicKeyVecWire,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        let identity = identity_for_gate(gate, lookup_id);
        let circuit_table = circuit.lookup_table(lookup_id);
        let invocation = naive_invocation_for_gate(
            &self.invocations,
            &identity,
            std::sync::Arc::as_ptr(&circuit_table) as usize,
            gate,
        )?;
        require_slot_count(&input.matrices.count, invocation.slots.len(), gate)?;
        let matrices = invocation
            .slots
            .iter()
            .map(|slot| {
                slot.compiler
                    .import_artifacts(builder, &slot.artifacts)
                    .map(|artifacts| artifacts.output_public_key)
                    .map_err(|source| CircuitCompileError::LweLookup {
                        gate: gate.local_gate().index(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NaiveBggPublicKeyVecWire {
            matrices: pack_lookup_family(builder, &matrices, gate)?,
            reveal_plaintext: true,
        })
    }
}

impl<P: Poly> AdvancedGateLowering<P, NaiveBggEncodingVecWire> for NaiveLweLookupEncodingLowering {
    fn slot_transfer(
        &mut self,
        _builder: &mut GraphBuilder,
        _input: &NaiveBggEncodingVecWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot transfer",
        })
    }

    fn slot_reduce(
        &mut self,
        _builder: &mut GraphBuilder,
        _inputs: &[NaiveBggEncodingVecWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "slot reduction",
        })
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &NaiveBggEncodingVecWire,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        let identity = identity_for_gate(gate, lookup_id);
        let circuit_table = circuit.lookup_table(lookup_id);
        let invocation = naive_invocation_for_gate(
            &self.invocations,
            &identity,
            std::sync::Arc::as_ptr(&circuit_table) as usize,
            gate,
        )?;
        let plaintexts =
            input.plaintexts.as_ref().ok_or_else(|| CircuitCompileError::LweLookup {
                gate: gate.local_gate().index(),
                source: LweLookupCompileError::MissingPlaintext,
            })?;
        for count in
            [&input.vectors.count, &input.pubkeys.count, &plaintexts.count, &self.c_b_by_slot.count]
        {
            require_slot_count(count, invocation.slots.len(), gate)?;
        }

        let mut vectors = Vec::with_capacity(invocation.slots.len());
        let mut output_pubkeys = Vec::with_capacity(invocation.slots.len());
        let mut output_plaintexts = Vec::with_capacity(invocation.slots.len());
        for (slot_index, slot) in invocation.slots.iter().enumerate() {
            let index = IntExpr::constant(slot_index);
            let input_vector = builder.family_get_static(&input.vectors, index.clone());
            let input_pubkey = builder.family_get_static(&input.pubkeys, index.clone());
            let input_plaintext = builder.family_get_static(plaintexts, index.clone());
            let c_b = builder.family_get_static(&self.c_b_by_slot, index);
            let artifacts =
                slot.compiler.import_artifacts(builder, &slot.artifacts).map_err(|source| {
                    CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
                })?;
            let output = slot
                .compiler
                .encoding(
                    builder,
                    &BggEncodingWire {
                        vector: input_vector,
                        pubkey: BggPublicKeyWire {
                            matrix: input_pubkey,
                            reveal_plaintext: input.pubkey_reveal_plaintext,
                        },
                        plaintext: Some(input_plaintext),
                    },
                    &c_b,
                    &artifacts,
                )
                .map_err(|source| CircuitCompileError::LweLookup {
                    gate: gate.local_gate().index(),
                    source,
                })?;
            vectors.push(output.vector);
            output_pubkeys.push(output.pubkey.matrix);
            output_plaintexts.push(output.plaintext.expect("revealed lookup output"));
        }
        Ok(NaiveBggEncodingVecWire {
            vectors: pack_lookup_family(builder, &vectors, gate)?,
            pubkeys: pack_lookup_family(builder, &output_pubkeys, gate)?,
            pubkey_reveal_plaintext: true,
            plaintexts: Some(pack_lookup_family(builder, &output_plaintexts, gate)?),
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

fn invocation_for_gate<'a>(
    invocations: &'a BTreeMap<LweLookupIdentity, LweLookupInvocation>,
    identity: &LweLookupIdentity,
    circuit_table_identity: usize,
    gate: GateInstance<'_>,
) -> Result<&'a LweLookupInvocation, CircuitCompileError> {
    let invocation = invocations.get(identity).ok_or(CircuitCompileError::MissingGateContext {
        gate: gate.local_gate().index(),
        kind: "LWE lookup artifacts",
    })?;
    if invocation.circuit_table_identity != circuit_table_identity {
        return Err(CircuitCompileError::LweLookup {
            gate: gate.local_gate().index(),
            source: LweLookupCompileError::CircuitTableBinding,
        });
    }
    Ok(invocation)
}

fn naive_invocation_for_gate<'a>(
    invocations: &'a BTreeMap<LweLookupIdentity, NaiveLweLookupInvocation>,
    identity: &LweLookupIdentity,
    circuit_table_identity: usize,
    gate: GateInstance<'_>,
) -> Result<&'a NaiveLweLookupInvocation, CircuitCompileError> {
    let invocation = invocations.get(identity).ok_or(CircuitCompileError::MissingGateContext {
        gate: gate.local_gate().index(),
        kind: "naive LWE lookup artifacts",
    })?;
    if invocation.circuit_table_identity != circuit_table_identity {
        return Err(CircuitCompileError::LweLookup {
            gate: gate.local_gate().index(),
            source: LweLookupCompileError::CircuitTableBinding,
        });
    }
    Ok(invocation)
}

fn require_slot_count(
    count: &IntExpr,
    expected: usize,
    gate: GateInstance<'_>,
) -> Result<(), CircuitCompileError> {
    if count != &IntExpr::constant(expected) {
        return Err(CircuitCompileError::LweLookup {
            gate: gate.local_gate().index(),
            source: LweLookupCompileError::SlotCountMismatch,
        });
    }
    Ok(())
}

fn pack_lookup_family(
    builder: &mut GraphBuilder,
    values: &[MatrixWire],
    gate: GateInstance<'_>,
) -> Result<MatrixFamilyWire, CircuitCompileError> {
    builder.family_pack(values).map_err(LweLookupCompileError::from).map_err(|source| {
        CircuitCompileError::LweLookup { gate: gate.local_gate().index(), source }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{ArtifactType, ProductionId, SpecHash},
        node::NodeKind,
        validate,
        validate::validate_with_manifests,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
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
    use std::collections::BTreeMap;

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn input_family(
        builder: &mut GraphBuilder,
        prefix: &str,
        matrix_type: &MatrixType,
        count: usize,
    ) -> MatrixFamilyWire {
        let members = (0..count)
            .map(|slot| builder.input(format!("{prefix}_{slot}"), matrix_type.clone()))
            .collect::<Vec<_>>();
        builder.family_pack(&members).expect("homogeneous input family")
    }

    #[test]
    fn top_level_tags_keep_the_legacy_spelling() {
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 9,
            occurrence: 0,
            lookup: 4,
            slot: Some(2),
        };
        assert_eq!(identity.output_public_key_tag(), b"A_LT_9_slot2");
        assert_eq!(identity.low_matrix_tag(3), b"LWE_R_G_9_4_3_slot2");
    }

    #[test]
    fn table_rows_must_form_the_historical_dense_permutation() {
        assert_eq!(
            LweLookupTable::new([(0, BigInt::from(1)), (0, BigInt::from(2))]),
            Err(LweLookupCompileError::DuplicateRow(0))
        );
        assert_eq!(
            LweLookupTable::new([(0, BigInt::from(1)), (2, BigInt::from(2))]),
            Err(LweLookupCompileError::RowOutOfRange { row: 2, length: 2 })
        );
    }

    #[test]
    fn preprocessing_artifacts_drive_the_online_lwe_equation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let rows = 1usize;
        let digit_count = parameters.modulus_digits();
        let gadget_columns = rows * digit_count;
        let trapdoor_columns = rows * (digit_count + 2);
        let public_key_type = matrix_type(&parameters, rows, gadget_columns);
        let compiler = LweLookupCompiler {
            identity: LweLookupIdentity {
                call_path: Vec::new(),
                gate: 7,
                occurrence: 0,
                lookup: 3,
                slot: None,
            },
            table: LweLookupTable::new([(1, BigInt::from(11)), (0, BigInt::from(13))])
                .expect("permutation table"),
            public_key_type: public_key_type.clone(),
            low_matrix_type: matrix_type(&parameters, gadget_columns, gadget_columns),
            high_matrix_type: matrix_type(&parameters, trapdoor_columns, gadget_columns),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let names = LweLookupArtifactNames::for_compiler(&compiler);

        let mut producer = GraphBuilder::new("lwe-lookup-producer", Vec::new());
        let hash_key = producer.bytes_input("hash_key", 32);
        let input_public_key = BggPublicKeyWire {
            matrix: producer.input("input_public_key", public_key_type.clone()),
            reveal_plaintext: true,
        };
        let trapdoor = producer.trapdoor_sample(
            matrix_type(&parameters, rows, trapdoor_columns),
            mxx_ir_core::RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            compiler.gadget_base.clone(),
            compiler.digit_count.clone(),
        );
        let wires = compiler
            .preprocess(&mut producer, hash_key, &input_public_key, &trapdoor)
            .expect("preprocessing graph");
        compiler.export_preprocessing(&mut producer, &wires, &names);
        let producer_graph = producer.finish();
        assert_eq!(
            producer_graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::ParallelLoop(_)))
                .count(),
            1
        );
        let producer = validate(&producer_graph, &ParamEnv::default()).expect("validated producer");

        let mut producer_inputs = BTreeMap::new();
        producer_inputs.insert("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]));
        producer_inputs.insert(
            "input_public_key".to_owned(),
            RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, rows, gadget_columns)),
        );
        let mut store = MemoryArtifactStore::default();
        let mut producer_backend = cpu_backend([parameters.clone()]);
        let produced = execute(
            &producer,
            &mut producer_backend,
            producer_inputs,
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("preprocessing execution");
        assert_eq!(producer_backend.preimage_batch_calls(), 1);
        let production_id = produced.production_id.clone().expect("artifact production");
        let manifest = store.manifest(&production_id).expect("manifest").clone();
        assert_eq!(
            manifest.artifacts[&names.low_matrices].family_count,
            Some(compiler.table.len())
        );
        assert_eq!(
            manifest.artifacts[&names.high_matrices].family_count,
            Some(compiler.table.len())
        );
        let selected_low_handle = &produced.artifact_handles[&names.low_matrices][1];
        let selected_low_descriptor = &manifest.artifacts[&names.low_matrices];
        let ArtifactPayload::Matrix(selected_low_bytes) = store
            .load(&selected_low_handle.key, selected_low_descriptor)
            .expect("selected low matrix payload")
        else {
            panic!("selected low matrix payload");
        };
        let ArtifactType::Matrix(selected_low_type) = &selected_low_handle.artifact_type else {
            panic!("selected low matrix type");
        };
        let selected_low = producer_backend
            .matrix_from_bytes(selected_low_type, &selected_low_bytes)
            .expect("selected low matrix");

        let artifacts = LweLookupArtifacts::for_compiler(production_id.clone(), &compiler);
        let vector_type = matrix_type(&parameters, 1, gadget_columns);
        let scalar_type = matrix_type(&parameters, 1, 1);
        let mut consumer = GraphBuilder::new("lwe-lookup-consumer", Vec::new());
        let imported =
            compiler.import_artifacts(&mut consumer, &artifacts).expect("artifact imports");
        let input = BggEncodingWire {
            vector: consumer.input("input_vector", vector_type.clone()),
            pubkey: BggPublicKeyWire {
                matrix: consumer.input("input_public_key", public_key_type),
                reveal_plaintext: true,
            },
            plaintext: Some(consumer.input("input_plaintext", scalar_type)),
        };
        let c_b = consumer.input("c_b", matrix_type(&parameters, 1, trapdoor_columns));
        let output =
            compiler.encoding(&mut consumer, &input, &c_b, &imported).expect("online graph");
        consumer.value_output_wire("vector", output.vector.wire);
        consumer.value_output_wire(
            "plaintext",
            output.plaintext.as_ref().expect("revealed output").wire,
        );
        let consumer = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(production_id, manifest)]),
        )
        .expect("validated consumer");

        let input_vector = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            (0..gadget_columns)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        &parameters,
                        index % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        );
        let expected_vector = input_vector.clone() * &selected_low;
        let mut consumer_inputs = BTreeMap::new();
        consumer_inputs.insert("input_vector".to_owned(), RuntimeValue::matrix(input_vector));
        consumer_inputs.insert(
            "input_public_key".to_owned(),
            RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, rows, gadget_columns)),
        );
        consumer_inputs.insert(
            "input_plaintext".to_owned(),
            RuntimeValue::matrix(DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_usize_to_constant(&parameters, 1)],
            )),
        );
        consumer_inputs.insert(
            "c_b".to_owned(),
            RuntimeValue::matrix(DCRTPolyMatrix::zero(&parameters, 1, trapdoor_columns)),
        );
        let mut consumer_backend = cpu_backend([parameters.clone()]);
        let consumed = execute(
            &consumer,
            &mut consumer_backend,
            consumer_inputs,
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("online execution");
        let RuntimeValue::Matrix(actual_vector) = &consumed.outputs["vector"] else {
            panic!("vector output");
        };
        assert_eq!(actual_vector.as_ref(), &expected_vector);
        let RuntimeValue::Matrix(actual_plaintext) = &consumed.outputs["plaintext"] else {
            panic!("plaintext output");
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
    fn poly_circuit_advanced_lowering_uses_lwe_artifact_imports() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let public_key_type = matrix_type(&parameters, 1, digit_count);
        let vector_type = matrix_type(&parameters, 1, digit_count);
        let scalar_type = matrix_type(&parameters, 1, 1);
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 2,
            occurrence: 0,
            lookup: 0,
            slot: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        assert_eq!(lookup_id, identity.lookup);
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        assert_eq!(output_gate.as_single_wire().index(), identity.gate);
        circuit.output([output_gate]);
        let lookup_compiler = LweLookupCompiler {
            identity: identity.clone(),
            table: LweLookupTable::new([(0, BigInt::from(0)), (1, BigInt::from(1))])
                .expect("identity rows"),
            public_key_type: public_key_type.clone(),
            low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let artifacts = LweLookupArtifacts::for_compiler(
            ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            &lookup_compiler,
        );
        let invocation =
            LweLookupInvocation::bind(lookup_compiler, artifacts, &parameters, &circuit)
                .expect("bound lookup invocation");

        let mut builder = GraphBuilder::new("lwe-advanced-lowering", Vec::new());
        let one = BggEncodingWire {
            vector: builder.input("one_vector", vector_type.clone()),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("one_public_key", public_key_type.clone()),
                reveal_plaintext: true,
            },
            plaintext: Some(builder.input("one_plaintext", scalar_type.clone())),
        };
        let input = BggEncodingWire {
            vector: builder.input("input_vector", vector_type),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("input_public_key", public_key_type),
                reveal_plaintext: true,
            },
            plaintext: Some(builder.input("input_plaintext", scalar_type)),
        };
        let c_b = builder.input("c_b", matrix_type(&parameters, 1, digit_count + 2));
        let mut lowering =
            LweLookupEncodingLowering::new([invocation], c_b).expect("lookup lowering");
        let outputs = crate::PolyCircuitCompiler {
            public_key: crate::BggPublicKeyCompiler {
                base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                decomposed_type: matrix_type(&parameters, digit_count, digit_count),
            },
        }
        .compile_encodings_with_lowering(&mut builder, &circuit, one, [input], &mut lowering)
        .expect("advanced lookup lowering");
        assert_eq!(outputs.len(), 1);
        let graph = builder.finish();
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| { matches!(node.kind, NodeKind::Input { artifact: Some(_), .. }) })
        );
        assert_eq!(
            graph.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Select { .. })).count(),
            1
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetDynamic))
                .count(),
            2
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetStatic { .. }))
                .count(),
            0
        );
    }

    #[test]
    fn poly_encoding_lowering_uses_one_lazy_lookup_pair_per_slot() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let public_key_type = matrix_type(&parameters, 1, digit_count);
        let vector_type = matrix_type(&parameters, 1, digit_count);
        let scalar_type = matrix_type(&parameters, 1, 1);
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 2,
            occurrence: 0,
            lookup: 0,
            slot: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let lookup_compiler = LweLookupCompiler {
            identity,
            table: LweLookupTable::from_public_lut(
                &parameters,
                circuit.lookup_table(lookup_id).as_ref(),
            )
            .expect("circuit lookup table"),
            public_key_type: public_key_type.clone(),
            low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let artifacts = LweLookupArtifacts::for_compiler(
            ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] },
            &lookup_compiler,
        );
        let invocation =
            LweLookupInvocation::bind(lookup_compiler, artifacts, &parameters, &circuit)
                .expect("bound lookup invocation");

        let mut builder = GraphBuilder::new("lwe-poly-advanced-lowering", Vec::new());
        let one = BggPolyEncodingWire {
            vectors: input_family(&mut builder, "one_vector", &vector_type, 2),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("one_public_key", public_key_type.clone()),
                reveal_plaintext: true,
            },
            plaintexts: Some(input_family(&mut builder, "one_plaintext", &scalar_type, 2)),
        };
        let input = BggPolyEncodingWire {
            vectors: input_family(&mut builder, "input_vector", &vector_type, 2),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("input_public_key", public_key_type),
                reveal_plaintext: true,
            },
            plaintexts: Some(input_family(&mut builder, "input_plaintext", &scalar_type, 2)),
        };
        let c_b_by_slot =
            input_family(&mut builder, "c_b", &matrix_type(&parameters, 1, digit_count + 2), 2);
        let mut lowering = LweLookupPolyEncodingLowering::new([invocation], c_b_by_slot)
            .expect("poly lookup lowering");
        let outputs = crate::PolyCircuitCompiler {
            public_key: crate::BggPublicKeyCompiler {
                base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                decomposed_type: matrix_type(&parameters, digit_count, digit_count),
            },
        }
        .compile_poly_encodings_with_lowering(&mut builder, &circuit, one, [input], &mut lowering)
        .expect("advanced poly lookup lowering");
        assert_eq!(outputs.len(), 1);
        let graph = builder.finish();
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            3
        );
        let lookup_body = graph
            .subgraphs
            .values()
            .find(|body| body.name.starts_with("lwe-poly-lookup-"))
            .expect("slotwise lookup loop");
        assert_eq!(
            lookup_body
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetDynamic))
                .count(),
            2
        );
        assert_eq!(
            lookup_body
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetStatic { .. }))
                .count(),
            0
        );
        assert_eq!(
            lookup_body
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Select { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn naive_encoding_lowering_uses_slot_qualified_lookup_artifacts() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let public_key_type = matrix_type(&parameters, 1, digit_count);
        let vector_type = matrix_type(&parameters, 1, digit_count);
        let scalar_type = matrix_type(&parameters, 1, 1);
        let base_identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 2,
            occurrence: 0,
            lookup: 0,
            slot: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let table =
            LweLookupTable::from_public_lut(&parameters, circuit.lookup_table(lookup_id).as_ref())
                .expect("circuit lookup table");
        let slots = (0..2)
            .map(|slot| {
                let compiler = LweLookupCompiler {
                    identity: LweLookupIdentity { slot: Some(slot), ..base_identity.clone() },
                    table: table.clone(),
                    public_key_type: public_key_type.clone(),
                    low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
                    high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
                    gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                    digit_count: IntExpr::constant(digit_count),
                };
                let artifacts = LweLookupArtifacts::for_compiler(
                    ProductionId {
                        spec_hash: SpecHash([10 + slot as u8; 32]),
                        execution_nonce: [20 + slot as u8; 32],
                    },
                    &compiler,
                );
                LweLookupInvocation::bind(compiler, artifacts, &parameters, &circuit)
                    .expect("bound slot invocation")
            })
            .collect::<Vec<_>>();
        let invocation =
            NaiveLweLookupInvocation::new(slots).expect("naive slot invocation bundle");
        let circuit_compiler = crate::PolyCircuitCompiler {
            public_key: crate::BggPublicKeyCompiler {
                base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                decomposed_type: matrix_type(&parameters, digit_count, digit_count),
            },
        };

        let mut public_key_builder =
            GraphBuilder::new("naive-lwe-public-key-advanced-lowering", Vec::new());
        let public_key_one = NaiveBggPublicKeyVecWire {
            matrices: input_family(&mut public_key_builder, "one_public_key", &public_key_type, 2),
            reveal_plaintext: true,
        };
        let public_key_input = NaiveBggPublicKeyVecWire {
            matrices: input_family(
                &mut public_key_builder,
                "input_public_key",
                &public_key_type,
                2,
            ),
            reveal_plaintext: true,
        };
        let mut public_key_lowering = NaiveLweLookupPublicKeyLowering::new([invocation.clone()])
            .expect("naive public-key lookup lowering");
        let public_key_outputs = circuit_compiler
            .compile_naive_public_keys_with_lowering(
                &mut public_key_builder,
                &circuit,
                public_key_one,
                [public_key_input],
                &mut public_key_lowering,
            )
            .expect("advanced naive public-key lookup lowering");
        assert_eq!(public_key_outputs.len(), 1);
        assert_eq!(public_key_outputs[0].matrices.count, IntExpr::constant(2));
        let public_key_graph = public_key_builder.finish();
        assert_eq!(
            public_key_graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            6
        );

        let mut builder = GraphBuilder::new("naive-lwe-advanced-lowering", Vec::new());
        let one = NaiveBggEncodingVecWire {
            vectors: input_family(&mut builder, "one_vector", &vector_type, 2),
            pubkeys: input_family(&mut builder, "one_pubkey", &public_key_type, 2),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(input_family(&mut builder, "one_plaintext", &scalar_type, 2)),
        };
        let input = NaiveBggEncodingVecWire {
            vectors: input_family(&mut builder, "input_vector", &vector_type, 2),
            pubkeys: input_family(&mut builder, "input_pubkey", &public_key_type, 2),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(input_family(&mut builder, "input_plaintext", &scalar_type, 2)),
        };
        let c_b_by_slot =
            input_family(&mut builder, "c_b", &matrix_type(&parameters, 1, digit_count + 2), 2);
        let mut lowering = NaiveLweLookupEncodingLowering::new([invocation], c_b_by_slot)
            .expect("naive lookup lowering");
        let outputs = circuit_compiler
            .compile_naive_encodings_with_lowering(
                &mut builder,
                &circuit,
                one,
                [input],
                &mut lowering,
            )
            .expect("advanced naive lookup lowering");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].vectors.count, IntExpr::constant(2));
        assert_eq!(outputs[0].pubkeys.count, IntExpr::constant(2));
        assert_eq!(
            outputs[0].plaintexts.as_ref().expect("revealed outputs").count,
            IntExpr::constant(2)
        );
        let graph = builder.finish();
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            6
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, NodeKind::FamilyGetDynamic))
                .count(),
            4
        );
    }

    #[test]
    fn invocation_binding_rejects_a_same_length_different_lookup_table() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 2,
            occurrence: 0,
            lookup: 0,
            slot: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let compiler = LweLookupCompiler {
            identity: identity.clone(),
            table: LweLookupTable::new([(0, BigInt::from(1)), (1, BigInt::from(0))])
                .expect("different lookup table"),
            public_key_type: matrix_type(&parameters, 1, digit_count),
            low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let artifacts = LweLookupArtifacts::for_compiler(
            ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            &compiler,
        );
        let result = LweLookupInvocation::bind(compiler, artifacts, &parameters, &circuit);
        assert_eq!(result.unwrap_err(), LweLookupCompileError::CircuitTableMismatch);
    }

    #[test]
    fn invocation_binding_rejects_stale_same_length_artifacts() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: 2,
            occurrence: 0,
            lookup: 0,
            slot: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
        circuit.output([output_gate]);
        let compiler_for = |table| LweLookupCompiler {
            identity: identity.clone(),
            table,
            public_key_type: matrix_type(&parameters, 1, digit_count),
            low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let current = compiler_for(
            LweLookupTable::new([(0, BigInt::from(0)), (1, BigInt::from(1))])
                .expect("current lookup table"),
        );
        let stale = compiler_for(
            LweLookupTable::new([(0, BigInt::from(1)), (1, BigInt::from(0))])
                .expect("stale lookup table"),
        );
        let stale_artifacts = LweLookupArtifacts::for_compiler(
            ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            &stale,
        );
        let mut builder = GraphBuilder::new("stale-lwe-artifact-import", Vec::new());
        assert_eq!(
            current.import_artifacts(&mut builder, &stale_artifacts).unwrap_err(),
            LweLookupCompileError::ArtifactTableCommitment
        );
        let result = LweLookupInvocation::bind(current, stale_artifacts, &parameters, &circuit);
        assert_eq!(result.unwrap_err(), LweLookupCompileError::ArtifactTableCommitment);
    }
}
