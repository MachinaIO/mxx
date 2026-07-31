use crate::BggSlotTransferGateRequest;
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, RealExpr, SubgraphBuildError,
    TrapdoorWire, WireRef,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{
        ConcatAxis, ConstantMatrix, HashVariant, IndexRange, LoopInputMode, MatrixBinaryOp,
        SampleRange,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use std::collections::BTreeMap;
use thiserror::Error;

const B0_PUBLIC: &str = "slot_transfer_b0_public";
const B0_TRAPDOOR: &str = "slot_transfer_b0_trapdoor";
const B1_PUBLIC: &str = "slot_transfer_b1_public";
const B1_TRAPDOOR: &str = "slot_transfer_b1_trapdoor";
const SLOT_SECRET: &str = "slot_transfer_slot_secret";
const SLOT_PUBLIC_KEY: &str = "slot_transfer_slot_a";

#[derive(Clone, Debug)]
pub struct BggSlotTransferArtifactCompiler {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_size: usize,
    pub slot_count: usize,
    pub digit_count: usize,
    pub chunk_columns: usize,
    pub gadget_base: IntExpr,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SlotTransferArtifactError {
    #[error("slot-transfer dimensions, slot count, and chunk width must be nonzero")]
    EmptyLayout,
    #[error("slot-transfer gate request is incompatible with the artifact layout")]
    InvalidGateRequest,
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferBaseWires {
    pub b0: TrapdoorWire,
    pub b1: TrapdoorWire,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferBaseArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferSlotWires {
    pub secrets: MatrixFamilyWire,
    pub public_keys: MatrixFamilyWire,
    pub b0_preimage_chunks: Vec<MatrixFamilyWire>,
    pub b1_preimage_chunks: Vec<MatrixFamilyWire>,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferPublicSlotWires {
    pub public_keys: MatrixFamilyWire,
    pub b0_preimage_chunks: Vec<MatrixFamilyWire>,
    pub b1_preimage_chunks: Vec<MatrixFamilyWire>,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferSlotArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferGateWires {
    pub preimage_chunks: BTreeMap<String, MatrixFamilyWire>,
}

#[derive(Clone, Debug)]
pub struct BggSlotTransferGateArtifacts {
    pub production_id: ProductionId,
}

impl BggSlotTransferArtifactCompiler {
    pub fn validate_layout(&self) -> Result<(), SlotTransferArtifactError> {
        if self.secret_size == 0 ||
            self.slot_count == 0 ||
            self.digit_count == 0 ||
            self.chunk_columns == 0
        {
            return Err(SlotTransferArtifactError::EmptyLayout);
        }
        Ok(())
    }

    pub fn public_key_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.gadget_columns())
    }

    pub fn build_base(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<BggSlotTransferBaseWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        Ok(BggSlotTransferBaseWires {
            b0: builder.trapdoor_sample(
                self.b0_public_type(),
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                IntExpr::constant(self.digit_count),
            ),
            b1: builder.trapdoor_sample(
                self.b1_public_type(),
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                IntExpr::constant(self.digit_count),
            ),
        })
    }

    pub fn export_base(&self, builder: &mut GraphBuilder, base: &BggSlotTransferBaseWires) {
        builder.output(B0_PUBLIC, &base.b0.public, ArtifactConfidentiality::Public);
        builder.output_wire(B0_TRAPDOOR, base.b0.wire, ArtifactConfidentiality::Private);
        builder.output(B1_PUBLIC, &base.b1.public, ArtifactConfidentiality::Public);
        builder.output_wire(B1_TRAPDOOR, base.b1.wire, ArtifactConfidentiality::Private);
    }

    pub fn import_base(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &BggSlotTransferBaseArtifacts,
    ) -> Result<BggSlotTransferBaseWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        Ok(BggSlotTransferBaseWires {
            b0: builder.artifact_trapdoor_input(
                "slot_transfer_b0_trapdoor_input",
                self.b0_public_type(),
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                IntExpr::constant(self.digit_count),
                artifacts.production_id.clone(),
                B0_TRAPDOOR,
                ArtifactConfidentiality::Private,
            ),
            b1: builder.artifact_trapdoor_input(
                "slot_transfer_b1_trapdoor_input",
                self.b1_public_type(),
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                IntExpr::constant(self.digit_count),
                artifacts.production_id.clone(),
                B1_TRAPDOOR,
                ArtifactConfidentiality::Private,
            ),
        })
    }

    pub fn build_slots(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        base: &BggSlotTransferBaseWires,
    ) -> Result<BggSlotTransferSlotWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        let mut body = GraphBuilder::new(
            format!(
                "bgg-slot-transfer-slots-d{}-k{}-chunk{}",
                self.secret_size, self.digit_count, self.chunk_columns
            ),
            Vec::new(),
        );
        let body_hash_key = body.bytes_input("0_hash_key", 32);
        let body_b0 = body.trapdoor_input(
            "1_b0",
            self.b0_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let body_b1 = body.trapdoor_input(
            "2_b1",
            self.b1_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let secret = body.uniform_sample(
            self.slot_secret_type(),
            SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
        );
        let slot_public_key = body.hash_sample_with_encoded_tags(
            body_hash_key,
            self.public_key_type(),
            HashVariant::Plain,
            b"slot_transfer_slot_a_".to_vec(),
            Vec::new(),
            vec![IntExpr::Var("slot".to_owned())],
            Vec::new(),
            None,
            None,
        );
        let identity = body.constant_matrix(self.slot_secret_type(), ConstantMatrix::Identity);
        let secret_identity = body.concat(
            ConcatAxis::Columns,
            &[secret.clone(), identity],
            self.matrix_type(self.secret_size, self.secret_size * 2),
        );
        let gadget = body.constant_matrix(
            self.public_key_type(),
            ConstantMatrix::Gadget { base: self.gadget_base.clone(), small: false },
        );

        body.value_output_wire("00000000_secret", secret.wire);
        body.value_output_wire("00000001_public_key", slot_public_key.wire);
        let mut output_types = vec![self.slot_secret_type(), self.public_key_type()];
        for (chunk, columns) in self.chunks(self.b1_public_columns()).into_iter().enumerate() {
            let b1_chunk = body.slice(
                &body_b1.public,
                None,
                Some(columns),
                self.matrix_type(self.secret_size * 2, columns.end - columns.start),
            );
            let target = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &secret_identity,
                &b1_chunk,
                self.matrix_type(self.secret_size, columns.end - columns.start),
            );
            let target = self.add_error(&mut body, target);
            let preimage = body.preimage_sample(
                &body_b0,
                &target,
                self.matrix_type(self.b0_public_columns(), columns.end - columns.start),
            );
            body.value_output_wire(format!("{:08}_b0_chunk_{chunk}", 2 + chunk), preimage.wire);
            output_types.push(preimage.matrix_type);
        }
        let b0_chunk_count = self.chunk_count(self.b1_public_columns());
        for (chunk, columns) in self.chunks(self.gadget_columns()).into_iter().enumerate() {
            let a_chunk = body.slice(
                &slot_public_key,
                None,
                Some(columns),
                self.matrix_type(self.secret_size, columns.end - columns.start),
            );
            let gadget_chunk = body.slice(
                &gadget,
                None,
                Some(columns),
                self.matrix_type(self.secret_size, columns.end - columns.start),
            );
            let secret_gadget = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &secret,
                &gadget_chunk,
                self.matrix_type(self.secret_size, columns.end - columns.start),
            );
            let negative = body.matrix_negate(&secret_gadget);
            let target = body.concat(
                ConcatAxis::Rows,
                &[a_chunk, negative],
                self.matrix_type(self.secret_size * 2, columns.end - columns.start),
            );
            let target = self.add_error(&mut body, target);
            let preimage = body.preimage_sample(
                &body_b1,
                &target,
                self.matrix_type(self.b1_public_columns(), columns.end - columns.start),
            );
            body.value_output_wire(
                format!("{:08}_b1_chunk_{chunk}", 2 + b0_chunk_count + chunk),
                preimage.wire,
            );
            output_types.push(preimage.matrix_type);
        }

        let mut outputs = builder.parallel_loop(
            body.finish(),
            IntExpr::constant(self.slot_count),
            "slot",
            Vec::new(),
            vec![hash_key, base.b0.wire, base.b1.wire],
            vec![LoopInputMode::Broadcast; 3],
            &output_types,
        )?;
        let secrets = outputs.remove(0);
        let public_keys = outputs.remove(0);
        let b0_preimage_chunks = outputs.drain(..b0_chunk_count).collect();
        Ok(BggSlotTransferSlotWires {
            secrets,
            public_keys,
            b0_preimage_chunks,
            b1_preimage_chunks: outputs,
        })
    }

    pub fn export_slots(&self, builder: &mut GraphBuilder, slots: &BggSlotTransferSlotWires) {
        builder.output_family_wire(SLOT_SECRET, &slots.secrets, ArtifactConfidentiality::Private);
        builder.output_family_wire(
            SLOT_PUBLIC_KEY,
            &slots.public_keys,
            ArtifactConfidentiality::Public,
        );
        for (chunk, family) in slots.b0_preimage_chunks.iter().enumerate() {
            builder.output_family_wire(
                b0_preimage_name(chunk),
                family,
                ArtifactConfidentiality::Public,
            );
        }
        for (chunk, family) in slots.b1_preimage_chunks.iter().enumerate() {
            builder.output_family_wire(
                b1_preimage_name(chunk),
                family,
                ArtifactConfidentiality::Public,
            );
        }
    }

    pub fn import_slots(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &BggSlotTransferSlotArtifacts,
    ) -> Result<BggSlotTransferSlotWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        let count = IntExpr::constant(self.slot_count);
        let family = |builder: &mut GraphBuilder,
                      input: String,
                      artifact: String,
                      matrix_type: MatrixType,
                      confidentiality| {
            builder.artifact_family_input(
                input,
                matrix_type,
                artifacts.production_id.clone(),
                artifact,
                count.clone(),
                confidentiality,
            )
        };
        Ok(BggSlotTransferSlotWires {
            secrets: family(
                builder,
                "slot_transfer_slot_secret_input".to_owned(),
                SLOT_SECRET.to_owned(),
                self.slot_secret_type(),
                ArtifactConfidentiality::Private,
            ),
            public_keys: family(
                builder,
                "slot_transfer_slot_a_input".to_owned(),
                SLOT_PUBLIC_KEY.to_owned(),
                self.public_key_type(),
                ArtifactConfidentiality::Public,
            ),
            b0_preimage_chunks: self
                .chunks(self.b1_public_columns())
                .into_iter()
                .enumerate()
                .map(|(chunk, columns)| {
                    family(
                        builder,
                        format!("slot_transfer_b0_preimage_chunk_{chunk}_input"),
                        b0_preimage_name(chunk),
                        self.matrix_type(self.b0_public_columns(), columns.end - columns.start),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect(),
            b1_preimage_chunks: self
                .chunks(self.gadget_columns())
                .into_iter()
                .enumerate()
                .map(|(chunk, columns)| {
                    family(
                        builder,
                        format!("slot_transfer_b1_preimage_chunk_{chunk}_input"),
                        b1_preimage_name(chunk),
                        self.matrix_type(self.b1_public_columns(), columns.end - columns.start),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect(),
        })
    }

    pub fn import_slots_public(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &BggSlotTransferSlotArtifacts,
    ) -> Result<BggSlotTransferPublicSlotWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        let family = |builder: &mut GraphBuilder,
                      input_name: String,
                      artifact_name: String,
                      matrix_type: MatrixType| {
            builder.artifact_family_input(
                input_name,
                matrix_type,
                artifacts.production_id.clone(),
                artifact_name,
                IntExpr::constant(self.slot_count),
                ArtifactConfidentiality::Public,
            )
        };
        Ok(BggSlotTransferPublicSlotWires {
            public_keys: family(
                builder,
                "slot_transfer_slot_public_key_input".to_owned(),
                SLOT_PUBLIC_KEY.to_owned(),
                self.public_key_type(),
            ),
            b0_preimage_chunks: self
                .chunks(self.b1_public_columns())
                .into_iter()
                .enumerate()
                .map(|(chunk, columns)| {
                    family(
                        builder,
                        format!("slot_transfer_b0_preimage_chunk_{chunk}_input"),
                        b0_preimage_name(chunk),
                        self.matrix_type(self.b0_public_columns(), columns.end - columns.start),
                    )
                })
                .collect(),
            b1_preimage_chunks: self
                .chunks(self.gadget_columns())
                .into_iter()
                .enumerate()
                .map(|(chunk, columns)| {
                    family(
                        builder,
                        format!("slot_transfer_b1_preimage_chunk_{chunk}_input"),
                        b1_preimage_name(chunk),
                        self.matrix_type(self.b1_public_columns(), columns.end - columns.start),
                    )
                })
                .collect(),
        })
    }

    pub fn build_gate_preimages(
        &self,
        builder: &mut GraphBuilder,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        requests: &[BggSlotTransferGateRequest],
    ) -> Result<BggSlotTransferGateWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        let mut preimage_chunks = BTreeMap::new();
        for request in requests {
            self.validate_gate_request(request)?;
            match request {
                BggSlotTransferGateRequest::Transfer {
                    identity,
                    input_public_key,
                    output_public_key,
                    source_slots,
                } => {
                    for (chunk, columns) in
                        self.chunks(self.gadget_columns()).into_iter().enumerate()
                    {
                        let family = self.build_transfer_gate_chunk(
                            builder,
                            base,
                            slots,
                            input_public_key,
                            output_public_key,
                            source_slots,
                            identity,
                            columns,
                        )?;
                        preimage_chunks.insert(gate_preimage_name(false, identity, chunk), family);
                    }
                }
                BggSlotTransferGateRequest::Reduce {
                    identity,
                    input_public_keys,
                    output_public_key,
                    source_slot_count,
                } => {
                    for (chunk, columns) in
                        self.chunks(self.gadget_columns()).into_iter().enumerate()
                    {
                        let family = self.build_reduce_gate_chunk(
                            builder,
                            base,
                            slots,
                            input_public_keys,
                            output_public_key,
                            *source_slot_count,
                            identity,
                            columns,
                        )?;
                        preimage_chunks.insert(gate_preimage_name(true, identity, chunk), family);
                    }
                }
            }
        }
        Ok(BggSlotTransferGateWires { preimage_chunks })
    }

    pub fn export_gate_preimages(
        &self,
        builder: &mut GraphBuilder,
        gates: &BggSlotTransferGateWires,
    ) {
        for (name, family) in &gates.preimage_chunks {
            builder.output_family_wire(name, family, ArtifactConfidentiality::Public);
        }
    }

    pub fn import_gate_preimages(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &BggSlotTransferGateArtifacts,
        requests: &[BggSlotTransferGateRequest],
    ) -> Result<BggSlotTransferGateWires, SlotTransferArtifactError> {
        self.validate_layout()?;
        let mut preimage_chunks = BTreeMap::new();
        for request in requests {
            self.validate_gate_request(request)?;
            let (reduction, identity, count) = match request {
                BggSlotTransferGateRequest::Transfer { identity, source_slots, .. } => {
                    (false, identity, source_slots.len())
                }
                BggSlotTransferGateRequest::Reduce { identity, input_public_keys, .. } => {
                    (true, identity, input_public_keys.len())
                }
            };
            for (chunk, columns) in self.chunks(self.gadget_columns()).into_iter().enumerate() {
                let name = gate_preimage_name(reduction, identity, chunk);
                let family = builder.artifact_family_input(
                    format!("{name}_input"),
                    self.matrix_type(self.b0_public_columns(), columns.end - columns.start),
                    artifacts.production_id.clone(),
                    name.clone(),
                    IntExpr::constant(count),
                    ArtifactConfidentiality::Public,
                );
                preimage_chunks.insert(name, family);
            }
        }
        Ok(BggSlotTransferGateWires { preimage_chunks })
    }

    fn validate_gate_request(
        &self,
        request: &BggSlotTransferGateRequest,
    ) -> Result<(), SlotTransferArtifactError> {
        let valid = match request {
            BggSlotTransferGateRequest::Transfer {
                input_public_key,
                output_public_key,
                source_slots,
                ..
            } => {
                input_public_key.matrix_type == self.public_key_type() &&
                    output_public_key.matrix_type == self.public_key_type() &&
                    source_slots.len() <= self.slot_count &&
                    source_slots.iter().all(|(source, _)| (*source as usize) < self.slot_count)
            }
            BggSlotTransferGateRequest::Reduce {
                input_public_keys,
                output_public_key,
                source_slot_count,
                ..
            } => {
                !input_public_keys.is_empty() &&
                    input_public_keys.len() <= *source_slot_count &&
                    *source_slot_count > 0 &&
                    *source_slot_count <= self.slot_count &&
                    output_public_key.matrix_type == self.public_key_type() &&
                    input_public_keys
                        .iter()
                        .all(|input| input.matrix_type == self.public_key_type())
            }
        };
        if valid { Ok(()) } else { Err(SlotTransferArtifactError::InvalidGateRequest) }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_transfer_gate_chunk(
        &self,
        builder: &mut GraphBuilder,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        input_public_key: &MatrixWire,
        output_public_key: &MatrixWire,
        source_slots: &[(u32, Option<u32>)],
        identity: &str,
        columns: IndexRange,
    ) -> Result<MatrixFamilyWire, SlotTransferArtifactError> {
        let chunk_columns = columns.end - columns.start;
        if source_slots.is_empty() {
            let mut body = GraphBuilder::new(
                format!("bgg-slot-transfer-empty-gate-{identity}-chunk-{}", columns.start),
                Vec::new(),
            );
            let preimage = body.constant_matrix(
                self.matrix_type(self.b0_public_columns(), chunk_columns),
                ConstantMatrix::Zero,
            );
            body.value_output_wire("preimage", preimage.wire);
            return Ok(builder
                .parallel_loop(
                    body.finish(),
                    IntExpr::constant(0),
                    "destination",
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    &[preimage.matrix_type],
                )?
                .remove(0));
        }
        let mut body = GraphBuilder::new(
            format!("bgg-slot-transfer-gate-{identity}-chunk-{}", columns.start),
            Vec::new(),
        );
        let b0 = body.trapdoor_input(
            "0_b0",
            self.b0_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let secrets = body.family_input(
            "1_secrets",
            self.slot_secret_type(),
            IntExpr::constant(self.slot_count),
        );
        let public_keys = body.family_input(
            "2_public_keys",
            self.public_key_type(),
            IntExpr::constant(self.slot_count),
        );
        let input = body.input("3_input_public_key", self.public_key_type());
        let output = body.input("4_output_public_key", self.public_key_type());
        let destination = body.evaluate_int(IntExpr::Var("destination".to_owned()));
        let source_branches =
            source_slots.iter().map(|(source, _)| body.constant_int(*source)).collect::<Vec<_>>();
        let source = body.select_wire(destination, &source_branches);
        let source_secret = body.family_get_dynamic(&secrets, source);
        let destination_secret = body.family_get_dynamic(&secrets, destination);
        let destination_public_key = body.family_get_dynamic(&public_keys, destination);
        let destination_public_key_chunk = body.slice(
            &destination_public_key,
            None,
            Some(columns),
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let decomposed = body.gadget_decompose(
            &destination_public_key_chunk,
            self.gadget_base.clone(),
            self.matrix_type(self.gadget_columns(), chunk_columns),
        );
        let input_secret = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &source_secret,
            &input,
            self.public_key_type(),
        );
        let rhs = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &input_secret,
            &decomposed,
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let scalar_branches = source_slots
            .iter()
            .map(|(_, scalar)| {
                body.constant_polynomial(
                    self.matrix_type(1, 1),
                    [BigInt::from(scalar.unwrap_or(1))],
                )
            })
            .collect::<Vec<_>>();
        let scalar = body.select(destination, &scalar_branches);
        let rhs = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &rhs,
            &scalar,
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let output_chunk = body.slice(
            &output,
            None,
            Some(columns),
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let lhs = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &destination_secret,
            &output_chunk,
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let target = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &lhs,
            &rhs,
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let target = self.add_error(&mut body, target);
        let preimage = body.preimage_sample(
            &b0,
            &target,
            self.matrix_type(self.b0_public_columns(), chunk_columns),
        );
        body.value_output_wire("preimage", preimage.wire);
        Ok(builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(source_slots.len()),
                "destination",
                Vec::new(),
                vec![
                    base.b0.wire,
                    slots.secrets.wire,
                    slots.public_keys.wire,
                    input_public_key.wire,
                    output_public_key.wire,
                ],
                vec![LoopInputMode::Broadcast; 5],
                &[preimage.matrix_type],
            )?
            .remove(0))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_reduce_gate_chunk(
        &self,
        builder: &mut GraphBuilder,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        input_public_keys: &[MatrixWire],
        output_public_key: &MatrixWire,
        source_slot_count: usize,
        identity: &str,
        columns: IndexRange,
    ) -> Result<MatrixFamilyWire, SlotTransferArtifactError> {
        let chunk_columns = columns.end - columns.start;
        let mut body = GraphBuilder::new(
            format!("bgg-slot-reduce-gate-{identity}-chunk-{}", columns.start),
            Vec::new(),
        );
        let b0 = body.trapdoor_input(
            "0_b0",
            self.b0_public_type(),
            self.trapdoor_sigma.clone(),
            self.gadget_base.clone(),
            IntExpr::constant(self.digit_count),
        );
        let secrets = body.family_input(
            "1_secrets",
            self.slot_secret_type(),
            IntExpr::constant(self.slot_count),
        );
        let public_keys = body.family_input(
            "2_public_keys",
            self.public_key_type(),
            IntExpr::constant(self.slot_count),
        );
        let output = body.input("3_output_public_key", self.public_key_type());
        let input_branches = input_public_keys
            .iter()
            .enumerate()
            .map(|(index, _)| {
                body.input(format!("{}_input_public_key", 4 + index), self.public_key_type())
            })
            .collect::<Vec<_>>();
        let destination = body.evaluate_int(IntExpr::Var("destination".to_owned()));
        let input = body.select(destination, &input_branches);
        let destination_secret = body.family_get_dynamic(&secrets, destination);
        let destination_public_key = body.family_get_dynamic(&public_keys, destination);
        let destination_public_key_chunk = body.slice(
            &destination_public_key,
            None,
            Some(columns),
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let decomposed = body.gadget_decompose(
            &destination_public_key_chunk,
            self.gadget_base.clone(),
            self.matrix_type(self.gadget_columns(), chunk_columns),
        );
        let mut rhs = None;
        for source in 0..source_slot_count {
            let source_secret = body.family_get_static(&secrets, IntExpr::constant(source));
            let input_secret = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &source_secret,
                &input,
                self.public_key_type(),
            );
            let term = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input_secret,
                &decomposed,
                self.matrix_type(self.secret_size, chunk_columns),
            );
            let rotation = body.constant_matrix(
                self.matrix_type(1, 1),
                ConstantMatrix::Rotation { exponent: IntExpr::constant(source) },
            );
            let term = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &term,
                &rotation,
                self.matrix_type(self.secret_size, chunk_columns),
            );
            rhs = Some(match rhs {
                Some(rhs) => body.matrix_binary(
                    MatrixBinaryOp::Add,
                    &rhs,
                    &term,
                    self.matrix_type(self.secret_size, chunk_columns),
                ),
                None => term,
            });
        }
        let output_chunk = body.slice(
            &output,
            None,
            Some(columns),
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let lhs = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &destination_secret,
            &output_chunk,
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let target = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &lhs,
            &rhs.expect("source_slot_count was checked nonzero"),
            self.matrix_type(self.secret_size, chunk_columns),
        );
        let target = self.add_error(&mut body, target);
        let preimage = body.preimage_sample(
            &b0,
            &target,
            self.matrix_type(self.b0_public_columns(), chunk_columns),
        );
        body.value_output_wire("preimage", preimage.wire);
        let mut args =
            vec![base.b0.wire, slots.secrets.wire, slots.public_keys.wire, output_public_key.wire];
        args.extend(input_public_keys.iter().map(|input| input.wire));
        Ok(builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(input_public_keys.len()),
                "destination",
                Vec::new(),
                args.clone(),
                vec![LoopInputMode::Broadcast; args.len()],
                &[preimage.matrix_type],
            )?
            .remove(0))
    }

    fn add_error(&self, builder: &mut GraphBuilder, target: MatrixWire) -> MatrixWire {
        let error = builder.gaussian_sample(target.matrix_type.clone(), self.error_sigma.clone());
        builder.matrix_binary(MatrixBinaryOp::Add, &target, &error, target.matrix_type.clone())
    }

    pub(crate) fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn slot_secret_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.secret_size)
    }

    pub(crate) fn gadget_columns(&self) -> usize {
        self.secret_size * self.digit_count
    }

    pub(crate) fn b0_public_columns(&self) -> usize {
        self.secret_size * (self.digit_count + 2)
    }

    pub(crate) fn b1_public_columns(&self) -> usize {
        self.secret_size * 2 * (self.digit_count + 2)
    }

    fn b0_public_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.b0_public_columns())
    }

    fn b1_public_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size * 2, self.b1_public_columns())
    }

    fn chunk_count(&self, columns: usize) -> usize {
        columns.div_ceil(self.chunk_columns)
    }

    pub(crate) fn chunks(&self, columns: usize) -> Vec<IndexRange> {
        (0..columns)
            .step_by(self.chunk_columns)
            .map(|start| IndexRange { start, end: (start + self.chunk_columns).min(columns) })
            .collect()
    }
}

fn b0_preimage_name(chunk: usize) -> String {
    format!("slot_transfer_slot_preimage_b0_chunk_{chunk}")
}

fn b1_preimage_name(chunk: usize) -> String {
    format!("slot_transfer_slot_preimage_b1_chunk_{chunk}")
}

pub(crate) fn gate_preimage_name(reduction: bool, identity: &str, chunk: usize) -> String {
    let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
    format!("{operation}_gate_{identity}_preimage_chunk_{chunk}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use keccak_asm::Keccak256;
    use mxx_ir_core::{ParamEnv, validate, validate_with_manifests};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn output_matrix(
        result: &mxx_runtime::ExecutionResult<mxx_runtime::backend::poly::CpuDcrtBackend>,
        name: &str,
    ) -> DCRTPolyMatrix {
        let RuntimeValue::Matrix(matrix) = &result.outputs[name] else {
            panic!("{name} must be a matrix");
        };
        matrix.as_ref().clone()
    }

    #[test]
    fn slot_preprocessing_preserves_the_chunked_legacy_relations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let compiler = BggSlotTransferArtifactCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 2,
            slot_count: 11,
            digit_count: parameters.modulus_digits(),
            chunk_columns: 3,
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: RealExpr::from_f64_exact(0.0).expect("finite sigma"),
        };

        let mut store = MemoryArtifactStore::default();
        let mut base_builder = GraphBuilder::new("slot-transfer-base-test", Vec::new());
        let base = compiler.build_base(&mut base_builder).expect("base graph");
        compiler.export_base(&mut base_builder, &base);
        let base_graph = validate(&base_builder.finish(), &ParamEnv::default()).expect("base");
        let mut backend = cpu_backend([parameters.clone()]);
        let base_result =
            execute(&base_graph, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("base execution");
        let base_production = base_result.production_id.expect("base production");
        let base_manifest = store.manifest(&base_production).expect("base manifest").clone();

        let mut slot_builder = GraphBuilder::new("slot-transfer-slot-test", Vec::new());
        let hash_key = slot_builder.bytes_input("hash_key", 32);
        let imported_base = compiler
            .import_base(
                &mut slot_builder,
                &BggSlotTransferBaseArtifacts { production_id: base_production.clone() },
            )
            .expect("base imports");
        let slots =
            compiler.build_slots(&mut slot_builder, hash_key, &imported_base).expect("slot graph");
        compiler.export_slots(&mut slot_builder, &slots);
        let slot_graph = validate_with_manifests(
            &slot_builder.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([(base_production.clone(), base_manifest.clone())]),
        )
        .expect("slot validation");
        let slot_result = execute(
            &slot_graph,
            &mut backend,
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("slot execution");
        let slot_production = slot_result.production_id.expect("slot production");
        let slot_manifest = store.manifest(&slot_production).expect("slot manifest").clone();
        let b0_chunks = compiler.chunks(compiler.b1_public_columns());
        assert!(b0_chunks.len() >= 10, "the test must exercise numeric ordering past chunk 9");
        assert!(
            b0_chunks.iter().any(|range| range.end - range.start < compiler.chunk_columns),
            "the test must exercise a nonmultiple tail chunk"
        );

        let mut inspect = GraphBuilder::new("slot-transfer-slot-inspect", Vec::new());
        let base = compiler
            .import_base(
                &mut inspect,
                &BggSlotTransferBaseArtifacts { production_id: base_production.clone() },
            )
            .expect("base imports");
        let slots = compiler
            .import_slots(
                &mut inspect,
                &BggSlotTransferSlotArtifacts { production_id: slot_production.clone() },
            )
            .expect("slot imports");
        inspect.value_output_wire("b0", base.b0.public.wire);
        inspect.value_output_wire("b1", base.b1.public.wire);
        let inspected_slot = 10;
        let secret = inspect.family_get_static(&slots.secrets, IntExpr::constant(inspected_slot));
        let public_key =
            inspect.family_get_static(&slots.public_keys, IntExpr::constant(inspected_slot));
        inspect.value_output_wire("secret", secret.wire);
        inspect.value_output_wire("public_key", public_key.wire);
        for (chunk, family) in slots.b0_preimage_chunks.iter().enumerate() {
            let value = inspect.family_get_static(family, IntExpr::constant(inspected_slot));
            inspect.value_output_wire(format!("b0_preimage_{chunk}"), value.wire);
        }
        for (chunk, family) in slots.b1_preimage_chunks.iter().enumerate() {
            let value = inspect.family_get_static(family, IntExpr::constant(inspected_slot));
            inspect.value_output_wire(format!("b1_preimage_{chunk}"), value.wire);
        }
        let inspect = validate_with_manifests(
            &inspect.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([
                (base_production.clone(), base_manifest.clone()),
                (slot_production.clone(), slot_manifest.clone()),
            ]),
        )
        .expect("inspect validation");
        let result =
            execute(&inspect, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .expect("inspect execution");

        let b0 = output_matrix(&result, "b0");
        let b1 = output_matrix(&result, "b1");
        let secret = output_matrix(&result, "secret");
        let public_key = output_matrix(&result, "public_key");
        let expected_public_key = DCRTPolyHashSampler::<Keccak256>::new().sample_hash(
            &parameters,
            [0x42; 32],
            format!("slot_transfer_slot_a_{inspected_slot}"),
            compiler.secret_size,
            compiler.gadget_columns(),
            DistType::FinRingDist,
        );
        assert_eq!(public_key, expected_public_key);
        let identity = DCRTPolyMatrix::identity(&parameters, compiler.secret_size, None);
        let secret_identity = secret.clone().concat_columns(&[&identity]);
        for (chunk, columns) in
            compiler.chunks(compiler.b1_public_columns()).into_iter().enumerate()
        {
            let preimage = output_matrix(&result, &format!("b0_preimage_{chunk}"));
            assert_eq!(
                b0.clone() * &preimage,
                secret_identity.clone() * &b1.slice_columns(columns.start, columns.end)
            );
        }
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
        for (chunk, columns) in compiler.chunks(compiler.gadget_columns()).into_iter().enumerate() {
            let preimage = output_matrix(&result, &format!("b1_preimage_{chunk}"));
            let negative = -(secret.clone() * &gadget.slice_columns(columns.start, columns.end));
            let expected =
                public_key.slice_columns(columns.start, columns.end).concat_rows(&[&negative]);
            assert_eq!(b1.clone() * &preimage, expected);
        }

        let mut gate_builder = GraphBuilder::new("slot-transfer-gate-test", Vec::new());
        let base = compiler
            .import_base(
                &mut gate_builder,
                &BggSlotTransferBaseArtifacts { production_id: base_production.clone() },
            )
            .expect("base imports");
        let slots = compiler
            .import_slots(
                &mut gate_builder,
                &BggSlotTransferSlotArtifacts { production_id: slot_production.clone() },
            )
            .expect("slot imports");
        let gate_hash_key = gate_builder.bytes_input("hash_key", 32);
        let input_public_key = gate_builder.constant_matrix(
            compiler.public_key_type(),
            ConstantMatrix::Gadget { base: compiler.gadget_base.clone(), small: false },
        );
        let transfer_output_public_key = gate_builder.hash_sample(
            gate_hash_key,
            compiler.public_key_type(),
            HashVariant::Plain,
            b"slot_transfer_gate_a_out_7".to_vec(),
            Vec::new(),
            None,
            None,
        );
        let reduce_output_public_key = gate_builder.hash_sample(
            gate_hash_key,
            compiler.public_key_type(),
            HashVariant::Plain,
            b"slot_reduce_gate_a_out_8".to_vec(),
            Vec::new(),
            None,
            None,
        );
        let requests = gate_requests(
            &input_public_key,
            &transfer_output_public_key,
            &reduce_output_public_key,
        );
        let gates = compiler
            .build_gate_preimages(&mut gate_builder, &base, &slots, &requests)
            .expect("gate preimages");
        compiler.export_gate_preimages(&mut gate_builder, &gates);
        let gate_graph = validate_with_manifests(
            &gate_builder.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([
                (base_production.clone(), base_manifest.clone()),
                (slot_production.clone(), slot_manifest.clone()),
            ]),
        )
        .expect("gate validation");
        let gate_result = execute(
            &gate_graph,
            &mut backend,
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("gate execution");
        let gate_production = gate_result.production_id.expect("gate production");
        let gate_manifest = store.manifest(&gate_production).expect("gate manifest").clone();
        for chunk in 0..compiler.chunk_count(compiler.gadget_columns()) {
            assert_eq!(
                gate_manifest.artifacts[&gate_preimage_name(false, "9", chunk)].family_count,
                Some(0)
            );
        }
        let mut invalid_import = GraphBuilder::new("invalid-slot-transfer-gate-import", Vec::new());
        let wrong_type =
            invalid_import.constant_matrix(compiler.matrix_type(1, 1), ConstantMatrix::Identity);
        let invalid_request = BggSlotTransferGateRequest::Transfer {
            identity: "7".to_owned(),
            input_public_key: wrong_type,
            output_public_key: transfer_output_public_key.clone(),
            source_slots: vec![(0, None)],
        };
        assert!(matches!(
            compiler.import_gate_preimages(
                &mut invalid_import,
                &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                &[invalid_request],
            ),
            Err(SlotTransferArtifactError::InvalidGateRequest)
        ));

        let mut inspect = GraphBuilder::new("slot-transfer-gate-inspect", Vec::new());
        let base = compiler
            .import_base(
                &mut inspect,
                &BggSlotTransferBaseArtifacts { production_id: base_production.clone() },
            )
            .expect("base imports");
        let slots = compiler
            .import_slots(
                &mut inspect,
                &BggSlotTransferSlotArtifacts { production_id: slot_production.clone() },
            )
            .expect("slot imports");
        let gate_hash_key = inspect.bytes_input("hash_key", 32);
        let input_public_key = inspect.constant_matrix(
            compiler.public_key_type(),
            ConstantMatrix::Gadget { base: compiler.gadget_base.clone(), small: false },
        );
        let transfer_output_public_key = inspect.hash_sample(
            gate_hash_key,
            compiler.public_key_type(),
            HashVariant::Plain,
            b"slot_transfer_gate_a_out_7".to_vec(),
            Vec::new(),
            None,
            None,
        );
        let reduce_output_public_key = inspect.hash_sample(
            gate_hash_key,
            compiler.public_key_type(),
            HashVariant::Plain,
            b"slot_reduce_gate_a_out_8".to_vec(),
            Vec::new(),
            None,
            None,
        );
        let requests = gate_requests(
            &input_public_key,
            &transfer_output_public_key,
            &reduce_output_public_key,
        );
        let gates = compiler
            .import_gate_preimages(
                &mut inspect,
                &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                &requests,
            )
            .expect("gate imports");
        inspect.value_output_wire("b0", base.b0.public.wire);
        inspect.value_output_wire("transfer_output_public_key", transfer_output_public_key.wire);
        inspect.value_output_wire("reduce_output_public_key", reduce_output_public_key.wire);
        for slot in 0..2 {
            let secret = inspect.family_get_static(&slots.secrets, IntExpr::constant(slot));
            let public_key = inspect.family_get_static(&slots.public_keys, IntExpr::constant(slot));
            inspect.value_output_wire(format!("secret_{slot}"), secret.wire);
            inspect.value_output_wire(format!("public_key_{slot}"), public_key.wire);
        }
        for (chunk, _) in compiler.chunks(compiler.gadget_columns()).iter().enumerate() {
            for (reduction, identity) in [(false, "7"), (true, "8")] {
                let name = gate_preimage_name(reduction, identity, chunk);
                for destination in 0..2 {
                    let preimage = inspect.family_get_static(
                        &gates.preimage_chunks[&name],
                        IntExpr::constant(destination),
                    );
                    let operation = if reduction { "reduce" } else { "transfer" };
                    inspect.value_output_wire(
                        format!("{operation}_preimage_{chunk}_{destination}"),
                        preimage.wire,
                    );
                }
            }
        }
        let inspect = validate_with_manifests(
            &inspect.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([
                (base_production, base_manifest),
                (slot_production, slot_manifest),
                (gate_production, gate_manifest),
            ]),
        )
        .expect("gate inspect validation");
        let result = execute(
            &inspect,
            &mut backend,
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("gate inspect execution");
        let b0 = output_matrix(&result, "b0");
        let secrets = [output_matrix(&result, "secret_0"), output_matrix(&result, "secret_1")];
        let public_keys =
            [output_matrix(&result, "public_key_0"), output_matrix(&result, "public_key_1")];
        let input_public_key = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
        let transfer_output_public_key = output_matrix(&result, "transfer_output_public_key");
        let reduce_output_public_key = output_matrix(&result, "reduce_output_public_key");
        for (chunk, columns) in compiler.chunks(compiler.gadget_columns()).into_iter().enumerate() {
            for destination in 0..2 {
                let source = [1, 0][destination];
                let scalar = DCRTPoly::from_usize_to_constant(&parameters, [1, 3][destination]);
                let rhs = ((secrets[source].clone() * &input_public_key) *
                    public_keys[destination]
                        .slice_columns(columns.start, columns.end)
                        .decompose()) *
                    &scalar;
                let expected = secrets[destination].clone() *
                    &transfer_output_public_key.slice_columns(columns.start, columns.end) -
                    &rhs;
                assert_eq!(
                    b0.clone() *
                        &output_matrix(
                            &result,
                            &format!("transfer_preimage_{chunk}_{destination}"),
                        ),
                    expected
                );

                let decomposed =
                    public_keys[destination].slice_columns(columns.start, columns.end).decompose();
                let mut reduction_rhs = DCRTPolyMatrix::zero(
                    &parameters,
                    compiler.secret_size,
                    columns.end - columns.start,
                );
                for source in 0..2 {
                    let rotation = DCRTPoly::const_rotate_poly(&parameters, source);
                    let term =
                        ((secrets[source].clone() * &input_public_key) * &decomposed) * &rotation;
                    reduction_rhs = reduction_rhs + &term;
                }
                let expected = secrets[destination].clone() *
                    &reduce_output_public_key.slice_columns(columns.start, columns.end) -
                    &reduction_rhs;
                assert_eq!(
                    b0.clone() *
                        &output_matrix(
                            &result,
                            &format!("reduce_preimage_{chunk}_{destination}"),
                        ),
                    expected
                );
            }
        }
    }

    fn gate_requests(
        input_public_key: &MatrixWire,
        transfer_output_public_key: &MatrixWire,
        reduce_output_public_key: &MatrixWire,
    ) -> Vec<BggSlotTransferGateRequest> {
        vec![
            BggSlotTransferGateRequest::Transfer {
                identity: "7".to_owned(),
                input_public_key: input_public_key.clone(),
                output_public_key: transfer_output_public_key.clone(),
                source_slots: vec![(1, None), (0, Some(3))],
            },
            BggSlotTransferGateRequest::Reduce {
                identity: "8".to_owned(),
                input_public_keys: vec![input_public_key.clone(), input_public_key.clone()],
                output_public_key: reduce_output_public_key.clone(),
                source_slot_count: 2,
            },
            BggSlotTransferGateRequest::Transfer {
                identity: "9".to_owned(),
                input_public_key: input_public_key.clone(),
                output_public_key: transfer_output_public_key.clone(),
                source_slots: Vec::new(),
            },
        ]
    }
}
