//! Cryptographic slot-transfer preprocessing and artifact wiring.

use crate::BggSlotTransferGateRequest;
use mxx_dsl::{Bytes, DslContext, DslError, Family, HashTag, Mat, Parallel, Ring, Trapdoor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, IndexRange},
    types::MatrixType,
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use thiserror::Error;

const B0_PUBLIC: &str = "slot_transfer_b0_public";
const B0_TRAPDOOR: &str = "slot_transfer_b0_trapdoor";
const B1_PUBLIC: &str = "slot_transfer_b1_public";
const B1_TRAPDOOR: &str = "slot_transfer_b1_trapdoor";
const SLOT_SECRET: &str = "slot_transfer_slot_secret";
const SLOT_PUBLIC_KEY: &str = "slot_transfer_slot_a";

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Debug, Error)]
pub enum BggSlotTransferArtifactError {
    #[error("slot-transfer dimensions, slot count, and chunk width must be nonzero")]
    EmptyLayout,
    #[error("slot-transfer gate request is incompatible with the artifact layout")]
    InvalidGateRequest,
    #[error("slot-transfer artifact family is missing: {0}")]
    MissingArtifact(String),
    #[error(transparent)]
    Dsl(#[from] DslError),
}

#[derive(Clone)]
pub struct BggSlotTransferBaseWires {
    pub b0: Trapdoor,
    pub b1: Trapdoor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSlotTransferBaseArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone)]
pub struct BggSlotTransferSlotWires {
    pub secrets: Family<Mat>,
    pub public_keys: Family<Mat>,
    pub b0_preimage_chunks: Vec<Family<Mat>>,
    pub b1_preimage_chunks: Vec<Family<Mat>>,
}

#[derive(Clone)]
pub struct BggSlotTransferPublicSlotWires {
    pub public_keys: Family<Mat>,
    pub b0_preimage_chunks: Vec<Family<Mat>>,
    pub b1_preimage_chunks: Vec<Family<Mat>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSlotTransferSlotArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone, Default)]
pub struct BggSlotTransferGateWires {
    pub preimage_chunks: BTreeMap<String, Family<Mat>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSlotTransferGateArtifacts {
    pub production_id: ProductionId,
}

impl BggSlotTransferArtifactCompiler {
    pub fn validate_layout(&self) -> Result<(), BggSlotTransferArtifactError> {
        if self.secret_size == 0 ||
            self.slot_count == 0 ||
            self.digit_count == 0 ||
            self.chunk_columns == 0
        {
            return Err(BggSlotTransferArtifactError::EmptyLayout);
        }
        Ok(())
    }

    pub fn public_key_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.gadget_columns())
    }

    pub fn build_base(&self) -> Result<BggSlotTransferBaseWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let ring = self.ring();
        Ok(BggSlotTransferBaseWires {
            b0: ring.sample_trapdoor(
                self.secret_size,
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                self.digit_count,
            ),
            b1: ring.sample_trapdoor(
                self.secret_size * 2,
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                self.digit_count,
            ),
        })
    }

    pub fn export_base(
        &self,
        context: DslContext,
        base: BggSlotTransferBaseWires,
    ) -> Result<DslContext, BggSlotTransferArtifactError> {
        Ok(context
            .public_output(B0_PUBLIC, base.b0.public_matrix())?
            .private_trapdoor_output(B0_TRAPDOOR, base.b0)?
            .public_output(B1_PUBLIC, base.b1.public_matrix())?
            .private_trapdoor_output(B1_TRAPDOOR, base.b1)?)
    }

    pub fn import_base(
        &self,
        artifacts: &BggSlotTransferBaseArtifacts,
    ) -> Result<BggSlotTransferBaseWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let ring = self.ring();
        Ok(BggSlotTransferBaseWires {
            b0: ring.trapdoor_artifact_input(
                artifacts.production_id.clone(),
                B0_PUBLIC,
                B0_TRAPDOOR,
                self.secret_size,
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                self.digit_count,
            ),
            b1: ring.trapdoor_artifact_input(
                artifacts.production_id.clone(),
                B1_PUBLIC,
                B1_TRAPDOOR,
                self.secret_size * 2,
                self.trapdoor_sigma.clone(),
                self.gadget_base.clone(),
                self.digit_count,
            ),
        })
    }

    pub fn build_slots(
        &self,
        hash_key: Bytes,
        base: &BggSlotTransferBaseWires,
    ) -> Result<BggSlotTransferSlotWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let ring = self.ring();
        let secret_size = self.secret_size;
        let public_columns = self.gadget_columns();
        let (secrets, public_keys) = Parallel::range(self.slot_count).map_values({
            let ring = ring.clone();
            move |index| {
                let mut tag = HashTag::from(b"slot_transfer_slot_a_".as_slice());
                tag.push_decimal(index);
                (
                    ring.uniform_in((secret_size, secret_size), -1, 1),
                    ring.hash_matrix(hash_key.clone(), tag, (secret_size, public_columns)),
                )
            }
        })?;

        let identity = ring.identity(secret_size);
        let b1_public = base.b1.public_matrix();
        let b0_preimage_chunks = self
            .chunks(self.b1_public_columns())
            .into_iter()
            .map(|columns| {
                let target_columns = columns.clone();
                let error_sigma = self.error_sigma.clone();
                let ring = ring.clone();
                let b0 = base.b0.clone();
                let b1_public = b1_public.clone();
                let identity = identity.clone();
                secrets.clone().parallel_map(move |_, secret| {
                    let secret_identity =
                        Mat::concat(ConcatAxis::Columns, vec![secret, identity.clone()]);
                    let target = secret_identity *
                        b1_public.clone().slice(None, Some(target_columns.clone()));
                    let columns = target.matrix_type().columns.clone();
                    b0.sample_preimage(
                        target + ring.gaussian((secret_size, columns.clone()), error_sigma.clone()),
                        (b0.public_matrix().matrix_type().columns.clone(), columns),
                    )
                    .as_mat()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let gadget = ring.gadget(secret_size, self.gadget_base.clone(), self.digit_count);
        let b1_preimage_chunks = self
            .chunks(self.gadget_columns())
            .into_iter()
            .map(|columns| {
                let target_columns = columns.clone();
                let error_sigma = self.error_sigma.clone();
                let ring = ring.clone();
                let b1 = base.b1.clone();
                let gadget = gadget.clone();
                secrets.clone().parallel_zip(public_keys.clone(), move |_, secret, public_key| {
                    let a_chunk = public_key.slice(None, Some(target_columns.clone()));
                    let gadget_chunk = gadget.clone().slice(None, Some(target_columns.clone()));
                    let secret_gadget = -(secret * gadget_chunk);
                    let target = Mat::concat(ConcatAxis::Rows, vec![a_chunk, secret_gadget]);
                    let columns = target.matrix_type().columns.clone();
                    b1.sample_preimage(
                        target +
                            ring.gaussian(
                                (secret_size * 2, columns.clone()),
                                error_sigma.clone(),
                            ),
                        (b1.public_matrix().matrix_type().columns.clone(), columns),
                    )
                    .as_mat()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BggSlotTransferSlotWires {
            secrets,
            public_keys,
            b0_preimage_chunks,
            b1_preimage_chunks,
        })
    }

    pub fn export_slots(
        &self,
        context: DslContext,
        slots: BggSlotTransferSlotWires,
    ) -> Result<DslContext, BggSlotTransferArtifactError> {
        let context = context
            .private_family_output(SLOT_SECRET, slots.secrets)?
            .public_family_output(SLOT_PUBLIC_KEY, slots.public_keys)?;
        let context = slots.b0_preimage_chunks.into_iter().enumerate().try_fold(
            context,
            |context, (chunk, family)| {
                context.public_family_output(b0_preimage_name(chunk), family)
            },
        )?;
        Ok(slots.b1_preimage_chunks.into_iter().enumerate().try_fold(
            context,
            |context, (chunk, family)| {
                context.public_family_output(b1_preimage_name(chunk), family)
            },
        )?)
    }

    pub fn import_slots(
        &self,
        artifacts: &BggSlotTransferSlotArtifacts,
    ) -> Result<BggSlotTransferSlotWires, BggSlotTransferArtifactError> {
        let ring = self.ring();
        let public = self.import_slots_public(artifacts)?;
        Ok(BggSlotTransferSlotWires {
            secrets: ring.family_artifact_input(
                artifacts.production_id.clone(),
                SLOT_SECRET,
                self.slot_count,
                (self.secret_size, self.secret_size),
                ArtifactConfidentiality::Private,
            ),
            public_keys: public.public_keys,
            b0_preimage_chunks: public.b0_preimage_chunks,
            b1_preimage_chunks: public.b1_preimage_chunks,
        })
    }

    pub fn import_slots_public(
        &self,
        artifacts: &BggSlotTransferSlotArtifacts,
    ) -> Result<BggSlotTransferPublicSlotWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let ring = self.ring();
        Ok(BggSlotTransferPublicSlotWires {
            public_keys: ring.family_artifact_input(
                artifacts.production_id.clone(),
                SLOT_PUBLIC_KEY,
                self.slot_count,
                (self.secret_size, self.gadget_columns()),
                ArtifactConfidentiality::Public,
            ),
            b0_preimage_chunks: self.import_slot_chunks(
                &artifacts.production_id,
                true,
                self.b1_public_columns(),
                self.b0_public_columns(),
            ),
            b1_preimage_chunks: self.import_slot_chunks(
                &artifacts.production_id,
                false,
                self.gadget_columns(),
                self.b1_public_columns(),
            ),
        })
    }

    pub fn build_gate_preimages(
        &self,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        requests: &[BggSlotTransferGateRequest],
    ) -> Result<BggSlotTransferGateWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let mut preimage_chunks = BTreeMap::new();
        for request in requests {
            self.validate_gate_request(request)?;
            for (chunk, columns) in self.chunks(self.gadget_columns()).into_iter().enumerate() {
                let (name, family) = match request {
                    BggSlotTransferGateRequest::Transfer {
                        identity,
                        input_public_key,
                        output_public_key,
                        source_slots,
                    } => (
                        gate_preimage_name(false, identity, chunk),
                        self.build_transfer_gate_chunk(
                            base,
                            slots,
                            input_public_key,
                            output_public_key,
                            source_slots,
                            columns,
                        )?,
                    ),
                    BggSlotTransferGateRequest::Reduce {
                        identity,
                        input_public_keys,
                        output_public_key,
                        source_slot_count,
                    } => (
                        gate_preimage_name(true, identity, chunk),
                        self.build_reduce_gate_chunk(
                            base,
                            slots,
                            input_public_keys,
                            output_public_key,
                            *source_slot_count,
                            columns,
                        )?,
                    ),
                };
                preimage_chunks.insert(name, family);
            }
        }
        Ok(BggSlotTransferGateWires { preimage_chunks })
    }

    pub fn export_gate_preimages(
        &self,
        context: DslContext,
        gates: BggSlotTransferGateWires,
    ) -> Result<DslContext, BggSlotTransferArtifactError> {
        Ok(gates.preimage_chunks.into_iter().try_fold(context, |context, (name, family)| {
            context.public_family_output(name, family)
        })?)
    }

    pub fn import_gate_preimages(
        &self,
        artifacts: &BggSlotTransferGateArtifacts,
        requests: &[BggSlotTransferGateRequest],
    ) -> Result<BggSlotTransferGateWires, BggSlotTransferArtifactError> {
        self.validate_layout()?;
        let ring = self.ring();
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
                let family = ring.family_artifact_input(
                    artifacts.production_id.clone(),
                    name.clone(),
                    count,
                    (self.b0_public_columns(), range_len(&columns)),
                    ArtifactConfidentiality::Public,
                );
                preimage_chunks.insert(name, family);
            }
        }
        Ok(BggSlotTransferGateWires { preimage_chunks })
    }

    fn build_transfer_gate_chunk(
        &self,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        input: &Mat,
        output: &Mat,
        source_slots: &[(u32, Option<u32>)],
        columns: IndexRange,
    ) -> Result<Family<Mat>, BggSlotTransferArtifactError> {
        let ring = self.ring();
        if source_slots.is_empty() {
            let rows = self.b0_public_columns();
            let columns = range_len(&columns);
            return Ok(Parallel::range(0).map(move |_| ring.zero((rows, columns)))?);
        }
        let results = source_slots
            .iter()
            .enumerate()
            .map(|(destination, (source, scalar))| {
                let source = usize::try_from(*source).expect("u32 fits usize");
                let source_secret = slots.secrets.get_static(source);
                let destination_secret = slots.secrets.get_static(destination);
                let destination_public = slots.public_keys.get_static(destination);
                let destination_chunk = destination_public.slice(None, Some(columns.clone()));
                let decomposed = destination_chunk
                    .decompose(self.gadget_base.clone(), self.digit_count)
                    .as_mat();
                let rhs = source_secret *
                    input.clone() *
                    decomposed *
                    ring.polynomial([IntExpr::constant(scalar.unwrap_or(1))]);
                let lhs = destination_secret * output.clone().slice(None, Some(columns.clone()));
                let target = lhs - rhs +
                    ring.gaussian(
                        (self.secret_size, range_len(&columns)),
                        self.error_sigma.clone(),
                    );
                base.b0
                    .sample_preimage(target, (self.b0_public_columns(), range_len(&columns)))
                    .as_mat()
            })
            .collect::<Vec<_>>();
        Ok(Family::pack(results)?)
    }

    fn build_reduce_gate_chunk(
        &self,
        base: &BggSlotTransferBaseWires,
        slots: &BggSlotTransferSlotWires,
        inputs: &[Mat],
        output: &Mat,
        source_slot_count: usize,
        columns: IndexRange,
    ) -> Result<Family<Mat>, BggSlotTransferArtifactError> {
        let ring = self.ring();
        let results = inputs
            .iter()
            .enumerate()
            .map(|(destination, input)| {
                let destination_secret = slots.secrets.get_static(destination);
                let destination_chunk =
                    slots.public_keys.get_static(destination).slice(None, Some(columns.clone()));
                let decomposed = destination_chunk
                    .decompose(self.gadget_base.clone(), self.digit_count)
                    .as_mat();
                let rhs = (0..source_slot_count)
                    .map(|source| {
                        slots.secrets.get_static(source) *
                            input.clone() *
                            decomposed.clone() *
                            ring.constant(
                                (1, 1),
                                ConstantMatrix::Rotation { exponent: IntExpr::constant(source) },
                            )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .reduce(|sum, term| sum + term)
                    .expect("validated nonzero source slots");
                let lhs = destination_secret * output.clone().slice(None, Some(columns.clone()));
                let target = lhs - rhs +
                    ring.gaussian(
                        (self.secret_size, range_len(&columns)),
                        self.error_sigma.clone(),
                    );
                base.b0
                    .sample_preimage(target, (self.b0_public_columns(), range_len(&columns)))
                    .as_mat()
            })
            .collect::<Vec<_>>();
        Ok(Family::pack(results)?)
    }

    fn validate_gate_request(
        &self,
        request: &BggSlotTransferGateRequest,
    ) -> Result<(), BggSlotTransferArtifactError> {
        let valid = match request {
            BggSlotTransferGateRequest::Transfer {
                input_public_key,
                output_public_key,
                source_slots,
                ..
            } => {
                input_public_key.matrix_type() == &self.public_key_type() &&
                    output_public_key.matrix_type() == &self.public_key_type() &&
                    source_slots.len() <= self.slot_count &&
                    source_slots
                        .par_iter()
                        .all(|(source, _)| (*source as usize) < self.slot_count)
            }
            BggSlotTransferGateRequest::Reduce {
                input_public_keys,
                output_public_key,
                source_slot_count,
                ..
            } => {
                !input_public_keys.is_empty() &&
                    input_public_keys.len() <= *source_slot_count &&
                    *source_slot_count <= self.slot_count &&
                    output_public_key.matrix_type() == &self.public_key_type() &&
                    input_public_keys
                        .par_iter()
                        .all(|input| input.matrix_type() == &self.public_key_type())
            }
        };
        if valid { Ok(()) } else { Err(BggSlotTransferArtifactError::InvalidGateRequest) }
    }

    fn import_slot_chunks(
        &self,
        production_id: &ProductionId,
        b0: bool,
        columns: usize,
        rows: usize,
    ) -> Vec<Family<Mat>> {
        let ring = self.ring();
        self.chunks(columns)
            .into_iter()
            .enumerate()
            .map(|(chunk, range)| {
                ring.family_artifact_input(
                    production_id.clone(),
                    if b0 { b0_preimage_name(chunk) } else { b1_preimage_name(chunk) },
                    self.slot_count,
                    (rows, range_len(&range)),
                    ArtifactConfidentiality::Public,
                )
            })
            .collect()
    }

    pub(crate) fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }
    pub(crate) fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        self.ring().matrix_type((rows, columns))
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
    pub(crate) fn chunks(&self, columns: usize) -> Vec<IndexRange> {
        (0..columns)
            .step_by(self.chunk_columns)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|start| IndexRange {
                start: IntExpr::constant(start),
                end: IntExpr::constant((start + self.chunk_columns).min(columns)),
            })
            .collect()
    }
}

fn range_len(range: &IndexRange) -> usize {
    let (IntExpr::Const(start), IntExpr::Const(end)) = (&range.start, &range.end) else {
        unreachable!("slot-transfer chunk ranges are static")
    };
    usize::try_from(end - start).expect("nonnegative static chunk length")
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
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_ir_core::ParamEnv;
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
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn compiler() -> BggSlotTransferArtifactCompiler {
        BggSlotTransferArtifactCompiler {
            modulus: 65_537.into(),
            ring_dimension: 8.into(),
            secret_size: 2,
            slot_count: 3,
            digit_count: 4,
            chunk_columns: 3,
            gadget_base: 4.into(),
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(3),
        }
    }

    fn static_range(range: &IndexRange) -> (usize, usize) {
        let (IntExpr::Const(start), IntExpr::Const(end)) = (&range.start, &range.end) else {
            panic!("test slot-transfer chunks must be static")
        };
        (
            usize::try_from(start).expect("nonnegative start"),
            usize::try_from(end).expect("nonnegative end"),
        )
    }

    #[test]
    fn runtime_preprocessing_and_gate_preimages_satisfy_the_primitive_relations() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let compiler = BggSlotTransferArtifactCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            slot_count: 3,
            digit_count: parameters.modulus_digits(),
            chunk_columns: 2,
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: RealExpr::from_integer(0),
        };
        let ring = compiler.ring();
        let base = compiler.build_base().expect("base");
        let slots = compiler
            .build_slots(ring.bytes_input("slot_hash_key", 32), &base)
            .expect("slot preprocessing");
        let input_key_value = row(&parameters, compiler.gadget_columns(), 1);
        let input_key_two_value = row(&parameters, compiler.gadget_columns(), 3);
        let transfer_output_value = row(&parameters, compiler.gadget_columns(), 5);
        let reduce_output_value = row(&parameters, compiler.gadget_columns(), 7);
        let input_key = ring.input("input_key", (1, compiler.gadget_columns()));
        let input_key_two = ring.input("input_key_two", (1, compiler.gadget_columns()));
        let transfer_output = ring.input("transfer_output", (1, compiler.gadget_columns()));
        let reduce_output = ring.input("reduce_output", (1, compiler.gadget_columns()));
        let requests = [
            BggSlotTransferGateRequest::Transfer {
                identity: "transfer".to_owned(),
                input_public_key: input_key,
                output_public_key: transfer_output,
                source_slots: vec![(1, None), (0, Some(3))],
            },
            BggSlotTransferGateRequest::Reduce {
                identity: "reduce".to_owned(),
                input_public_keys: vec![input_key_two.clone(), input_key_two],
                output_public_key: reduce_output,
                source_slot_count: 2,
            },
        ];
        let gates =
            compiler.build_gate_preimages(&base, &slots, &requests).expect("gate preimages");

        let mut context = DslContext::new("slot-transfer-runtime-relations")
            .output("b0", base.b0.public_matrix())
            .expect("b0")
            .output("b1", base.b1.public_matrix())
            .expect("b1");
        for slot in 0..compiler.slot_count {
            context = context
                .output(format!("secret_{slot}"), slots.secrets.get_static(slot))
                .expect("secret")
                .output(format!("public_{slot}"), slots.public_keys.get_static(slot))
                .expect("public key");
            for (chunk, family) in slots.b0_preimage_chunks.iter().enumerate() {
                context = context
                    .output(format!("slot_b0_{chunk}_{slot}"), family.get_static(slot))
                    .expect("b0 preimage");
            }
            for (chunk, family) in slots.b1_preimage_chunks.iter().enumerate() {
                context = context
                    .output(format!("slot_b1_{chunk}_{slot}"), family.get_static(slot))
                    .expect("b1 preimage");
            }
        }
        for (reduction, identity) in [(false, "transfer"), (true, "reduce")] {
            for chunk in 0..compiler.chunks(compiler.gadget_columns()).len() {
                let family =
                    &gates.preimage_chunks[&gate_preimage_name(reduction, identity, chunk)];
                for destination in 0..2 {
                    context = context
                        .output(
                            format!("gate_{identity}_{chunk}_{destination}"),
                            family.get_static(destination),
                        )
                        .expect("gate preimage");
                }
            }
        }
        let result = execute_graph(
            context.build().expect("runtime graph"),
            parameters.clone(),
            BTreeMap::from([
                ("slot_hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
                ("input_key".to_owned(), RuntimeValue::matrix(input_key_value.clone())),
                ("input_key_two".to_owned(), RuntimeValue::matrix(input_key_two_value.clone())),
                ("transfer_output".to_owned(), RuntimeValue::matrix(transfer_output_value.clone())),
                ("reduce_output".to_owned(), RuntimeValue::matrix(reduce_output_value.clone())),
            ]),
        );

        let b0 = matrix_output(&result, "b0");
        let b1 = matrix_output(&result, "b1");
        let identity = DCRTPolyMatrix::identity(&parameters, compiler.secret_size, None);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
        for slot in 0..compiler.slot_count {
            let secret = matrix_output(&result, &format!("secret_{slot}"));
            let public = matrix_output(&result, &format!("public_{slot}"));
            let secret_identity = secret.clone().concat_columns(&[&identity]);
            for (chunk, range) in
                compiler.chunks(compiler.b1_public_columns()).into_iter().enumerate()
            {
                let (start, end) = static_range(&range);
                assert_eq!(
                    b0.clone() * matrix_output(&result, &format!("slot_b0_{chunk}_{slot}")),
                    secret_identity.clone() * &b1.slice_columns(start, end)
                );
            }
            for (chunk, range) in compiler.chunks(compiler.gadget_columns()).into_iter().enumerate()
            {
                let (start, end) = static_range(&range);
                let expected = public
                    .slice_columns(start, end)
                    .concat_rows(&[&-(secret.clone() * &gadget.slice_columns(start, end))]);
                assert_eq!(
                    b1.clone() * matrix_output(&result, &format!("slot_b1_{chunk}_{slot}")),
                    expected
                );
            }
        }

        for (chunk, range) in compiler.chunks(compiler.gadget_columns()).into_iter().enumerate() {
            let (start, end) = static_range(&range);
            for destination in 0..2 {
                let source = [1usize, 0][destination];
                let scalar =
                    DCRTPoly::from_usize_to_constant(&parameters, [1usize, 3][destination]);
                let source_secret = matrix_output(&result, &format!("secret_{source}"));
                let destination_secret = matrix_output(&result, &format!("secret_{destination}"));
                let destination_public = matrix_output(&result, &format!("public_{destination}"));
                let rhs = ((source_secret.clone() * &input_key_value) *
                    &destination_public.slice_columns(start, end).decompose()) *
                    &scalar;
                let expected = destination_secret.clone() *
                    &transfer_output_value.slice_columns(start, end) -
                    &rhs;
                assert_eq!(
                    b0.clone() *
                        matrix_output(&result, &format!("gate_transfer_{chunk}_{destination}"),),
                    expected
                );

                let decomposed = destination_public.slice_columns(start, end).decompose();
                let mut rhs = DCRTPolyMatrix::zero(&parameters, compiler.secret_size, end - start);
                for source in 0..2 {
                    let rotation = DCRTPoly::const_rotate_poly(&parameters, source);
                    let source_secret = matrix_output(&result, &format!("secret_{source}"));
                    rhs = rhs +
                        &(((source_secret.clone() * &input_key_two_value) * &decomposed) *
                            &rotation);
                }
                let expected = destination_secret.clone() *
                    &reduce_output_value.slice_columns(start, end) -
                    &rhs;
                assert_eq!(
                    b0.clone() *
                        matrix_output(&result, &format!("gate_reduce_{chunk}_{destination}"),),
                    expected
                );
            }
        }
    }

    #[test]
    fn runtime_artifact_productions_preserve_import_order_tail_chunks_and_gate_families() {
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
            error_sigma: RealExpr::from_integer(0),
        };
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();

        let base = compiler.build_base().expect("base");
        let base_graph = compiler
            .export_base(DslContext::new("slot-base-production"), base)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let base_result =
            execute(&base_graph, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let base_production = base_result.production_id.expect("base production");
        let base_manifest = store.manifest(&base_production).unwrap().clone();

        let imported_base = compiler
            .import_base(&BggSlotTransferBaseArtifacts { production_id: base_production.clone() })
            .unwrap();
        let slots = compiler
            .build_slots(compiler.ring().bytes_input("slot-hash-key", 32), &imported_base)
            .unwrap();
        let slot_graph = compiler
            .export_slots(DslContext::new("slot-production"), slots)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(base_production.clone(), base_manifest.clone())]),
            )
            .unwrap();
        let slot_result = execute(
            &slot_graph,
            &mut backend,
            BTreeMap::from([("slot-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let slot_production = slot_result.production_id.expect("slot production");
        let slot_manifest = store.manifest(&slot_production).unwrap().clone();
        let b0_ranges = compiler.chunks(compiler.b1_public_columns());
        assert!(b0_ranges.len() >= 10, "test must cross the chunk-9 name boundary");
        assert!(
            b0_ranges.iter().any(|range| range_len(range) < compiler.chunk_columns),
            "test must contain a nonmultiple tail chunk"
        );

        let imported_base = compiler
            .import_base(&BggSlotTransferBaseArtifacts { production_id: base_production.clone() })
            .unwrap();
        let imported_slots = compiler
            .import_slots(&BggSlotTransferSlotArtifacts { production_id: slot_production.clone() })
            .unwrap();
        let inspected_slot = 10;
        let mut inspect = DslContext::new("slot-import-inspection")
            .output("b0", imported_base.b0.public_matrix())
            .unwrap()
            .output("b1", imported_base.b1.public_matrix())
            .unwrap()
            .output("secret", imported_slots.secrets.get_static(inspected_slot))
            .unwrap()
            .output("public", imported_slots.public_keys.get_static(inspected_slot))
            .unwrap();
        for (chunk, family) in imported_slots.b0_preimage_chunks.iter().enumerate() {
            inspect = inspect
                .output(format!("b0-preimage-{chunk}"), family.get_static(inspected_slot))
                .unwrap();
        }
        for (chunk, family) in imported_slots.b1_preimage_chunks.iter().enumerate() {
            inspect = inspect
                .output(format!("b1-preimage-{chunk}"), family.get_static(inspected_slot))
                .unwrap();
        }
        let inspect = inspect
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([
                    (base_production.clone(), base_manifest.clone()),
                    (slot_production.clone(), slot_manifest.clone()),
                ]),
            )
            .unwrap();
        let inspected =
            execute(&inspect, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let b0 = matrix_output(&inspected, "b0");
        let b1 = matrix_output(&inspected, "b1");
        let secret = matrix_output(&inspected, "secret");
        let public = matrix_output(&inspected, "public");
        let expected_public = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
            &parameters,
            [0x42; 32],
            format!("slot_transfer_slot_a_{inspected_slot}"),
            compiler.secret_size,
            compiler.gadget_columns(),
            DistType::FinRingDist,
        );
        assert_eq!(public, &expected_public);
        let identity = DCRTPolyMatrix::identity(&parameters, compiler.secret_size, None);
        let secret_identity = secret.clone().concat_columns(&[&identity]);
        for (chunk, range) in b0_ranges.iter().enumerate() {
            let (start, end) = static_range(range);
            assert_eq!(
                b0.clone() * matrix_output(&inspected, &format!("b0-preimage-{chunk}")),
                secret_identity.clone() * &b1.slice_columns(start, end)
            );
        }
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
        for (chunk, range) in compiler.chunks(compiler.gadget_columns()).iter().enumerate() {
            let (start, end) = static_range(range);
            let expected = public
                .slice_columns(start, end)
                .concat_rows(&[&-(secret.clone() * &gadget.slice_columns(start, end))]);
            assert_eq!(
                b1.clone() * matrix_output(&inspected, &format!("b1-preimage-{chunk}")),
                expected
            );
        }

        let ring = compiler.ring();
        let base = compiler
            .import_base(&BggSlotTransferBaseArtifacts { production_id: base_production.clone() })
            .unwrap();
        let slots = compiler
            .import_slots(&BggSlotTransferSlotArtifacts { production_id: slot_production.clone() })
            .unwrap();
        let gate_hash = ring.bytes_input("gate-hash-key", 32);
        let input_key =
            ring.gadget(compiler.secret_size, compiler.gadget_base.clone(), compiler.digit_count);
        let transfer_output = ring.hash_matrix(
            gate_hash.clone(),
            b"slot_transfer_gate_a_out_7".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let reduce_output = ring.hash_matrix(
            gate_hash,
            b"slot_reduce_gate_a_out_8".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let requests = vec![
            BggSlotTransferGateRequest::Transfer {
                identity: "7".to_owned(),
                input_public_key: input_key.clone(),
                output_public_key: transfer_output.clone(),
                source_slots: vec![(1, None), (0, Some(3))],
            },
            BggSlotTransferGateRequest::Reduce {
                identity: "8".to_owned(),
                input_public_keys: vec![input_key.clone(), input_key.clone()],
                output_public_key: reduce_output.clone(),
                source_slot_count: 2,
            },
            BggSlotTransferGateRequest::Transfer {
                identity: "9".to_owned(),
                input_public_key: input_key.clone(),
                output_public_key: transfer_output.clone(),
                source_slots: Vec::new(),
            },
        ];
        let gates = compiler.build_gate_preimages(&base, &slots, &requests).unwrap();
        let gate_graph = compiler
            .export_gate_preimages(DslContext::new("slot-gate-production"), gates)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([
                    (base_production.clone(), base_manifest.clone()),
                    (slot_production.clone(), slot_manifest.clone()),
                ]),
            )
            .unwrap();
        let gate_result = execute(
            &gate_graph,
            &mut backend,
            BTreeMap::from([("gate-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let gate_production = gate_result.production_id.expect("gate production");
        let gate_manifest = store.manifest(&gate_production).unwrap().clone();
        for chunk in 0..compiler.chunks(compiler.gadget_columns()).len() {
            assert_eq!(
                gate_manifest.artifacts[&gate_preimage_name(false, "9", chunk)].family_count,
                Some(0)
            );
        }
        let invalid = BggSlotTransferGateRequest::Transfer {
            identity: "7".to_owned(),
            input_public_key: ring.identity(1),
            output_public_key: transfer_output.clone(),
            source_slots: vec![(0, None)],
        };
        assert!(matches!(
            compiler.import_gate_preimages(
                &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                &[invalid],
            ),
            Err(BggSlotTransferArtifactError::InvalidGateRequest)
        ));

        let imported_base = compiler
            .import_base(&BggSlotTransferBaseArtifacts { production_id: base_production.clone() })
            .unwrap();
        let imported_slots = compiler
            .import_slots(&BggSlotTransferSlotArtifacts { production_id: slot_production.clone() })
            .unwrap();
        let imported_gates = compiler
            .import_gate_preimages(
                &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                &requests,
            )
            .unwrap();
        let transfer_name = gate_preimage_name(false, "7", 0);
        let reduce_name = gate_preimage_name(true, "8", 0);
        let gate_hash = ring.bytes_input("gate-hash-key", 32);
        let transfer_output = ring.hash_matrix(
            gate_hash.clone(),
            b"slot_transfer_gate_a_out_7".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let reduce_output = ring.hash_matrix(
            gate_hash,
            b"slot_reduce_gate_a_out_8".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let consumer = DslContext::new("slot-gate-import-consumer")
            .output("b0", imported_base.b0.public_matrix())
            .unwrap()
            .output("secret-0", imported_slots.secrets.get_static(0))
            .unwrap()
            .output("secret-1", imported_slots.secrets.get_static(1))
            .unwrap()
            .output("public-0", imported_slots.public_keys.get_static(0))
            .unwrap()
            .output("transfer-output", transfer_output)
            .unwrap()
            .output("reduce-output", reduce_output)
            .unwrap()
            .output(
                "transfer-preimage",
                imported_gates.preimage_chunks[&transfer_name].get_static(0),
            )
            .unwrap()
            .output("reduce-preimage", imported_gates.preimage_chunks[&reduce_name].get_static(0))
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([
                    (base_production, base_manifest),
                    (slot_production, slot_manifest),
                    (gate_production, gate_manifest),
                ]),
            )
            .unwrap();
        let consumed = execute(
            &consumer,
            &mut backend,
            BTreeMap::from([("gate-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let (start, end) = static_range(&compiler.chunks(compiler.gadget_columns())[0]);
        let b0 = matrix_output(&consumed, "b0");
        let secret_0 = matrix_output(&consumed, "secret-0");
        let secret_1 = matrix_output(&consumed, "secret-1");
        let public_0 = matrix_output(&consumed, "public-0");
        let input_key = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
        let decomposed = public_0.slice_columns(start, end).decompose();
        let transfer_rhs = (secret_1.clone() * &input_key) * &decomposed;
        let transfer_expected = secret_0.clone() *
            &matrix_output(&consumed, "transfer-output").slice_columns(start, end) -
            &transfer_rhs;
        assert_eq!(b0.clone() * matrix_output(&consumed, "transfer-preimage"), transfer_expected);
        let mut reduce_rhs = DCRTPolyMatrix::zero(&parameters, compiler.secret_size, end - start);
        for (source, secret) in [secret_0, secret_1].into_iter().enumerate() {
            reduce_rhs = reduce_rhs +
                &(((secret.clone() * &input_key) * &decomposed) *
                    &DCRTPoly::const_rotate_poly(&parameters, source));
        }
        let reduce_expected = matrix_output(&consumed, "secret-0").clone() *
            &matrix_output(&consumed, "reduce-output").slice_columns(start, end) -
            &reduce_rhs;
        assert_eq!(b0.clone() * matrix_output(&consumed, "reduce-preimage"), reduce_expected);
    }

    #[test]
    fn base_and_slot_preprocessing_build_valid_graphs() {
        let compiler = compiler();
        let base = compiler.build_base().expect("base");
        compiler
            .export_base(DslContext::new("slot-base"), base.clone())
            .expect("base outputs")
            .build()
            .expect("base graph")
            .validate(&ParamEnv::default())
            .expect("valid base graph");
        let slots = compiler
            .build_slots(compiler.ring().bytes_input("hash-key", 32), &base)
            .expect("slots");
        let slot_graph = compiler
            .export_slots(DslContext::new("slot-preprocessing"), slots.clone())
            .expect("slot outputs")
            .build()
            .expect("slot graph");
        slot_graph.validate(&ParamEnv::default()).expect("valid slot graph");
        slot_graph.elaborate(&ParamEnv::default()).expect("symbolic slot graph");

        let key = compiler.ring().bytes_input("gate-hash-key", 32);
        let input = compiler.ring().hash_matrix(
            key.clone(),
            b"input".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let output = compiler.ring().hash_matrix(
            key,
            b"output".as_slice(),
            (compiler.secret_size, compiler.gadget_columns()),
        );
        let gates = compiler
            .build_gate_preimages(
                &base,
                &slots,
                &[
                    BggSlotTransferGateRequest::Transfer {
                        identity: "test".to_owned(),
                        input_public_key: input.clone(),
                        output_public_key: output.clone(),
                        source_slots: vec![(1, None), (0, Some(2))],
                    },
                    BggSlotTransferGateRequest::Transfer {
                        identity: "empty".to_owned(),
                        input_public_key: input,
                        output_public_key: output,
                        source_slots: Vec::new(),
                    },
                ],
            )
            .expect("gate preimages");
        let gate_graph = compiler
            .export_gate_preimages(DslContext::new("slot-gates"), gates)
            .expect("gate outputs")
            .build()
            .expect("gate graph");
        gate_graph.validate(&ParamEnv::default()).expect("valid gate graph");
        gate_graph.elaborate(&ParamEnv::default()).expect("symbolic gate graph");
    }
}
