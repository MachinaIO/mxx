use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, OutputFamilyError, SubgraphBuildError,
    WireRef,
    artifact::ArtifactConfidentiality,
    node::{ConcatAxis, ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp},
    types::MatrixType,
};
use thiserror::Error;

const HASH_TAG_PREFIX: &[u8] = b"wee25_w_block_";

/// Graph-IR compiler for the standalone WEE25 commitment tree.
///
/// This compiler deliberately does not implement the excluded WEE25-backed
/// lookup evaluator. Message blocks and every internal commitment have shape
/// `secret_size × public_columns`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25CommitmentCompiler {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_size: usize,
    pub tree_base: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum Wee25CommitmentError {
    #[error(
        "WEE25 dimensions, digit count, and tree base must be nonzero, with tree base at least two"
    )]
    InvalidLayout,
    #[error(
        "WEE25 message block count must be a positive power of tree_base and at least tree_base"
    )]
    InvalidBlockCount,
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25CommitmentTreeWire {
    pub root: MatrixWire,
    /// Breadth-first tree order, identical to the legacy `CommitCache::node_index`.
    pub cached_nodes: MatrixFamilyWire,
}

impl Wee25CommitmentCompiler {
    pub fn public_columns(&self) -> usize {
        self.secret_size * (2 + self.digit_count)
    }

    pub fn gadget_rows(&self) -> usize {
        self.secret_size * self.digit_count
    }

    pub fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    pub fn block_type(&self) -> MatrixType {
        self.matrix_type(self.secret_size, self.public_columns())
    }

    pub fn validate_layout(&self) -> Result<(), Wee25CommitmentError> {
        if self.secret_size == 0 || self.digit_count == 0 || self.tree_base < 2 {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        Ok(())
    }

    pub fn validate_block_count(&self, count: usize) -> Result<(), Wee25CommitmentError> {
        self.validate_layout()?;
        if count < self.tree_base {
            return Err(Wee25CommitmentError::InvalidBlockCount);
        }
        let mut remaining = count;
        while remaining > self.tree_base {
            if !remaining.is_multiple_of(self.tree_base) {
                return Err(Wee25CommitmentError::InvalidBlockCount);
            }
            remaining /= self.tree_base;
        }
        (remaining == self.tree_base).then_some(()).ok_or(Wee25CommitmentError::InvalidBlockCount)
    }

    /// Builds the WEE25 commitment root while preserving the legacy tree order.
    pub fn commitment(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        message_blocks: &[MatrixWire],
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        Ok(self.commitment_tree(builder, hash_key, message_blocks)?.root)
    }

    pub fn commitment_tree(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        message_blocks: &[MatrixWire],
    ) -> Result<Wee25CommitmentTreeWire, Wee25CommitmentError> {
        self.validate_block_count(message_blocks.len())?;
        if message_blocks.iter().any(|block| block.matrix_type != self.block_type()) {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        let mut nodes = Vec::with_capacity(self.cache_node_count(message_blocks.len()));
        let root = self.commit_level(
            builder,
            hash_key,
            message_blocks,
            0,
            message_blocks.len(),
            &mut nodes,
        )?;
        nodes.sort_by_key(|(index, _)| *index);
        let cached_nodes =
            builder.family_pack(&nodes.into_iter().map(|(_, node)| node).collect::<Vec<_>>())?;
        Ok(Wee25CommitmentTreeWire { root, cached_nodes })
    }

    pub fn export_commitment_tree(
        &self,
        builder: &mut GraphBuilder,
        tree: &Wee25CommitmentTreeWire,
    ) {
        builder.output("wee25_commitment", &tree.root, ArtifactConfidentiality::Public);
        builder.output_family_wire(
            "wee25_commitment_nodes",
            &tree.cached_nodes,
            ArtifactConfidentiality::Public,
        );
    }

    fn commit_level(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        blocks: &[MatrixWire],
        offset: usize,
        total_count: usize,
        nodes: &mut Vec<(usize, MatrixWire)>,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        if blocks.len() == self.tree_base {
            let commitment = self.call_base(builder, hash_key, blocks)?;
            nodes.push((
                self.cache_node_index(total_count, offset, blocks.len()),
                commitment.clone(),
            ));
            return Ok(commitment);
        }
        let child_len = blocks.len() / self.tree_base;
        let children = blocks
            .chunks(child_len)
            .enumerate()
            .map(|(child_index, child)| {
                self.commit_level(
                    builder,
                    hash_key,
                    child,
                    offset + child_index * child_len,
                    total_count,
                    nodes,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let commitment = self.call_base(builder, hash_key, &children)?;
        nodes.push((self.cache_node_index(total_count, offset, blocks.len()), commitment.clone()));
        Ok(commitment)
    }

    pub fn cache_node_count(&self, block_count: usize) -> usize {
        let mut level_nodes = 1usize;
        let mut level_length = block_count;
        let mut total = 0usize;
        while level_length >= self.tree_base {
            total += level_nodes;
            if level_length == self.tree_base {
                break;
            }
            level_length /= self.tree_base;
            level_nodes *= self.tree_base;
        }
        total
    }

    pub fn cache_node_index(&self, total_count: usize, offset: usize, length: usize) -> usize {
        let mut level_length = total_count;
        let mut level_nodes = 1usize;
        let mut nodes_before_level = 0usize;
        while level_length > length {
            nodes_before_level += level_nodes;
            level_nodes *= self.tree_base;
            level_length /= self.tree_base;
        }
        nodes_before_level + offset / length
    }

    fn call_base(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        blocks: &[MatrixWire],
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        debug_assert_eq!(blocks.len(), self.tree_base);
        let mut body = GraphBuilder::new(
            format!(
                "wee25-commit-base-d{}-b{}-k{}",
                self.secret_size, self.tree_base, self.digit_count
            ),
            Vec::new(),
        );
        let body_hash_key = body.bytes_input("0_hash_key", 32);
        let body_message = body.input(
            "1_message",
            self.matrix_type(self.secret_size, self.tree_base * self.public_columns()),
        );
        let output = self.base_commitment(&mut body, body_hash_key, &body_message);
        body.value_output_wire("0_commitment", output.wire);

        let message = builder.concat(
            ConcatAxis::Columns,
            blocks,
            self.matrix_type(self.secret_size, self.tree_base * self.public_columns()),
        );
        Ok(builder
            .subgraph_call(body.finish(), vec![hash_key, message.wire], &[self.block_type()])?
            .remove(0))
    }

    fn base_commitment(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        message: &MatrixWire,
    ) -> MatrixWire {
        let base_columns = self.tree_base * self.public_columns();
        let decomposition = builder.gadget_decompose_with_layout(
            message,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(self.gadget_rows(), base_columns),
        );
        let mut commitment = builder.constant_matrix(self.block_type(), ConstantMatrix::Zero);
        for column in 0..base_columns {
            for digit_row in 0..self.gadget_rows() {
                let block_index = column * self.gadget_rows() + digit_row;
                let w_block = builder.hash_sample_with_encoded_tags(
                    hash_key,
                    self.block_type(),
                    HashVariant::Plain,
                    HASH_TAG_PREFIX.to_vec(),
                    Vec::new(),
                    Vec::new(),
                    vec![IntExpr::constant(block_index)],
                    None,
                    None,
                );
                let digit = builder.slice(
                    &decomposition,
                    Some(IndexRange { start: digit_row, end: digit_row + 1 }),
                    Some(IndexRange { start: column, end: column + 1 }),
                    self.matrix_type(1, 1),
                );
                let term = builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &w_block,
                    &digit,
                    self.block_type(),
                );
                commitment = builder.matrix_binary(
                    MatrixBinaryOp::Add,
                    &commitment,
                    &term,
                    self.block_type(),
                );
            }
        }
        commitment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality, validate};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    type HashSampler = DCRTPolyHashSampler<keccak_asm::Keccak256>;

    fn direct_base(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        hash_key: [u8; 32],
        blocks: &[DCRTPolyMatrix],
    ) -> DCRTPolyMatrix {
        let refs = blocks.iter().collect::<Vec<_>>();
        let message = blocks[0].concat_columns(&refs[1..]);
        let sampler = HashSampler::new();
        let mut result =
            DCRTPolyMatrix::zero(parameters, compiler.secret_size, compiler.public_columns());
        for column in 0..message.col_size() {
            let decomposed = message.get_column_matrix_decompose(column);
            for digit_row in 0..compiler.gadget_rows() {
                let block_index = column * compiler.gadget_rows() + digit_row;
                let mut tag = HASH_TAG_PREFIX.to_vec();
                tag.extend_from_slice(&block_index.to_le_bytes());
                let w = sampler.sample_hash(
                    parameters,
                    hash_key,
                    tag,
                    compiler.secret_size,
                    compiler.public_columns(),
                    DistType::FinRingDist,
                );
                result = result + &(w * decomposed.entry(digit_row, 0));
            }
        }
        result
    }

    #[test]
    #[serial_test::serial]
    fn commitment_tree_matches_legacy_formula_exactly() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let compiler = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            tree_base: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        };
        let hash_key = [0x37; 32];
        let mut builder = GraphBuilder::new("wee25-commitment-tree-test", Vec::new());
        let hash_key_wire = builder.bytes_input("hash_key", 32);
        let mut inputs = BTreeMap::from([(
            "hash_key".to_owned(),
            RuntimeValue::<CpuDcrtBackend>::Bytes(hash_key.to_vec()),
        )]);
        let blocks = (0..4)
            .map(|index| {
                let value = DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![
                        (0..compiler.public_columns())
                            .map(|column| {
                                DCRTPoly::from_usize_to_constant(
                                    &parameters,
                                    1 + index * compiler.public_columns() + column,
                                )
                            })
                            .collect(),
                    ],
                );
                let name = format!("message_{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                (builder.input(name, compiler.block_type()), value)
            })
            .collect::<Vec<_>>();
        let block_wires = blocks.iter().map(|(wire, _)| wire.clone()).collect::<Vec<_>>();
        let commitment = compiler
            .commitment(&mut builder, hash_key_wire, &block_wires)
            .expect("valid commitment tree");
        builder.output("commitment", &commitment, ArtifactConfidentiality::Public);
        let graph = validate(&builder.finish(), &ParamEnv::default()).expect("valid graph");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("runtime execution");

        let values = blocks.into_iter().map(|(_, value)| value).collect::<Vec<_>>();
        let left = direct_base(&compiler, &parameters, hash_key, &values[..2]);
        let right = direct_base(&compiler, &parameters, hash_key, &values[2..]);
        let expected = direct_base(&compiler, &parameters, hash_key, &[left, right]);
        let RuntimeValue::Matrix(actual) = &result.outputs["commitment"] else {
            panic!("matrix commitment output");
        };
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    #[serial_test::serial]
    fn rejects_non_power_block_count() {
        let compiler = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(4),
            secret_size: 1,
            tree_base: 2,
            digit_count: 3,
            gadget_base: IntExpr::constant(2),
        };
        assert_eq!(compiler.validate_block_count(3), Err(Wee25CommitmentError::InvalidBlockCount));
    }
}
