//! Standalone WEE25 commitment trees expressed with the declarative DSL.

use mxx_dsl::{Bytes, DslContext, DslError, Family, HashTag, Mat, Ring};
use mxx_ir_core::{IntExpr, node::ConcatAxis};
use rayon::prelude::*;
use thiserror::Error;

const HASH_TAG_PREFIX: &[u8] = b"wee25_w_block_";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25CommitmentCompiler {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_size: usize,
    pub tree_base: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
}

#[derive(Debug, Error)]
pub enum Wee25CommitmentError {
    #[error("WEE25 dimensions and digit count must be nonzero and tree base must be at least two")]
    InvalidLayout,
    #[error("WEE25 block count must be a positive power of tree base and at least tree base")]
    InvalidBlockCount,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

#[derive(Clone)]
pub struct Wee25CommitmentTreeWire {
    pub root: Mat,
    /// Breadth-first order, matching the historical cache index.
    pub cached_nodes: Family<Mat>,
}

impl Wee25CommitmentCompiler {
    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }

    pub fn public_columns(&self) -> usize {
        self.secret_size * (2 + self.digit_count)
    }

    pub fn gadget_rows(&self) -> usize {
        self.secret_size * self.digit_count
    }

    pub fn matrix_type(&self, rows: usize, columns: usize) -> mxx_ir_core::types::MatrixType {
        self.ring().matrix_type((rows, columns))
    }

    pub fn block_type(&self) -> mxx_ir_core::types::MatrixType {
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

    pub fn commitment(
        &self,
        hash_key: Bytes,
        message_blocks: &[Mat],
    ) -> Result<Mat, Wee25CommitmentError> {
        Ok(self.commitment_tree(hash_key, message_blocks)?.root)
    }

    pub fn commitment_tree(
        &self,
        hash_key: Bytes,
        message_blocks: &[Mat],
    ) -> Result<Wee25CommitmentTreeWire, Wee25CommitmentError> {
        self.validate_block_count(message_blocks.len())?;
        if message_blocks.par_iter().any(|block| block.matrix_type() != &self.block_type()) {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        let (root, mut nodes) =
            self.commit_level(hash_key, message_blocks, 0, message_blocks.len())?;
        nodes.par_sort_unstable_by_key(|(index, _)| *index);
        let cached_nodes = Family::pack(nodes.into_iter().map(|(_, node)| node).collect())?;
        Ok(Wee25CommitmentTreeWire { root, cached_nodes })
    }

    pub fn export_commitment_tree(
        &self,
        context: DslContext,
        tree: Wee25CommitmentTreeWire,
    ) -> Result<DslContext, DslError> {
        context
            .public_output("wee25_commitment", tree.root)?
            .public_family_output("wee25_commitment_nodes", tree.cached_nodes)
    }

    fn commit_level(
        &self,
        hash_key: Bytes,
        blocks: &[Mat],
        offset: usize,
        total_count: usize,
    ) -> Result<(Mat, Vec<(usize, Mat)>), Wee25CommitmentError> {
        if blocks.len() == self.tree_base {
            let commitment = self.base_commitment(hash_key, blocks)?;
            return Ok((
                commitment.clone(),
                vec![(self.cache_node_index(total_count, offset, blocks.len()), commitment)],
            ));
        }
        let child_len = blocks.len() / self.tree_base;
        let subtrees = blocks
            .chunks(child_len)
            .enumerate()
            .map(|(child, blocks)| {
                self.commit_level(hash_key.clone(), blocks, offset + child * child_len, total_count)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let children = subtrees.iter().map(|(root, _)| root.clone()).collect::<Vec<_>>();
        let commitment = self.base_commitment(hash_key, &children)?;
        let mut nodes = subtrees.into_iter().flat_map(|(_, nodes)| nodes).collect::<Vec<_>>();
        nodes.push((self.cache_node_index(total_count, offset, blocks.len()), commitment.clone()));
        Ok((commitment, nodes))
    }

    fn base_commitment(
        &self,
        hash_key: Bytes,
        blocks: &[Mat],
    ) -> Result<Mat, Wee25CommitmentError> {
        if blocks.len() != self.tree_base {
            return Err(Wee25CommitmentError::InvalidBlockCount);
        }
        let columns = self.tree_base * self.public_columns();
        let message = Mat::concat(ConcatAxis::Columns, blocks.to_vec());
        let decomposition = message.decompose(self.gadget_base.clone(), self.digit_count);
        let terms = (0..columns * self.gadget_rows())
            .map(|index| {
                let column = index / self.gadget_rows();
                let digit_row = index % self.gadget_rows();
                let mut tag = HashTag::from(HASH_TAG_PREFIX);
                tag.push(IntExpr::constant(index));
                let w = self.ring().hash_matrix(
                    hash_key.clone(),
                    tag,
                    (self.secret_size, self.public_columns()),
                );
                let digit = decomposition.entry(digit_row, column);
                w * digit
            })
            .collect::<Vec<_>>();
        Ok(terms
            .into_iter()
            .reduce(|sum, term| sum + term)
            .unwrap_or_else(|| self.ring().zero((self.secret_size, self.public_columns()))))
    }

    pub fn cache_node_count(&self, block_count: usize) -> usize {
        let mut level_nodes = 1;
        let mut level_length = block_count;
        let mut total = 0;
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
        let mut level_nodes = 1;
        let mut before = 0;
        while level_length > length {
            before += level_nodes;
            level_nodes *= self.tree_base;
            level_length /= self.tree_base;
        }
        before + offset / length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use keccak_asm::Keccak256;
    use mxx_dsl::Parallel;
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
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn direct_base(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        key: [u8; 32],
        blocks: &[DCRTPolyMatrix],
    ) -> DCRTPolyMatrix {
        let refs = blocks.iter().collect::<Vec<_>>();
        let message = blocks[0].concat_columns(&refs[1..]);
        let sampler = DCRTPolyHashSampler::<Keccak256>::new();
        let mut result =
            DCRTPolyMatrix::zero(parameters, compiler.secret_size, compiler.public_columns());
        for column in 0..message.col_size() {
            let decomposed = message.get_column_matrix_decompose(column);
            for digit_row in 0..compiler.gadget_rows() {
                let index = column * compiler.gadget_rows() + digit_row;
                let mut tag = HASH_TAG_PREFIX.to_vec();
                tag.extend_from_slice(&index.to_le_bytes());
                let w = sampler.sample_hash(
                    parameters,
                    key,
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
    fn commitment_tree_is_composable_inside_parallel_body() {
        let compiler = Wee25CommitmentCompiler {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_size: 1,
            tree_base: 2,
            digit_count: 2,
            gadget_base: 4.into(),
        };
        let ring = compiler.ring();
        let hash_key = ring.bytes_input("hash-key", 32);
        let blocks = (0..2)
            .map(|index| ring.input(format!("block-{index}"), (1, compiler.public_columns())))
            .collect::<Vec<_>>();
        let roots = Parallel::range(2)
            .map(move |_| {
                compiler
                    .commitment_tree(hash_key.clone(), &blocks)
                    .expect("commitment in parallel body")
                    .root
            })
            .expect("parallel family");
        let built = DslContext::new("wee25-parallel-composition")
            .family_output("roots", roots)
            .expect("family output")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("validate");
        built.validate(&ParamEnv::default()).expect("elaborate");
    }

    #[test]
    fn rejects_non_power_block_count() {
        let compiler = Wee25CommitmentCompiler {
            modulus: 17.into(),
            ring_dimension: 4.into(),
            secret_size: 1,
            tree_base: 2,
            digit_count: 3,
            gadget_base: 2.into(),
        };
        assert!(matches!(
            compiler.validate_block_count(3),
            Err(Wee25CommitmentError::InvalidBlockCount)
        ));
    }

    #[test]
    #[serial_test::serial]
    fn commitment_root_and_cache_order_match_the_concrete_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let compiler = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            tree_base: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        };
        let key = [0x37; 32];
        let ring = compiler.ring();
        let mut inputs = BTreeMap::from([(
            "hash-key".to_owned(),
            RuntimeValue::<CpuDcrtBackend>::Bytes(key.to_vec()),
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
                let name = format!("block-{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                (ring.input(name, (1, compiler.public_columns())), value)
            })
            .collect::<Vec<_>>();
        let tree = compiler
            .commitment_tree(
                ring.bytes_input("hash-key", 32),
                &blocks.iter().map(|(wire, _)| wire.clone()).collect::<Vec<_>>(),
            )
            .unwrap();
        let mut context = DslContext::new("wee25-parity").output("root", tree.root).unwrap();
        for index in 0..compiler.cache_node_count(4) {
            context = context
                .output(format!("cache-{index}"), tree.cached_nodes.get_static(index))
                .unwrap();
        }
        let built = context.build().unwrap();
        let validated = built.validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let values = blocks.into_iter().map(|(_, value)| value).collect::<Vec<_>>();
        let left = direct_base(&compiler, &parameters, key, &values[..2]);
        let right = direct_base(&compiler, &parameters, key, &values[2..]);
        let root = direct_base(&compiler, &parameters, key, &[left.clone(), right.clone()]);
        for (name, expected) in
            [("root", &root), ("cache-0", &root), ("cache-1", &left), ("cache-2", &right)]
        {
            let RuntimeValue::Matrix(actual) = &result.outputs[name] else { panic!("matrix") };
            assert_eq!(actual.as_ref(), expected, "{name}");
        }
    }
}
