//! Standalone WEE25 opening and verification graphs.

use crate::{Wee25CommitmentCompiler, Wee25CommitmentError};
use mxx_dsl::{Family, Mat};
use mxx_ir_core::{
    IntExpr,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, IndexRange},
};
use rayon::prelude::*;
use std::{collections::BTreeMap, ops::Range};

pub const WEE25_PUBLIC_B: &str = "wee25_public_b";
pub const WEE25_T_TOP: &str = "wee25_t_top";
pub const WEE25_T_BOTTOM: &str = "wee25_t_bottom";
pub const WEE25_COMMITMENT: &str = "wee25_commitment";
pub const WEE25_COMMITMENT_NODES: &str = "wee25_commitment_nodes";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25PublicParameterArtifacts {
    pub production_id: ProductionId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25CommitmentArtifacts {
    pub production_id: ProductionId,
    pub block_count: usize,
}

#[derive(Clone)]
pub struct Wee25PublicParameterWires {
    pub b: Mat,
    /// Ordered by `(digit_row, column_part)`.
    pub t_top: Vec<Family<Mat>>,
    pub t_bottom: Family<Mat>,
}

#[derive(Clone)]
pub struct Wee25VerificationWire {
    /// Valid exactly when every entry is zero.
    pub residual: Mat,
}

impl Wee25CommitmentCompiler {
    pub fn public_parameter_part_count(&self) -> usize {
        self.tree_base * self.digit_count
    }

    pub fn public_parameter_top_count(&self) -> usize {
        self.public_parameter_top_family_count() * self.public_parameter_block_count()
    }

    pub fn public_parameter_top_family_count(&self) -> usize {
        self.gadget_rows() * self.public_parameter_part_count()
    }

    pub fn public_parameter_block_count(&self) -> usize {
        self.tree_base * self.public_columns()
    }

    pub fn public_parameter_top_name(&self, digit_row: usize, part: usize) -> String {
        format!("{WEE25_T_TOP}_digit_{digit_row}_part_{part}")
    }

    pub fn import_public_parameters(
        &self,
        artifacts: &Wee25PublicParameterArtifacts,
    ) -> Result<Wee25PublicParameterWires, Wee25CommitmentError> {
        self.validate_layout()?;
        let ring = self.ring();
        let part_count = self.public_parameter_part_count();
        let t_top = (0..self.public_parameter_top_family_count())
            .map(|index| {
                let name = self.public_parameter_top_name(index / part_count, index % part_count);
                ring.family_artifact_input(
                    artifacts.production_id.clone(),
                    name,
                    self.public_parameter_block_count(),
                    (self.public_columns(), self.public_columns()),
                    ArtifactConfidentiality::Public,
                )
            })
            .collect();
        Ok(Wee25PublicParameterWires {
            b: ring.artifact_input(
                artifacts.production_id.clone(),
                WEE25_PUBLIC_B,
                (self.secret_size, self.public_columns()),
                ArtifactConfidentiality::Public,
            ),
            t_top,
            t_bottom: ring.family_artifact_input(
                artifacts.production_id.clone(),
                WEE25_T_BOTTOM,
                part_count,
                (self.public_columns(), self.public_columns()),
                ArtifactConfidentiality::Public,
            ),
        })
    }

    pub fn import_commitment_artifacts(
        &self,
        artifacts: &Wee25CommitmentArtifacts,
    ) -> Result<(Mat, Family<Mat>), Wee25CommitmentError> {
        self.validate_block_count(artifacts.block_count)?;
        let ring = self.ring();
        Ok((
            ring.artifact_input(
                artifacts.production_id.clone(),
                WEE25_COMMITMENT,
                (self.secret_size, self.public_columns()),
                ArtifactConfidentiality::Public,
            ),
            ring.family_artifact_input(
                artifacts.production_id.clone(),
                WEE25_COMMITMENT_NODES,
                self.cache_node_count(artifacts.block_count),
                (self.secret_size, self.public_columns()),
                ArtifactConfidentiality::Public,
            ),
        ))
    }

    pub fn opening(
        &self,
        message_blocks: &[Mat],
        range: Option<Range<usize>>,
        parameters: &Wee25PublicParameterWires,
        commitment_nodes: &Family<Mat>,
    ) -> Result<Mat, Wee25CommitmentError> {
        self.validate_opening_inputs(message_blocks, commitment_nodes)?;
        self.validate_public_parameter_wires(parameters)?;
        let range = checked_range(range, message_blocks.len())?;
        let base = self.verifier_base(parameters, false);
        let base_last = self.verifier_base(parameters, true);
        let mut z_cache = BTreeMap::new();
        let mut verifier_cache = BTreeMap::new();
        let openings = range
            .map(|column| {
                self.open_recursive(
                    message_blocks,
                    0,
                    message_blocks.len(),
                    column,
                    parameters,
                    commitment_nodes,
                    &base,
                    &base_last,
                    &mut z_cache,
                    &mut verifier_cache,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(concat_columns(openings))
    }

    pub fn verifier(
        &self,
        block_count: usize,
        range: Option<Range<usize>>,
        parameters: &Wee25PublicParameterWires,
    ) -> Result<Mat, Wee25CommitmentError> {
        self.validate_block_count(block_count)?;
        self.validate_public_parameter_wires(parameters)?;
        let range = checked_range(range, block_count)?;
        let base = self.verifier_base(parameters, false);
        let base_last = self.verifier_base(parameters, true);
        let mut cache = BTreeMap::new();
        let columns = range
            .map(|column| {
                self.verifier_recursive(&base, &base_last, block_count, column, &mut cache)
            })
            .collect();
        Ok(concat_columns(columns))
    }

    pub fn verification_residual(
        &self,
        message_blocks: &[Mat],
        commitment: &Mat,
        opening: &Mat,
        range: Option<Range<usize>>,
        parameters: &Wee25PublicParameterWires,
    ) -> Result<Wee25VerificationWire, Wee25CommitmentError> {
        self.validate_block_count(message_blocks.len())?;
        let range = checked_range(range, message_blocks.len())?;
        let verifier = self.verifier(message_blocks.len(), Some(range.clone()), parameters)?;
        let message = concat_columns(message_blocks[range].to_vec());
        Ok(Wee25VerificationWire {
            residual: commitment.clone() * verifier -
                (message - parameters.b.clone() * opening.clone()),
        })
    }

    fn validate_opening_inputs(
        &self,
        blocks: &[Mat],
        nodes: &Family<Mat>,
    ) -> Result<(), Wee25CommitmentError> {
        self.validate_block_count(blocks.len())?;
        if blocks.par_iter().any(|block| block.matrix_type() != &self.block_type()) ||
            nodes.element_type() != &self.block_type() ||
            nodes.count() != &IntExpr::constant(self.cache_node_count(blocks.len()))
        {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        Ok(())
    }

    fn validate_public_parameter_wires(
        &self,
        parameters: &Wee25PublicParameterWires,
    ) -> Result<(), Wee25CommitmentError> {
        let part_type = self.matrix_type(self.public_columns(), self.public_columns());
        if parameters.b.matrix_type() != &self.block_type() ||
            parameters.t_top.len() != self.public_parameter_top_family_count() ||
            parameters.t_top.par_iter().any(|family| {
                family.element_type() != &part_type ||
                    family.count() != &IntExpr::constant(self.public_parameter_block_count())
            }) ||
            parameters.t_bottom.element_type() != &part_type ||
            parameters.t_bottom.count() != &IntExpr::constant(self.public_parameter_part_count())
        {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn open_recursive(
        &self,
        blocks: &[Mat],
        offset: usize,
        total: usize,
        column: usize,
        parameters: &Wee25PublicParameterWires,
        nodes: &Family<Mat>,
        verifier_base: &Mat,
        verifier_base_last: &Mat,
        z_cache: &mut BTreeMap<(usize, usize, usize), Mat>,
        verifier_cache: &mut BTreeMap<(usize, usize), Mat>,
    ) -> Result<Mat, Wee25CommitmentError> {
        if blocks.len() == self.tree_base {
            return Ok(self.open_base(&concat_columns(blocks.to_vec()), column, parameters, true));
        }
        let child_count = blocks.len() / self.tree_base;
        let child_column = column % child_count;
        let sibling = column / child_count;
        let commitments = (0..self.tree_base)
            .map(|child| {
                nodes.get_static(self.cache_node_index(
                    total,
                    offset + child * child_count,
                    child_count,
                ))
            })
            .collect();
        let key = (offset, blocks.len(), sibling);
        let z_prime = if let Some(value) = z_cache.get(&key) {
            value.clone()
        } else {
            let value = self.open_base(&concat_columns(commitments), sibling, parameters, false);
            z_cache.insert(key, value.clone());
            value
        };
        let start = sibling * child_count;
        let z_child = self.open_recursive(
            &blocks[start..start + child_count],
            offset + start,
            total,
            child_column,
            parameters,
            nodes,
            verifier_base,
            verifier_base_last,
            z_cache,
            verifier_cache,
        )?;
        let verifier = self.verifier_recursive(
            verifier_base,
            verifier_base_last,
            child_count,
            child_column,
            verifier_cache,
        );
        Ok(z_prime * verifier.decompose(self.gadget_base.clone(), self.digit_count).as_mat() +
            z_child)
    }

    fn open_base(
        &self,
        message: &Mat,
        column: usize,
        parameters: &Wee25PublicParameterWires,
        leaf: bool,
    ) -> Mat {
        let base_columns = self.tree_base * self.public_columns();
        let width = self.public_columns() * self.digit_count;
        let decomposition =
            message.clone().decompose(self.gadget_base.clone(), self.digit_count).as_mat();
        let terms = (0..base_columns * self.gadget_rows())
            .map(|index| {
                let message_column = index / self.gadget_rows();
                let digit_row = index % self.gadget_rows();
                let chunks = (0..self.digit_count)
                    .map(|digit| {
                        parameters.t_top[digit_row * self.public_parameter_part_count() +
                            column * self.digit_count +
                            digit]
                            .get_static(message_column)
                    })
                    .collect();
                let scalar = decomposition.clone().slice(
                    Some(IndexRange { start: digit_row.into(), end: (digit_row + 1).into() }),
                    Some(IndexRange {
                        start: message_column.into(),
                        end: (message_column + 1).into(),
                    }),
                );
                concat_columns(chunks) * scalar
            })
            .collect::<Vec<_>>();
        let opening = terms
            .into_iter()
            .reduce(|sum, term| sum + term)
            .unwrap_or_else(|| self.ring().zero((self.public_columns(), width)));
        if !leaf {
            return opening;
        }
        opening *
            self.ring()
                .identity(self.public_columns())
                .decompose(self.gadget_base.clone(), self.digit_count)
                .as_mat()
    }

    fn verifier_base(&self, parameters: &Wee25PublicParameterWires, leaf: bool) -> Mat {
        let chunks = (0..self.public_parameter_part_count())
            .map(|part| parameters.t_bottom.get_static(part))
            .collect();
        let bottom = concat_columns(chunks);
        if !leaf {
            return bottom;
        }
        let columns = self.tree_base * self.public_columns();
        bottom *
            self.ring()
                .identity(columns)
                .decompose(self.gadget_base.clone(), self.digit_count)
                .as_mat()
    }

    fn verifier_recursive(
        &self,
        base: &Mat,
        base_last: &Mat,
        block_count: usize,
        column: usize,
        cache: &mut BTreeMap<(usize, usize), Mat>,
    ) -> Mat {
        if let Some(value) = cache.get(&(block_count, column)) {
            return value.clone();
        }
        let result = if block_count == self.tree_base {
            base_last.clone().slice(
                None,
                Some(IndexRange {
                    start: (self.public_columns() * column).into(),
                    end: (self.public_columns() * (column + 1)).into(),
                }),
            )
        } else {
            let child_count = block_count / self.tree_base;
            let child =
                self.verifier_recursive(base, base_last, child_count, column % child_count, cache);
            let sibling = column / child_count;
            let width = self.public_columns() * self.digit_count;
            base.clone().slice(
                None,
                Some(IndexRange {
                    start: (width * sibling).into(),
                    end: (width * (sibling + 1)).into(),
                }),
            ) * child.decompose(self.gadget_base.clone(), self.digit_count).as_mat()
        };
        cache.insert((block_count, column), result.clone());
        result
    }
}

fn concat_columns(values: Vec<Mat>) -> Mat {
    if values.len() == 1 { values[0].clone() } else { Mat::concat(ConcatAxis::Columns, values) }
}

fn checked_range(
    range: Option<Range<usize>>,
    count: usize,
) -> Result<Range<usize>, Wee25CommitmentError> {
    let range = range.unwrap_or(0..count);
    if range.start >= range.end || range.end > count {
        return Err(Wee25CommitmentError::InvalidBlockCount);
    }
    Ok(range)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Wee25PublicParameterCompiler;
    use mxx_dsl::DslContext;
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
    use std::collections::{BTreeMap, HashMap};

    type HashSampler = DCRTPolyHashSampler<keccak_asm::Keccak256>;

    fn matrix(
        parameters: &DCRTPolyParams,
        rows: usize,
        columns: usize,
        seed: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            (0..rows)
                .map(|row| {
                    (0..columns)
                        .map(|column| {
                            DCRTPoly::from_usize_to_constant(
                                parameters,
                                seed + row * columns + column + 1,
                            )
                        })
                        .collect()
                })
                .collect(),
        )
    }

    fn concat(values: &[DCRTPolyMatrix]) -> DCRTPolyMatrix {
        values[0].concat_columns(&values[1..].iter().collect::<Vec<_>>())
    }

    fn direct_commit_base(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        hash_key: [u8; 32],
        blocks: &[DCRTPolyMatrix],
    ) -> DCRTPolyMatrix {
        let message = concat(blocks);
        let sampler = HashSampler::new();
        let mut result =
            DCRTPolyMatrix::zero(parameters, compiler.secret_size, compiler.public_columns());
        for column in 0..message.col_size() {
            let decomposed = message.get_column_matrix_decompose(column);
            for digit_row in 0..compiler.gadget_rows() {
                let block_index = column * compiler.gadget_rows() + digit_row;
                let mut tag = b"wee25_w_block_".to_vec();
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

    fn direct_commit_tree(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        hash_key: [u8; 32],
        blocks: &[DCRTPolyMatrix],
        offset: usize,
        cache: &mut HashMap<(usize, usize), DCRTPolyMatrix>,
    ) -> DCRTPolyMatrix {
        let commitment = if blocks.len() == compiler.tree_base {
            direct_commit_base(compiler, parameters, hash_key, blocks)
        } else {
            let child_count = blocks.len() / compiler.tree_base;
            let children = blocks
                .chunks(child_count)
                .enumerate()
                .map(|(child, values)| {
                    direct_commit_tree(
                        compiler,
                        parameters,
                        hash_key,
                        values,
                        offset + child * child_count,
                        cache,
                    )
                })
                .collect::<Vec<_>>();
            direct_commit_base(compiler, parameters, hash_key, &children)
        };
        cache.insert((offset, blocks.len()), commitment.clone());
        commitment
    }

    fn direct_open_base(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        message: &DCRTPolyMatrix,
        column: usize,
        t_top: &[DCRTPolyMatrix],
        leaf: bool,
    ) -> DCRTPolyMatrix {
        let width = compiler.public_columns() * compiler.digit_count;
        let mut result = DCRTPolyMatrix::zero(parameters, compiler.public_columns(), width);
        for message_column in 0..message.col_size() {
            let decomposed = message.get_column_matrix_decompose(message_column);
            for digit_row in 0..compiler.gadget_rows() {
                let part = message_column * compiler.gadget_rows() + digit_row;
                let chunks = (0..compiler.digit_count)
                    .map(|digit| {
                        &t_top[part * compiler.public_parameter_part_count() +
                            column * compiler.digit_count +
                            digit]
                    })
                    .collect::<Vec<_>>();
                let joined = chunks[0].concat_columns(&chunks[1..]);
                result = result + &(joined * decomposed.entry(digit_row, 0));
            }
        }
        if leaf {
            result *
                DCRTPolyMatrix::identity(parameters, compiler.public_columns(), None).decompose()
        } else {
            result
        }
    }

    fn direct_verifier_base(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        t_bottom: &[DCRTPolyMatrix],
        leaf: bool,
    ) -> DCRTPolyMatrix {
        let bottom = concat(t_bottom);
        if leaf {
            let size = compiler.tree_base * compiler.public_columns();
            bottom * DCRTPolyMatrix::identity(parameters, size, None).decompose()
        } else {
            bottom
        }
    }

    fn direct_verifier_recursive(
        compiler: &Wee25CommitmentCompiler,
        base: &DCRTPolyMatrix,
        base_last: &DCRTPolyMatrix,
        count: usize,
        column: usize,
    ) -> DCRTPolyMatrix {
        if count == compiler.tree_base {
            return base_last.slice_columns(
                compiler.public_columns() * column,
                compiler.public_columns() * (column + 1),
            );
        }
        let child_count = count / compiler.tree_base;
        let child =
            direct_verifier_recursive(compiler, base, base_last, child_count, column % child_count);
        let sibling = column / child_count;
        let width = compiler.public_columns() * compiler.digit_count;
        base.slice_columns(width * sibling, width * (sibling + 1)) * child.decompose()
    }

    #[allow(clippy::too_many_arguments)]
    fn direct_open_recursive(
        compiler: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        blocks: &[DCRTPolyMatrix],
        offset: usize,
        column: usize,
        t_top: &[DCRTPolyMatrix],
        cache: &HashMap<(usize, usize), DCRTPolyMatrix>,
        verifier_base: &DCRTPolyMatrix,
        verifier_base_last: &DCRTPolyMatrix,
    ) -> DCRTPolyMatrix {
        if blocks.len() == compiler.tree_base {
            return direct_open_base(compiler, parameters, &concat(blocks), column, t_top, true);
        }
        let child_count = blocks.len() / compiler.tree_base;
        let child_column = column % child_count;
        let sibling = column / child_count;
        let commitments = (0..compiler.tree_base)
            .map(|child| cache[&(offset + child * child_count, child_count)].clone())
            .collect::<Vec<_>>();
        let z_prime =
            direct_open_base(compiler, parameters, &concat(&commitments), sibling, t_top, false);
        let child_start = sibling * child_count;
        let z_child = direct_open_recursive(
            compiler,
            parameters,
            &blocks[child_start..child_start + child_count],
            offset + child_start,
            child_column,
            t_top,
            cache,
            verifier_base,
            verifier_base_last,
        );
        let verifier = direct_verifier_recursive(
            compiler,
            verifier_base,
            verifier_base_last,
            child_count,
            child_column,
        );
        z_prime * verifier.decompose() + z_child
    }

    #[test]
    #[serial_test::serial]
    fn partial_range_opening_and_verifier_match_direct_recursive_oracles() {
        let parameters = DCRTPolyParams::new(4, 1, 12, 4);
        let compiler = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            tree_base: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        };
        let hash_key = [0x29; 32];
        let message_values = (0..4)
            .map(|index| matrix(&parameters, 1, compiler.public_columns(), 10 + index * 20))
            .collect::<Vec<_>>();
        let t_top_values = (0..compiler.public_parameter_top_count())
            .map(|index| {
                matrix(
                    &parameters,
                    compiler.public_columns(),
                    compiler.public_columns(),
                    100 + index * compiler.public_columns() * compiler.public_columns(),
                )
            })
            .collect::<Vec<_>>();
        let t_bottom_values = (0..compiler.public_parameter_part_count())
            .map(|index| {
                matrix(
                    &parameters,
                    compiler.public_columns(),
                    compiler.public_columns(),
                    10_000 + index * 100,
                )
            })
            .collect::<Vec<_>>();
        let ring = compiler.ring();
        let mut inputs =
            BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]);
        let blocks = message_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("message-{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                ring.input(name, (1, compiler.public_columns()))
            })
            .collect::<Vec<_>>();
        let tree = compiler
            .commitment_tree(ring.bytes_input("hash-key", 32), &blocks)
            .expect("commitment tree");
        let top_wires = t_top_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("top-{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                ring.input(name, (compiler.public_columns(), compiler.public_columns()))
            })
            .collect::<Vec<_>>();
        let mut top_families = Vec::with_capacity(compiler.public_parameter_top_family_count());
        for digit_row in 0..compiler.gadget_rows() {
            for part in 0..compiler.public_parameter_part_count() {
                top_families.push(
                    Family::pack(
                        (0..compiler.public_parameter_block_count())
                            .map(|block| {
                                top_wires[(block * compiler.gadget_rows() + digit_row) *
                                    compiler.public_parameter_part_count() +
                                    part]
                                    .clone()
                            })
                            .collect(),
                    )
                    .unwrap(),
                );
            }
        }
        let bottom_wires = t_bottom_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("bottom-{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                ring.input(name, (compiler.public_columns(), compiler.public_columns()))
            })
            .collect::<Vec<_>>();
        let public = Wee25PublicParameterWires {
            b: ring.zero((compiler.secret_size, compiler.public_columns())),
            t_top: top_families,
            t_bottom: Family::pack(bottom_wires).unwrap(),
        };
        let opening = compiler.opening(&blocks, Some(1..3), &public, &tree.cached_nodes).unwrap();
        let verifier = compiler.verifier(4, Some(1..3), &public).unwrap();
        let graph = DslContext::new("wee25-partial-range-runtime")
            .output("opening", opening)
            .unwrap()
            .output("verifier", verifier)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();

        let mut cache = HashMap::new();
        direct_commit_tree(&compiler, &parameters, hash_key, &message_values, 0, &mut cache);
        let base = direct_verifier_base(&compiler, &parameters, &t_bottom_values, false);
        let base_last = direct_verifier_base(&compiler, &parameters, &t_bottom_values, true);
        let expected_openings = (1..3)
            .map(|column| {
                direct_open_recursive(
                    &compiler,
                    &parameters,
                    &message_values,
                    0,
                    column,
                    &t_top_values,
                    &cache,
                    &base,
                    &base_last,
                )
            })
            .collect::<Vec<_>>();
        let expected_verifiers = (1..3)
            .map(|column| direct_verifier_recursive(&compiler, &base, &base_last, 4, column))
            .collect::<Vec<_>>();
        let RuntimeValue::Matrix(actual_opening) = &result.outputs["opening"] else {
            panic!("opening output")
        };
        let RuntimeValue::Matrix(actual_verifier) = &result.outputs["verifier"] else {
            panic!("verifier output")
        };
        assert_eq!(actual_opening.as_ref(), &concat(&expected_openings));
        assert_eq!(actual_verifier.as_ref(), &concat(&expected_verifiers));
    }

    #[test]
    fn opening_verifier_and_residual_build_and_elaborate() {
        let compiler = Wee25CommitmentCompiler {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_size: 1,
            tree_base: 2,
            digit_count: 2,
            gadget_base: 4.into(),
        };
        let ring = compiler.ring();
        let parameters = Wee25PublicParameterWires {
            b: ring.input("b", (1, compiler.public_columns())),
            t_top: (0..compiler.public_parameter_top_family_count())
                .map(|index| {
                    ring.input_family(
                        format!("top-{index}"),
                        compiler.public_parameter_block_count(),
                        (compiler.public_columns(), compiler.public_columns()),
                    )
                })
                .collect(),
            t_bottom: ring.input_family(
                "bottom",
                compiler.public_parameter_part_count(),
                (compiler.public_columns(), compiler.public_columns()),
            ),
        };
        let blocks = (0..4)
            .map(|index| ring.input(format!("block-{index}"), (1, compiler.public_columns())))
            .collect::<Vec<_>>();
        let tree = compiler.commitment_tree(ring.bytes_input("key", 32), &blocks).unwrap();
        let opening = compiler.opening(&blocks, None, &parameters, &tree.cached_nodes).unwrap();
        let residual = compiler
            .verification_residual(&blocks, &tree.root, &opening, None, &parameters)
            .unwrap();
        let built = DslContext::new("wee25-opening")
            .output("residual", residual.residual)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        built.elaborate(&ParamEnv::default()).unwrap();
    }

    #[test]
    #[serial_test::serial]
    fn generated_parameters_opening_and_verifier_have_zero_residual() {
        let parameters = DCRTPolyParams::new(4, 1, 12, 4);
        let compiler = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            tree_base: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        };
        let ring = compiler.ring();
        let hash_key = ring.bytes_input("hash-key", 32);
        let public =
            Wee25PublicParameterCompiler { layout: compiler.clone(), trapdoor_sigma: 5.into() }
                .build(hash_key.clone())
                .unwrap();
        let blocks = (0..2)
            .map(|index| ring.input(format!("block-{index}"), (1, compiler.public_columns())))
            .collect::<Vec<_>>();
        let tree = compiler.commitment_tree(hash_key, &blocks).unwrap();
        assert_eq!(tree.cached_nodes.element_type(), &compiler.block_type());
        assert_eq!(tree.cached_nodes.count(), &IntExpr::constant(1));
        assert_eq!(public.public_parameters.b.matrix_type(), &compiler.block_type());
        assert_eq!(
            public.public_parameters.t_top.len(),
            compiler.public_parameter_top_family_count()
        );
        assert!(public.public_parameters.t_top.iter().all(|family| family.element_type() ==
            &compiler.matrix_type(compiler.public_columns(), compiler.public_columns())));
        let opening =
            compiler.opening(&blocks, None, &public.public_parameters, &tree.cached_nodes).unwrap();
        let residual = compiler
            .verification_residual(&blocks, &tree.root, &opening, None, &public.public_parameters)
            .unwrap();
        let built = DslContext::new("wee25-end-to-end")
            .output("residual", residual.residual)
            .unwrap()
            .build()
            .unwrap();
        let validated = built.validate(&ParamEnv::default()).unwrap();
        let block = |offset| {
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    (0..compiler.public_columns())
                        .map(|column| {
                            DCRTPoly::from_usize_to_constant(&parameters, offset + column)
                        })
                        .collect(),
                ],
            )
        };
        let inputs = BTreeMap::from([
            ("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x45; 32])),
            ("block-0".to_owned(), RuntimeValue::matrix(block(1))),
            ("block-1".to_owned(), RuntimeValue::matrix(block(1 + compiler.public_columns()))),
        ]);
        let result = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(actual) = &result.outputs["residual"] else { panic!("matrix") };
        assert_eq!(
            actual.as_ref(),
            &DCRTPolyMatrix::zero(&parameters, 1, 2 * compiler.public_columns())
        );
    }
}
