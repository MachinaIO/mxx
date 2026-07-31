use crate::{Wee25CommitmentCompiler, Wee25CommitmentError};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire,
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, ConstantMatrix, IndexRange, MatrixBinaryOp},
    types::MatrixType,
};
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25PublicParameterWires {
    pub b: MatrixWire,
    /// Families are ordered by `(digit_row, column_part)`. Each family is
    /// indexed by the legacy message-column/block-group coordinate.
    pub t_top: Vec<MatrixFamilyWire>,
    pub t_bottom: MatrixFamilyWire,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25VerificationWire {
    /// The commitment is valid exactly when every entry of this matrix is zero.
    pub residual: MatrixWire,
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
        builder: &mut GraphBuilder,
        artifacts: &Wee25PublicParameterArtifacts,
    ) -> Result<Wee25PublicParameterWires, Wee25CommitmentError> {
        self.validate_layout()?;
        let b = builder.artifact_input(
            "wee25_public_b_input",
            self.block_type(),
            artifacts.production_id.clone(),
            WEE25_PUBLIC_B,
            ArtifactConfidentiality::Public,
        );
        let mut t_top = Vec::with_capacity(self.public_parameter_top_family_count());
        for digit_row in 0..self.gadget_rows() {
            for part in 0..self.public_parameter_part_count() {
                let artifact_name = self.public_parameter_top_name(digit_row, part);
                t_top.push(builder.artifact_family_input(
                    format!("{artifact_name}_input"),
                    self.matrix_type(self.public_columns(), self.public_columns()),
                    artifacts.production_id.clone(),
                    artifact_name,
                    IntExpr::constant(self.public_parameter_block_count()),
                    ArtifactConfidentiality::Public,
                ));
            }
        }
        let t_bottom = builder.artifact_family_input(
            "wee25_t_bottom_input",
            self.matrix_type(self.public_columns(), self.public_columns()),
            artifacts.production_id.clone(),
            WEE25_T_BOTTOM,
            IntExpr::constant(self.public_parameter_part_count()),
            ArtifactConfidentiality::Public,
        );
        Ok(Wee25PublicParameterWires { b, t_top, t_bottom })
    }

    pub fn import_commitment_artifacts(
        &self,
        builder: &mut GraphBuilder,
        artifacts: &Wee25CommitmentArtifacts,
    ) -> Result<(MatrixWire, MatrixFamilyWire), Wee25CommitmentError> {
        self.validate_block_count(artifacts.block_count)?;
        let commitment = builder.artifact_input(
            "wee25_commitment_input",
            self.block_type(),
            artifacts.production_id.clone(),
            WEE25_COMMITMENT,
            ArtifactConfidentiality::Public,
        );
        let nodes = builder.artifact_family_input(
            "wee25_commitment_nodes_input",
            self.block_type(),
            artifacts.production_id.clone(),
            WEE25_COMMITMENT_NODES,
            IntExpr::constant(self.cache_node_count(artifacts.block_count)),
            ArtifactConfidentiality::Public,
        );
        Ok((commitment, nodes))
    }

    pub fn opening(
        &self,
        builder: &mut GraphBuilder,
        message_blocks: &[MatrixWire],
        range: Option<Range<usize>>,
        public_parameters: &Wee25PublicParameterWires,
        commitment_nodes: &MatrixFamilyWire,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        self.validate_opening_inputs(message_blocks, commitment_nodes)?;
        self.validate_public_parameter_wires(public_parameters)?;
        let range = range.unwrap_or(0..message_blocks.len());
        if range.start >= range.end || range.end > message_blocks.len() {
            return Err(Wee25CommitmentError::InvalidBlockCount);
        }
        let verifier_base = self.verifier_base(builder, public_parameters, false);
        let verifier_base_last = self.verifier_base(builder, public_parameters, true);
        let mut z_prime_cache = BTreeMap::new();
        let mut verifier_cache = BTreeMap::new();
        let openings = range
            .map(|column| {
                self.open_recursive(
                    builder,
                    message_blocks,
                    0,
                    message_blocks.len(),
                    column,
                    public_parameters,
                    commitment_nodes,
                    &verifier_base,
                    &verifier_base_last,
                    &mut z_prime_cache,
                    &mut verifier_cache,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.concat_columns(builder, &openings))
    }

    pub fn verifier(
        &self,
        builder: &mut GraphBuilder,
        block_count: usize,
        range: Option<Range<usize>>,
        public_parameters: &Wee25PublicParameterWires,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        self.validate_block_count(block_count)?;
        self.validate_public_parameter_wires(public_parameters)?;
        let range = range.unwrap_or(0..block_count);
        if range.start >= range.end || range.end > block_count {
            return Err(Wee25CommitmentError::InvalidBlockCount);
        }
        let base = self.verifier_base(builder, public_parameters, false);
        let base_last = self.verifier_base(builder, public_parameters, true);
        let mut cache = BTreeMap::new();
        let columns = range
            .map(|column| {
                self.verifier_recursive(builder, &base, &base_last, block_count, column, &mut cache)
            })
            .collect::<Vec<_>>();
        Ok(self.concat_columns(builder, &columns))
    }

    pub fn verification_residual(
        &self,
        builder: &mut GraphBuilder,
        message_blocks: &[MatrixWire],
        commitment: &MatrixWire,
        opening: &MatrixWire,
        range: Option<Range<usize>>,
        public_parameters: &Wee25PublicParameterWires,
    ) -> Result<Wee25VerificationWire, Wee25CommitmentError> {
        self.validate_block_count(message_blocks.len())?;
        let range = range.unwrap_or(0..message_blocks.len());
        if range.start >= range.end || range.end > message_blocks.len() {
            return Err(Wee25CommitmentError::InvalidBlockCount);
        }
        let verifier =
            self.verifier(builder, message_blocks.len(), Some(range.clone()), public_parameters)?;
        let selected_message = self.concat_columns(builder, &message_blocks[range.clone()]);
        let output_type =
            self.matrix_type(self.secret_size, self.public_columns() * (range.end - range.start));
        let lhs = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            commitment,
            &verifier,
            output_type.clone(),
        );
        let b_opening = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &public_parameters.b,
            opening,
            output_type.clone(),
        );
        let rhs = builder.matrix_binary(
            MatrixBinaryOp::Subtract,
            &selected_message,
            &b_opening,
            output_type.clone(),
        );
        Ok(Wee25VerificationWire {
            residual: builder.matrix_binary(MatrixBinaryOp::Subtract, &lhs, &rhs, output_type),
        })
    }

    fn validate_opening_inputs(
        &self,
        message_blocks: &[MatrixWire],
        commitment_nodes: &MatrixFamilyWire,
    ) -> Result<(), Wee25CommitmentError> {
        self.validate_block_count(message_blocks.len())?;
        if message_blocks.iter().any(|block| block.matrix_type != self.block_type()) ||
            commitment_nodes.matrix_type != self.block_type() ||
            commitment_nodes.count !=
                IntExpr::constant(self.cache_node_count(message_blocks.len()))
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
        if parameters.b.matrix_type != self.block_type() ||
            parameters.t_top.len() != self.public_parameter_top_family_count() ||
            parameters.t_top.iter().any(|family| {
                family.matrix_type != part_type ||
                    family.count != IntExpr::constant(self.public_parameter_block_count())
            }) ||
            parameters.t_bottom.matrix_type != part_type ||
            parameters.t_bottom.count != IntExpr::constant(self.public_parameter_part_count())
        {
            return Err(Wee25CommitmentError::InvalidLayout);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn open_recursive(
        &self,
        builder: &mut GraphBuilder,
        blocks: &[MatrixWire],
        offset: usize,
        total_count: usize,
        column: usize,
        public_parameters: &Wee25PublicParameterWires,
        commitment_nodes: &MatrixFamilyWire,
        verifier_base: &MatrixWire,
        verifier_base_last: &MatrixWire,
        z_prime_cache: &mut BTreeMap<(usize, usize, usize), MatrixWire>,
        verifier_cache: &mut BTreeMap<(usize, usize), MatrixWire>,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        if blocks.len() == self.tree_base {
            let message = self.concat_columns(builder, blocks);
            return self.call_open_base(builder, &message, column, public_parameters, true);
        }
        let child_count = blocks.len() / self.tree_base;
        let child_column = column % child_count;
        let sibling = column / child_count;
        let commitments = (0..self.tree_base)
            .map(|child| {
                let child_offset = offset + child * child_count;
                builder.family_get_static(
                    commitment_nodes,
                    IntExpr::constant(self.cache_node_index(
                        total_count,
                        child_offset,
                        child_count,
                    )),
                )
            })
            .collect::<Vec<_>>();
        let commitment_message = self.concat_columns(builder, &commitments);
        let z_prime_key = (offset, blocks.len(), sibling);
        let z_prime = if let Some(value) = z_prime_cache.get(&z_prime_key) {
            value.clone()
        } else {
            let value = self.call_open_base(
                builder,
                &commitment_message,
                sibling,
                public_parameters,
                false,
            )?;
            z_prime_cache.insert(z_prime_key, value.clone());
            value
        };
        let child_start = sibling * child_count;
        let z_child = self.open_recursive(
            builder,
            &blocks[child_start..child_start + child_count],
            offset + child_start,
            total_count,
            child_column,
            public_parameters,
            commitment_nodes,
            verifier_base,
            verifier_base_last,
            z_prime_cache,
            verifier_cache,
        )?;
        let verifier = self.verifier_recursive(
            builder,
            verifier_base,
            verifier_base_last,
            child_count,
            child_column,
            verifier_cache,
        );
        let decomposed = builder.gadget_decompose_with_layout(
            &verifier,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(self.public_columns() * self.digit_count, self.public_columns()),
        );
        let product = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &z_prime,
            &decomposed,
            self.matrix_type(self.public_columns(), self.public_columns()),
        );
        Ok(builder.matrix_binary(
            MatrixBinaryOp::Add,
            &product,
            &z_child,
            self.matrix_type(self.public_columns(), self.public_columns()),
        ))
    }

    fn call_open_base(
        &self,
        builder: &mut GraphBuilder,
        message: &MatrixWire,
        column: usize,
        public_parameters: &Wee25PublicParameterWires,
        leaf: bool,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        let mut body = GraphBuilder::new(
            format!(
                "wee25-open-base-d{}-b{}-k{}-column{}-leaf{}",
                self.secret_size, self.tree_base, self.digit_count, column, leaf
            ),
            Vec::new(),
        );
        let body_message = body.input(
            "0_message",
            self.matrix_type(self.secret_size, self.tree_base * self.public_columns()),
        );
        let body_t_top = (0..self.public_parameter_top_family_count())
            .map(|family| {
                body.family_input(
                    format!("1_t_top_{family}"),
                    self.matrix_type(self.public_columns(), self.public_columns()),
                    IntExpr::constant(self.public_parameter_block_count()),
                )
            })
            .collect::<Vec<_>>();
        let output = self.open_base(&mut body, &body_message, column, &body_t_top, leaf);
        body.value_output_wire("0_opening", output.wire);
        let mut args = Vec::with_capacity(1 + public_parameters.t_top.len());
        args.push(message.wire);
        args.extend(public_parameters.t_top.iter().map(|family| family.wire));
        Ok(builder.subgraph_call(body.finish(), args, &[output.matrix_type])?.remove(0))
    }

    fn open_base(
        &self,
        builder: &mut GraphBuilder,
        message: &MatrixWire,
        column: usize,
        t_top: &[MatrixFamilyWire],
        leaf: bool,
    ) -> MatrixWire {
        let base_columns = self.tree_base * self.public_columns();
        let slice_width = self.public_columns() * self.digit_count;
        let decomposition = builder.gadget_decompose_with_layout(
            message,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(self.gadget_rows(), base_columns),
        );
        let output_type = self.matrix_type(self.public_columns(), slice_width);
        let mut opening = builder.constant_matrix(output_type.clone(), ConstantMatrix::Zero);
        for message_column in 0..base_columns {
            for digit_row in 0..self.gadget_rows() {
                let chunks = (0..self.digit_count)
                    .map(|digit| {
                        let chunk = column * self.digit_count + digit;
                        let family = digit_row * self.public_parameter_part_count() + chunk;
                        builder.family_get_static(
                            t_top.get(family).expect("t_top layout"),
                            IntExpr::constant(message_column),
                        )
                    })
                    .collect::<Vec<_>>();
                let t_part = self.concat_columns(builder, &chunks);
                let scalar = builder.slice(
                    &decomposition,
                    Some(IndexRange { start: digit_row, end: digit_row + 1 }),
                    Some(IndexRange { start: message_column, end: message_column + 1 }),
                    self.matrix_type(1, 1),
                );
                let term = builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    &t_part,
                    &scalar,
                    output_type.clone(),
                );
                opening = builder.matrix_binary(
                    MatrixBinaryOp::Add,
                    &opening,
                    &term,
                    output_type.clone(),
                );
            }
        }
        if !leaf {
            return opening;
        }
        let identity = builder.constant_matrix(
            self.matrix_type(self.public_columns(), self.public_columns()),
            ConstantMatrix::Identity,
        );
        let decomposed_identity = builder.gadget_decompose_with_layout(
            &identity,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(slice_width, self.public_columns()),
        );
        builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &opening,
            &decomposed_identity,
            self.matrix_type(self.public_columns(), self.public_columns()),
        )
    }

    fn verifier_base(
        &self,
        builder: &mut GraphBuilder,
        public_parameters: &Wee25PublicParameterWires,
        leaf: bool,
    ) -> MatrixWire {
        let chunks = (0..self.public_parameter_part_count())
            .map(|part| {
                builder.family_get_static(&public_parameters.t_bottom, IntExpr::constant(part))
            })
            .collect::<Vec<_>>();
        let t_bottom = self.concat_columns(builder, &chunks);
        if !leaf {
            return t_bottom;
        }
        let columns = self.tree_base * self.public_columns();
        let identity =
            builder.constant_matrix(self.matrix_type(columns, columns), ConstantMatrix::Identity);
        let decomposed = builder.gadget_decompose_with_layout(
            &identity,
            self.gadget_base.clone(),
            false,
            Some(IntExpr::constant(self.digit_count)),
            self.matrix_type(columns * self.digit_count, columns),
        );
        builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &t_bottom,
            &decomposed,
            self.matrix_type(self.public_columns(), columns),
        )
    }

    fn verifier_recursive(
        &self,
        builder: &mut GraphBuilder,
        base: &MatrixWire,
        base_last: &MatrixWire,
        block_count: usize,
        column: usize,
        cache: &mut BTreeMap<(usize, usize), MatrixWire>,
    ) -> MatrixWire {
        if let Some(value) = cache.get(&(block_count, column)) {
            return value.clone();
        }
        let result = if block_count == self.tree_base {
            builder.slice(
                base_last,
                None,
                Some(IndexRange {
                    start: self.public_columns() * column,
                    end: self.public_columns() * (column + 1),
                }),
                self.matrix_type(self.public_columns(), self.public_columns()),
            )
        } else {
            let child_count = block_count / self.tree_base;
            let child_column = column % child_count;
            let child =
                self.verifier_recursive(builder, base, base_last, child_count, child_column, cache);
            let sibling = column / child_count;
            let width = self.public_columns() * self.digit_count;
            let slice = builder.slice(
                base,
                None,
                Some(IndexRange { start: width * sibling, end: width * (sibling + 1) }),
                self.matrix_type(self.public_columns(), width),
            );
            let decomposed = builder.gadget_decompose_with_layout(
                &child,
                self.gadget_base.clone(),
                false,
                Some(IntExpr::constant(self.digit_count)),
                self.matrix_type(width, self.public_columns()),
            );
            builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &slice,
                &decomposed,
                self.matrix_type(self.public_columns(), self.public_columns()),
            )
        };
        cache.insert((block_count, column), result.clone());
        result
    }

    fn concat_columns(&self, builder: &mut GraphBuilder, values: &[MatrixWire]) -> MatrixWire {
        debug_assert!(!values.is_empty());
        if values.len() == 1 {
            return values[0].clone();
        }
        let rows = values[0].matrix_type.rows.clone();
        let columns = values.iter().fold(IntExpr::constant(0), |sum, value| {
            IntExpr::Add(Box::new(sum), Box::new(value.matrix_type.columns.clone()))
        });
        builder.concat(
            ConcatAxis::Columns,
            values,
            MatrixType {
                modulus: self.modulus.clone(),
                ring_dimension: self.ring_dimension.clone(),
                rows,
                columns,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv, artifact::ArtifactConfidentiality, validate, validate_with_manifests,
    };
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
        let refs = values.iter().collect::<Vec<_>>();
        values[0].concat_columns(&refs[1..])
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
        total: usize,
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
                        total,
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
        let slice_width = compiler.public_columns() * compiler.digit_count;
        let mut result = DCRTPolyMatrix::zero(parameters, compiler.public_columns(), slice_width);
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
                let t_part = chunks[0].concat_columns(&chunks[1..]);
                result = result + &(t_part * decomposed.entry(digit_row, 0));
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
    fn opening_and_verifier_match_legacy_recursions_exactly() {
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

        let mut builder = GraphBuilder::new("wee25-opening-test", Vec::new());
        let hash_key_wire = builder.bytes_input("hash_key", 32);
        let mut inputs = BTreeMap::from([(
            "hash_key".to_owned(),
            RuntimeValue::<CpuDcrtBackend>::Bytes(hash_key.to_vec()),
        )]);
        let message_wires = message_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("message_{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                builder.input(name, compiler.block_type())
            })
            .collect::<Vec<_>>();
        let tree = compiler
            .commitment_tree(&mut builder, hash_key_wire, &message_wires)
            .expect("commitment tree");
        let t_top_wires = t_top_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("t_top_{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                builder.input(
                    name,
                    compiler.matrix_type(compiler.public_columns(), compiler.public_columns()),
                )
            })
            .collect::<Vec<_>>();
        let mut t_top_families = Vec::with_capacity(compiler.public_parameter_top_family_count());
        for digit_row in 0..compiler.gadget_rows() {
            for part in 0..compiler.public_parameter_part_count() {
                let members = (0..compiler.public_parameter_block_count())
                    .map(|block| {
                        t_top_wires[(block * compiler.gadget_rows() + digit_row) *
                            compiler.public_parameter_part_count() +
                            part]
                            .clone()
                    })
                    .collect::<Vec<_>>();
                t_top_families.push(builder.family_pack(&members).expect("t_top family"));
            }
        }
        let t_bottom_wires = t_bottom_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("t_bottom_{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                builder.input(
                    name,
                    compiler.matrix_type(compiler.public_columns(), compiler.public_columns()),
                )
            })
            .collect::<Vec<_>>();
        let public_parameters = Wee25PublicParameterWires {
            b: builder.constant_matrix(compiler.block_type(), ConstantMatrix::Zero),
            t_top: t_top_families,
            t_bottom: builder.family_pack(&t_bottom_wires).expect("t_bottom family"),
        };
        let opening = compiler
            .opening(
                &mut builder,
                &message_wires,
                Some(1..3),
                &public_parameters,
                &tree.cached_nodes,
            )
            .expect("opening graph");
        let verifier = compiler
            .verifier(&mut builder, 4, Some(1..3), &public_parameters)
            .expect("verifier graph");
        builder.output("opening", &opening, ArtifactConfidentiality::Public);
        builder.output("verifier", &verifier, ArtifactConfidentiality::Public);
        let graph = validate(&builder.finish(), &ParamEnv::default()).expect("valid graph");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("runtime execution");

        let mut cache = HashMap::new();
        direct_commit_tree(
            &compiler,
            &parameters,
            hash_key,
            &message_values,
            0,
            message_values.len(),
            &mut cache,
        );
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
            panic!("opening output");
        };
        let RuntimeValue::Matrix(actual_verifier) = &result.outputs["verifier"] else {
            panic!("verifier output");
        };
        assert_eq!(actual_opening.as_ref(), &concat(&expected_openings));
        assert_eq!(actual_verifier.as_ref(), &concat(&expected_verifiers));

        let mut public_builder = GraphBuilder::new("wee25-public-artifact-test", Vec::new());
        let mut public_inputs = BTreeMap::new();
        let public_b_value =
            matrix(&parameters, compiler.secret_size, compiler.public_columns(), 50_000);
        public_inputs.insert("public_b".to_owned(), RuntimeValue::matrix(public_b_value.clone()));
        let public_b = public_builder.input("public_b", compiler.block_type());
        let public_t_top = t_top_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("public_t_top_{index}");
                public_inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                public_builder.input(
                    name,
                    compiler.matrix_type(compiler.public_columns(), compiler.public_columns()),
                )
            })
            .collect::<Vec<_>>();
        let public_t_bottom = t_bottom_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("public_t_bottom_{index}");
                public_inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                public_builder.input(
                    name,
                    compiler.matrix_type(compiler.public_columns(), compiler.public_columns()),
                )
            })
            .collect::<Vec<_>>();
        public_builder.output(WEE25_PUBLIC_B, &public_b, ArtifactConfidentiality::Public);
        for digit_row in 0..compiler.gadget_rows() {
            for part in 0..compiler.public_parameter_part_count() {
                let members = (0..compiler.public_parameter_block_count())
                    .map(|block| {
                        public_t_top[(block * compiler.gadget_rows() + digit_row) *
                            compiler.public_parameter_part_count() +
                            part]
                            .clone()
                    })
                    .collect::<Vec<_>>();
                public_builder
                    .output_family(
                        compiler.public_parameter_top_name(digit_row, part),
                        &members,
                        ArtifactConfidentiality::Public,
                    )
                    .expect("public t_top output");
            }
        }
        public_builder
            .output_family(WEE25_T_BOTTOM, &public_t_bottom, ArtifactConfidentiality::Public)
            .expect("public t_bottom output");
        let public_graph =
            validate(&public_builder.finish(), &ParamEnv::default()).expect("public producer");
        let public_result =
            execute(&public_graph, &mut backend, public_inputs, &mut store, SamplingMode::Fresh)
                .expect("public artifact execution");
        let public_production = public_result.production_id.expect("public production");
        let public_manifest = store.manifest(&public_production).expect("public manifest").clone();

        let mut commit_builder = GraphBuilder::new("wee25-commit-artifact-test", Vec::new());
        let commit_hash = commit_builder.bytes_input("hash_key", 32);
        let mut commit_inputs = BTreeMap::from([(
            "hash_key".to_owned(),
            RuntimeValue::<CpuDcrtBackend>::Bytes(hash_key.to_vec()),
        )]);
        let commit_messages = message_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("commit_message_{index}");
                commit_inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                commit_builder.input(name, compiler.block_type())
            })
            .collect::<Vec<_>>();
        let commit_tree = compiler
            .commitment_tree(&mut commit_builder, commit_hash, &commit_messages)
            .expect("artifact commitment tree");
        compiler.export_commitment_tree(&mut commit_builder, &commit_tree);
        let commit_graph =
            validate(&commit_builder.finish(), &ParamEnv::default()).expect("commit producer");
        let commit_result =
            execute(&commit_graph, &mut backend, commit_inputs, &mut store, SamplingMode::Fresh)
                .expect("commit artifact execution");
        let commit_production = commit_result.production_id.expect("commit production");
        let commit_manifest = store.manifest(&commit_production).expect("commit manifest").clone();

        let mut consumer = GraphBuilder::new("wee25-artifact-consumer-test", Vec::new());
        let mut consumer_inputs = BTreeMap::new();
        let consumer_messages = message_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("consumer_message_{index}");
                consumer_inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                consumer.input(name, compiler.block_type())
            })
            .collect::<Vec<_>>();
        let imported_public = compiler
            .import_public_parameters(
                &mut consumer,
                &Wee25PublicParameterArtifacts { production_id: public_production.clone() },
            )
            .expect("public imports");
        let (imported_commitment, imported_nodes) = compiler
            .import_commitment_artifacts(
                &mut consumer,
                &Wee25CommitmentArtifacts {
                    production_id: commit_production.clone(),
                    block_count: message_values.len(),
                },
            )
            .expect("commit imports");
        let imported_opening = compiler
            .opening(
                &mut consumer,
                &consumer_messages,
                Some(1..3),
                &imported_public,
                &imported_nodes,
            )
            .expect("imported opening");
        let residual = compiler
            .verification_residual(
                &mut consumer,
                &consumer_messages,
                &imported_commitment,
                &imported_opening,
                Some(1..3),
                &imported_public,
            )
            .expect("verification residual");
        consumer.output("imported_opening", &imported_opening, ArtifactConfidentiality::Public);
        consumer.output("residual", &residual.residual, ArtifactConfidentiality::Public);
        let consumer_graph = validate_with_manifests(
            &consumer.finish(),
            &ParamEnv::default(),
            &BTreeMap::from([
                (public_production, public_manifest),
                (commit_production, commit_manifest),
            ]),
        )
        .expect("artifact consumer validation");
        let consumer_result = execute(
            &consumer_graph,
            &mut backend,
            consumer_inputs,
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("artifact consumer execution");
        let expected_opening = concat(&expected_openings);
        let expected_verifier = concat(&expected_verifiers);
        let expected_commitment = cache[&(0, message_values.len())].clone();
        let expected_selected_message = concat(&message_values[1..3]);
        let expected_residual = expected_commitment * &expected_verifier -
            &(expected_selected_message - &(public_b_value * &expected_opening));
        let RuntimeValue::Matrix(actual_imported_opening) =
            &consumer_result.outputs["imported_opening"]
        else {
            panic!("imported opening output");
        };
        let RuntimeValue::Matrix(actual_residual) = &consumer_result.outputs["residual"] else {
            panic!("residual output");
        };
        assert_eq!(actual_imported_opening.as_ref(), &expected_opening);
        assert_eq!(actual_residual.as_ref(), &expected_residual);
    }
}
