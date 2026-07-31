use crate::{
    WEE25_PUBLIC_B, WEE25_T_BOTTOM, Wee25CommitmentCompiler, Wee25CommitmentError,
    Wee25PublicParameterWires,
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixWire, RealExpr, TrapdoorWire, WireRef,
    artifact::ArtifactConfidentiality,
    node::{
        ConcatAxis, ConstantMatrix, IndexRange, IntBinaryOp, IntCompareOp, LoopInputMode,
        MatrixBinaryOp,
    },
};

pub const WEE25_PUBLIC_B_TRAPDOOR: &str = "wee25_public_b_trapdoor";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25PublicParameterCompiler {
    pub layout: Wee25CommitmentCompiler,
    pub trapdoor_sigma: RealExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25PublicParameterPreprocessingWires {
    pub public_parameters: Wee25PublicParameterWires,
    pub b_trapdoor: TrapdoorWire,
}

impl Wee25PublicParameterCompiler {
    pub fn build(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
    ) -> Result<Wee25PublicParameterPreprocessingWires, Wee25CommitmentError> {
        self.layout.validate_layout()?;
        let public_columns = self.layout.public_columns();
        let part_count = self.layout.public_parameter_part_count();
        let b_trapdoor = builder.trapdoor_sample(
            self.layout.block_type(),
            self.trapdoor_sigma.clone(),
            self.layout.gadget_base.clone(),
            IntExpr::constant(self.layout.digit_count),
        );
        // The legacy implementation performs one Gaussian draw and only then
        // slices it into public chunks. Keep that draw boundary unchanged.
        let t_bottom_full = builder.gaussian_sample(
            self.layout.matrix_type(public_columns, part_count * public_columns),
            self.trapdoor_sigma.clone(),
        );
        let t_bottom_parts = (0..part_count)
            .map(|part| {
                builder.slice(
                    &t_bottom_full,
                    None,
                    Some(IndexRange {
                        start: part * public_columns,
                        end: (part + 1) * public_columns,
                    }),
                    self.layout.matrix_type(public_columns, public_columns),
                )
            })
            .collect::<Vec<_>>();
        let t_bottom = builder.family_pack(&t_bottom_parts)?;

        let mut t_top = Vec::with_capacity(self.layout.public_parameter_top_family_count());
        for digit_row in 0..self.layout.gadget_rows() {
            for part in 0..part_count {
                t_top.push(self.build_top_family(
                    builder,
                    hash_key,
                    &b_trapdoor,
                    &t_bottom_parts[part],
                    digit_row,
                    part,
                )?);
            }
        }
        Ok(Wee25PublicParameterPreprocessingWires {
            public_parameters: Wee25PublicParameterWires {
                b: b_trapdoor.public.clone(),
                t_top,
                t_bottom,
            },
            b_trapdoor,
        })
    }

    pub fn export(
        &self,
        builder: &mut GraphBuilder,
        wires: &Wee25PublicParameterPreprocessingWires,
    ) {
        builder.output(WEE25_PUBLIC_B, &wires.public_parameters.b, ArtifactConfidentiality::Public);
        builder.output_wire(
            WEE25_PUBLIC_B_TRAPDOOR,
            wires.b_trapdoor.wire,
            ArtifactConfidentiality::Private,
        );
        builder.output_family_wire(
            WEE25_T_BOTTOM,
            &wires.public_parameters.t_bottom,
            ArtifactConfidentiality::Public,
        );
        for digit_row in 0..self.layout.gadget_rows() {
            for part in 0..self.layout.public_parameter_part_count() {
                let family = digit_row * self.layout.public_parameter_part_count() + part;
                builder.output_family_wire(
                    self.layout.public_parameter_top_name(digit_row, part),
                    &wires.public_parameters.t_top[family],
                    ArtifactConfidentiality::Public,
                );
            }
        }
    }

    fn build_top_family(
        &self,
        builder: &mut GraphBuilder,
        hash_key: WireRef,
        b_trapdoor: &TrapdoorWire,
        t_bottom_part: &MatrixWire,
        digit_row: usize,
        part: usize,
    ) -> Result<mxx_ir_core::MatrixFamilyWire, Wee25CommitmentError> {
        let public_columns = self.layout.public_columns();
        let mut body = GraphBuilder::new(
            format!(
                "wee25-public-top-d{}-b{}-k{}-digit{}-part{}",
                self.layout.secret_size,
                self.layout.tree_base,
                self.layout.digit_count,
                digit_row,
                part
            ),
            Vec::new(),
        );
        let body_hash_key = body.bytes_input("0_hash_key", 32);
        let body_b = body.trapdoor_input(
            "1_b",
            self.layout.block_type(),
            self.trapdoor_sigma.clone(),
            self.layout.gadget_base.clone(),
            IntExpr::constant(self.layout.digit_count),
        );
        let body_t_bottom =
            body.input("2_t_bottom", self.layout.matrix_type(public_columns, public_columns));
        let block = IntExpr::Add(
            Box::new(IntExpr::Mul(
                Box::new(IntExpr::Var("block".to_owned())),
                Box::new(IntExpr::constant(self.layout.gadget_rows())),
            )),
            Box::new(IntExpr::constant(digit_row)),
        );
        let w = body.hash_sample_with_encoded_tags(
            body_hash_key,
            self.layout.block_type(),
            mxx_ir_core::node::HashVariant::Plain,
            b"wee25_w_block_".to_vec(),
            Vec::new(),
            Vec::new(),
            vec![block],
            None,
            None,
        );
        let j_block = self.build_j_block(&mut body, digit_row, part)?;
        let gadget = body.constant_matrix(
            self.layout.matrix_type(self.layout.secret_size, self.layout.gadget_rows()),
            ConstantMatrix::Gadget { base: self.layout.gadget_base.clone(), small: false },
        );
        let gadget_j = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &gadget,
            &j_block,
            self.layout.block_type(),
        );
        let w_bottom = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &w,
            &body_t_bottom,
            self.layout.block_type(),
        );
        let target = body.matrix_binary(
            MatrixBinaryOp::Subtract,
            &gadget_j,
            &w_bottom,
            self.layout.block_type(),
        );
        let preimage = body.preimage_sample(
            &body_b,
            &target,
            self.layout.matrix_type(public_columns, public_columns),
        );
        body.value_output_wire("0_preimage", preimage.wire);
        Ok(builder
            .parallel_loop(
                body.finish(),
                IntExpr::constant(self.layout.public_parameter_block_count()),
                "block",
                Vec::new(),
                vec![hash_key, b_trapdoor.wire, t_bottom_part.wire],
                vec![LoopInputMode::Broadcast; 3],
                &[preimage.matrix_type],
            )?
            .remove(0))
    }

    fn build_j_block(
        &self,
        builder: &mut GraphBuilder,
        digit_row: usize,
        part: usize,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        let d = self.layout.secret_size;
        let k = self.layout.digit_count;
        let m_g = self.layout.gadget_rows();
        let m_b = self.layout.public_columns();
        let mut decomposed_rows = Vec::with_capacity(d);
        for secret_row in 0..d {
            let offset = digit_row * m_g + secret_row * k;
            let step = m_g + 1;
            let c = offset.div_ceil(step);
            let position = c * step;
            let valid = c < m_g && position <= offset + k - 1;
            let mut row =
                builder.constant_matrix(self.layout.matrix_type(1, m_b), ConstantMatrix::Zero);
            if valid {
                let coefficient_digit = position - offset;
                for s in 0..k {
                    let global_column_expr = IntExpr::Add(
                        Box::new(IntExpr::Mul(
                            Box::new(IntExpr::Var("block".to_owned())),
                            Box::new(IntExpr::constant(k)),
                        )),
                        Box::new(IntExpr::constant(s)),
                    );
                    let global_column = builder.evaluate_int(global_column_expr);
                    let public_columns_wire = builder.constant_int(m_b);
                    let local_column = builder.int_binary(
                        IntBinaryOp::Remainder,
                        global_column,
                        public_columns_wire,
                    );
                    let unit = self.dynamic_unit_row(builder, local_column)?;
                    let coefficient = builder.constant_matrix(
                        self.layout.matrix_type(1, 1),
                        ConstantMatrix::PowerOfBase {
                            base: self.layout.gadget_base.clone(),
                            exponent: IntExpr::constant(coefficient_digit + s),
                        },
                    );
                    let term = builder.matrix_binary(
                        MatrixBinaryOp::Multiply,
                        &coefficient,
                        &unit,
                        self.layout.matrix_type(1, m_b),
                    );
                    let zero = builder
                        .constant_matrix(self.layout.matrix_type(1, m_b), ConstantMatrix::Zero);
                    let part_start = builder.constant_int(part * m_b);
                    let below_start =
                        builder.int_compare(IntCompareOp::Less, global_column, part_start);
                    let below_start = builder.bool_to_int(below_start);
                    let above_start = builder.select(below_start, &[term, zero.clone()]);
                    let part_end = builder.constant_int((part + 1) * m_b);
                    let below_end =
                        builder.int_compare(IntCompareOp::Less, global_column, part_end);
                    let below_end = builder.bool_to_int(below_end);
                    let in_part = builder.select(below_end, &[zero, above_start]);
                    row = builder.matrix_binary(
                        MatrixBinaryOp::Add,
                        &row,
                        &in_part,
                        self.layout.matrix_type(1, m_b),
                    );
                }
            }
            decomposed_rows.push(builder.gadget_decompose_with_layout(
                &row,
                self.layout.gadget_base.clone(),
                false,
                Some(IntExpr::constant(k)),
                self.layout.matrix_type(k, m_b),
            ));
        }
        Ok(if decomposed_rows.len() == 1 {
            decomposed_rows.remove(0)
        } else {
            builder.concat(ConcatAxis::Rows, &decomposed_rows, self.layout.matrix_type(m_g, m_b))
        })
    }

    fn dynamic_unit_row(
        &self,
        builder: &mut GraphBuilder,
        index: WireRef,
    ) -> Result<MatrixWire, Wee25CommitmentError> {
        let columns = self.layout.public_columns();
        let mut body = GraphBuilder::new(format!("wee25-unit-row-{columns}"), Vec::new());
        let body_index = body.integer_input("0_index");
        let branches = (0..columns)
            .map(|column| {
                body.constant_matrix(
                    self.layout.matrix_type(1, columns),
                    ConstantMatrix::UnitRow { index: IntExpr::constant(column) },
                )
            })
            .collect::<Vec<_>>();
        let row = body.select(body_index, &branches);
        body.value_output_wire("0_row", row.wire);
        Ok(builder.subgraph_call(body.finish(), vec![index], &[row.matrix_type])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, validate};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    type HashSampler = DCRTPolyHashSampler<keccak_asm::Keccak256>;

    fn direct_j_block(
        layout: &Wee25CommitmentCompiler,
        parameters: &DCRTPolyParams,
        block_index: usize,
        part: usize,
    ) -> DCRTPolyMatrix {
        let d = layout.secret_size;
        let k = layout.digit_count;
        let m_g = layout.gadget_rows();
        let m_b = layout.public_columns();
        let block_group = block_index / m_g;
        let gadget_row = DCRTPolyMatrix::gadget_matrix(parameters, 1).get_row(0);
        let mut rows = (0..d)
            .map(|secret_row| {
                let r = block_index * d + secret_row;
                let r_g_start = r * k;
                let slice_start = block_group * m_g * m_g;
                let offset = r_g_start - slice_start;
                let step = m_g + 1;
                let mut row = DCRTPolyMatrix::zero(parameters, 1, m_b);
                let c = offset.div_ceil(step);
                if c < m_g {
                    let position = slice_start + c * step;
                    if position <= r_g_start + k - 1 {
                        let coefficient_digit = position - r_g_start;
                        for s in 0..k {
                            let global_column = block_group * k + s;
                            let start = part * m_b;
                            if (start..start + m_b).contains(&global_column) {
                                row.set_entry(
                                    0,
                                    global_column - start,
                                    gadget_row[coefficient_digit].clone() * &gadget_row[s],
                                );
                            }
                        }
                    }
                }
                row.decompose()
            })
            .collect::<Vec<_>>();
        let first = rows.remove(0);
        first.concat_rows(&rows.iter().collect::<Vec<_>>())
    }

    #[test]
    #[serial_test::serial]
    fn public_parameters_preserve_every_legacy_chunk_relation() {
        let parameters = DCRTPolyParams::new(4, 1, 12, 4);
        let layout = Wee25CommitmentCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            tree_base: 2,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        };
        let compiler = Wee25PublicParameterCompiler {
            layout: layout.clone(),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
        };
        let hash_key = [0x45; 32];
        let mut builder = GraphBuilder::new("wee25-public-parameter-test", Vec::new());
        let hash_key_wire = builder.bytes_input("hash_key", 32);
        let wires = compiler.build(&mut builder, hash_key_wire).expect("public graph");
        builder.value_output_wire("inspect_b", wires.public_parameters.b.wire);
        for part in 0..layout.public_parameter_part_count() {
            let value = builder
                .family_get_static(&wires.public_parameters.t_bottom, IntExpr::constant(part));
            builder.value_output_wire(format!("inspect_t_bottom_{part}"), value.wire);
        }
        for digit_row in 0..layout.gadget_rows() {
            for part in 0..layout.public_parameter_part_count() {
                let family = digit_row * layout.public_parameter_part_count() + part;
                for block in 0..layout.public_parameter_block_count() {
                    let value = builder.family_get_static(
                        &wires.public_parameters.t_top[family],
                        IntExpr::constant(block),
                    );
                    builder.value_output_wire(
                        format!("inspect_t_top_{digit_row}_{part}_{block}"),
                        value.wire,
                    );
                }
            }
        }
        compiler.export(&mut builder, &wires);
        let graph = validate(&builder.finish(), &ParamEnv::default()).expect("valid graph");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &graph,
            &mut backend,
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("runtime execution");
        let RuntimeValue::Matrix(b) = &result.outputs["inspect_b"] else {
            panic!("public B");
        };
        let t_bottom = (0..layout.public_parameter_part_count())
            .map(|part| {
                let RuntimeValue::Matrix(value) =
                    &result.outputs[&format!("inspect_t_bottom_{part}")]
                else {
                    panic!("t_bottom part");
                };
                value.as_ref()
            })
            .collect::<Vec<_>>();
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_size);
        let hash_sampler = HashSampler::new();
        for digit_row in 0..layout.gadget_rows() {
            for part in 0..layout.public_parameter_part_count() {
                for block in 0..layout.public_parameter_block_count() {
                    let RuntimeValue::Matrix(top) =
                        &result.outputs[&format!("inspect_t_top_{digit_row}_{part}_{block}")]
                    else {
                        panic!("t_top part");
                    };
                    let block_index = block * layout.gadget_rows() + digit_row;
                    let mut tag = b"wee25_w_block_".to_vec();
                    tag.extend_from_slice(&block_index.to_le_bytes());
                    let w = hash_sampler.sample_hash(
                        &parameters,
                        hash_key,
                        tag,
                        layout.secret_size,
                        layout.public_columns(),
                        DistType::FinRingDist,
                    );
                    let j = direct_j_block(&layout, &parameters, block_index, part);
                    let expected = gadget.clone() * &j - &(w * t_bottom[part]);
                    assert_eq!(b.as_ref().clone() * top.as_ref(), expected);
                }
            }
        }
    }
}
