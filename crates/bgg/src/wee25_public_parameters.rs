//! WEE25 public-parameter preprocessing expressed with the declarative DSL.

use crate::{
    WEE25_PUBLIC_B, WEE25_T_BOTTOM, Wee25CommitmentCompiler, Wee25CommitmentError,
    Wee25PublicParameterWires,
};
use mxx_dsl::{Bytes, DslContext, DslError, Family, HashTag, Mat, Trapdoor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    node::{ConcatAxis, ConstantMatrix, IndexRange},
};

pub const WEE25_PUBLIC_B_TRAPDOOR: &str = "wee25_public_b_trapdoor";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wee25PublicParameterCompiler {
    pub layout: Wee25CommitmentCompiler,
    pub trapdoor_sigma: RealExpr,
    pub gaussian_max_coefficient_bound: IntExpr,
    pub preimage_max_coefficient_bound: IntExpr,
}

#[derive(Clone)]
pub struct Wee25PublicParameterPreprocessingWires {
    pub public_parameters: Wee25PublicParameterWires,
    pub b_trapdoor: Trapdoor,
}

impl Wee25PublicParameterCompiler {
    pub fn build(
        &self,
        hash_key: Bytes,
    ) -> Result<Wee25PublicParameterPreprocessingWires, Wee25CommitmentError> {
        self.layout.validate_layout()?;
        let ring = self.layout.ring();
        let public_columns = self.layout.public_columns();
        let part_count = self.layout.public_parameter_part_count();
        let b_trapdoor = ring.sample_trapdoor(
            self.layout.secret_size,
            self.trapdoor_sigma.clone(),
            self.layout.gadget_base.clone(),
            self.layout.digit_count,
            self.preimage_max_coefficient_bound.clone(),
        );
        let t_bottom_full = ring.gaussian(
            (public_columns, part_count * public_columns),
            self.trapdoor_sigma.clone(),
            self.gaussian_max_coefficient_bound.clone(),
        );
        let t_bottom_parts = (0..part_count)
            .map(|part| {
                t_bottom_full.clone().slice(
                    None,
                    Some(IndexRange {
                        start: (part * public_columns).into(),
                        end: ((part + 1) * public_columns).into(),
                    }),
                )
            })
            .collect::<Vec<_>>();
        let t_bottom = Family::pack(t_bottom_parts.clone())?;
        let top_indices = (0..self.layout.gadget_rows())
            .flat_map(|digit_row| (0..part_count).map(move |part| (digit_row, part)))
            .collect::<Vec<_>>();
        let t_top = top_indices
            .into_iter()
            .map(|(digit_row, part)| {
                self.build_top_family(
                    hash_key.clone(),
                    &b_trapdoor,
                    &t_bottom_parts[part],
                    digit_row,
                    part,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Wee25PublicParameterPreprocessingWires {
            public_parameters: Wee25PublicParameterWires {
                b: b_trapdoor.public_matrix(),
                t_top,
                t_bottom,
            },
            b_trapdoor,
        })
    }

    pub fn export(
        &self,
        mut context: DslContext,
        wires: Wee25PublicParameterPreprocessingWires,
    ) -> Result<DslContext, DslError> {
        context = context.public_output(WEE25_PUBLIC_B, wires.public_parameters.b)?;
        context = context.private_trapdoor_output(WEE25_PUBLIC_B_TRAPDOOR, wires.b_trapdoor)?;
        context = context.public_family_output(WEE25_T_BOTTOM, wires.public_parameters.t_bottom)?;
        for (index, family) in wires.public_parameters.t_top.into_iter().enumerate() {
            let part_count = self.layout.public_parameter_part_count();
            context = context.public_family_output(
                self.layout.public_parameter_top_name(index / part_count, index % part_count),
                family,
            )?;
        }
        Ok(context)
    }

    fn build_top_family(
        &self,
        hash_key: Bytes,
        b: &Trapdoor,
        t_bottom: &Mat,
        digit_row: usize,
        part: usize,
    ) -> Result<Family<Mat>, Wee25CommitmentError> {
        let matrices = (0..self.layout.public_parameter_block_count())
            .map(|block| {
                let block_index = block * self.layout.gadget_rows() + digit_row;
                let mut tag = HashTag::from(b"wee25_w_block_".as_slice());
                tag.push(IntExpr::constant(block_index));
                let w = self.layout.ring().hash_matrix(
                    hash_key.clone(),
                    tag,
                    (self.layout.secret_size, self.layout.public_columns()),
                );
                let j = self.build_j_block(block, digit_row, part);
                let gadget = self.layout.ring().gadget(
                    self.layout.secret_size,
                    self.layout.gadget_base.clone(),
                    self.layout.digit_count,
                );
                let target = gadget * j - w * t_bottom.clone();
                b.sample_preimage(
                    target,
                    (self.layout.public_columns(), self.layout.public_columns()),
                )
                .as_mat()
            })
            .collect::<Vec<_>>();
        Ok(Family::pack(matrices)?)
    }

    fn build_j_block(&self, block: usize, digit_row: usize, part: usize) -> Mat {
        let ring = self.layout.ring();
        let d = self.layout.secret_size;
        let k = self.layout.digit_count;
        let m_g = self.layout.gadget_rows();
        let m_b = self.layout.public_columns();
        let rows = (0..d)
            .map(|secret_row| {
                let offset = digit_row * m_g + secret_row * k;
                let step = m_g + 1;
                let c = offset.div_ceil(step);
                let position = c * step;
                let terms = if c < m_g && position <= offset + k - 1 {
                    (0..k)
                        .filter_map(|s| {
                            let global_column = block * k + s;
                            let start = part * m_b;
                            (start..start + m_b).contains(&global_column).then(|| {
                                ring.constant(
                                    (1, 1),
                                    ConstantMatrix::PowerOfBase {
                                        base: self.layout.gadget_base.clone(),
                                        exponent: IntExpr::constant(position - offset + s),
                                    },
                                ) * ring.constant(
                                    (1, m_b),
                                    ConstantMatrix::UnitRow {
                                        index: IntExpr::constant(global_column - start),
                                    },
                                )
                            })
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let row = terms
                    .into_iter()
                    .reduce(|sum, term| sum + term)
                    .unwrap_or_else(|| ring.zero((1, m_b)));
                row.decompose(self.layout.gadget_base.clone(), k).as_mat()
            })
            .collect::<Vec<_>>();
        if rows.len() == 1 { rows[0].clone() } else { Mat::concat(ConcatAxis::Rows, rows) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output};
    use mxx_dsl::DslContext;

    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::RuntimeValue;
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
                let row_index = block_index * d + secret_row;
                let gadget_start = row_index * k;
                let slice_start = block_group * m_g * m_g;
                let offset = gadget_start - slice_start;
                let step = m_g + 1;
                let mut row = DCRTPolyMatrix::zero(parameters, 1, m_b);
                let diagonal = offset.div_ceil(step);
                if diagonal < m_g {
                    let position = slice_start + diagonal * step;
                    if position <= gadget_start + k - 1 {
                        let coefficient_digit = position - gadget_start;
                        for digit in 0..k {
                            let global_column = block_group * k + digit;
                            let start = part * m_b;
                            if (start..start + m_b).contains(&global_column) {
                                row.set_entry(
                                    0,
                                    global_column - start,
                                    gadget_row[coefficient_digit].clone() * &gadget_row[digit],
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
    fn runtime_parameters_preserve_every_chunk_relation_against_direct_j_and_hash_oracles() {
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
            gaussian_max_coefficient_bound: 30.into(),
            preimage_max_coefficient_bound: 1_000_000.into(),
        };
        let hash_key = [0x45; 32];
        let wires = compiler.build(layout.ring().bytes_input("hash-key", 32)).unwrap();
        let mut context = DslContext::new("wee25-public-parameter-runtime")
            .output("b", wires.public_parameters.b)
            .unwrap();
        for part in 0..layout.public_parameter_part_count() {
            context = context
                .output(format!("bottom-{part}"), wires.public_parameters.t_bottom.get_static(part))
                .unwrap();
        }
        for digit_row in 0..layout.gadget_rows() {
            for part in 0..layout.public_parameter_part_count() {
                let family = digit_row * layout.public_parameter_part_count() + part;
                for block in 0..layout.public_parameter_block_count() {
                    context = context
                        .output(
                            format!("top-{digit_row}-{part}-{block}"),
                            wires.public_parameters.t_top[family].get_static(block),
                        )
                        .unwrap();
                }
            }
        }
        let result = execute_graph(
            context.build().unwrap(),
            parameters.clone(),
            BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]),
        );
        let b = matrix_output(&result, "b");
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_size);
        let hash_sampler = HashSampler::new();
        for digit_row in 0..layout.gadget_rows() {
            for part in 0..layout.public_parameter_part_count() {
                let bottom = matrix_output(&result, &format!("bottom-{part}"));
                for block in 0..layout.public_parameter_block_count() {
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
                    let expected = gadget.clone() * &j - &(w * bottom);
                    assert_eq!(
                        b.clone() *
                            matrix_output(&result, &format!("top-{digit_row}-{part}-{block}"),),
                        expected,
                        "digit row {digit_row}, part {part}, block {block}"
                    );
                }
            }
        }
    }
}
