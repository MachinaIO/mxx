//! Canonical binary transport for Graph IR values.
//!
//! The normative format is `docs/correctness/ir-binary-format-v1.md`.

use mxx_ir_core::{
    FrozenGraphScopeId, Graph, GraphScope, IntExpr, RealExpr, WireType,
    node::{ConstantMatrix, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::MatrixType,
};
use num_bigint::BigInt;
use std::collections::BTreeMap;
use thiserror::Error;

pub const IR_BINARY_FORMAT_VERSION: u8 = 1;
const DOCUMENT_PROG: u8 = 1;
const DOCUMENT_PROGRAM_DERIVATION: u8 = 2;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BinaryEncodeError {
    #[error("binary Graph IR field `{field}` exceeds u32")]
    U32Overflow { field: &'static str },
    #[error("frozen Graph IR contains an unresolved scope reference")]
    MissingScope,
    #[error("node output does not provide the matrix type required by its Lean transport")]
    MissingMatrixType,
}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
    strings: Vec<String>,
    string_indices: BTreeMap<String, u32>,
}

impl Encoder {
    fn u32(&mut self, value: usize, field: &'static str) -> Result<(), BinaryEncodeError> {
        let value = u32::try_from(value).map_err(|_| BinaryEncodeError::U32Overflow { field })?;
        self.bytes.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn raw_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn string(&mut self, value: &str) -> Result<(), BinaryEncodeError> {
        let index = if let Some(index) = self.string_indices.get(value) {
            *index
        } else {
            let index = u32::try_from(self.strings.len())
                .map_err(|_| BinaryEncodeError::U32Overflow { field: "string table" })?;
            self.strings.push(value.to_owned());
            self.string_indices.insert(value.to_owned(), index);
            index
        };
        self.raw_u32(index);
        Ok(())
    }

    fn array<T>(
        &mut self,
        values: &[T],
        mut encode: impl FnMut(&mut Self, &T) -> Result<(), BinaryEncodeError>,
    ) -> Result<(), BinaryEncodeError> {
        self.u32(values.len(), "array length")?;
        for value in values {
            encode(self, value)?;
        }
        Ok(())
    }

    fn record(
        &mut self,
        tag: u8,
        encode: impl FnOnce(&mut Self) -> Result<(), BinaryEncodeError>,
    ) -> Result<(), BinaryEncodeError> {
        let outer = std::mem::take(&mut self.bytes);
        encode(self)?;
        let payload = std::mem::replace(&mut self.bytes, outer);
        self.bytes.push(tag);
        self.u32(payload.len(), "TLV payload")?;
        self.bytes.extend(payload);
        Ok(())
    }

    fn int(&mut self, value: &BigInt) -> Result<(), BinaryEncodeError> {
        let bytes = value.to_signed_bytes_le();
        self.u32(bytes.len(), "integer byte length")?;
        self.bytes.extend(bytes);
        Ok(())
    }

    fn int_expr(&mut self, value: &IntExpr) -> Result<(), BinaryEncodeError> {
        match value {
            IntExpr::Const(value) => self.record(0, |out| out.int(value)),
            IntExpr::Var(name) => self.record(1, |out| out.string(name)),
            IntExpr::LoopIndex(slot) => self.record(2, |out| {
                out.raw_u32(*slot);
                Ok(())
            }),
            IntExpr::Add(left, right) => self.binary_expr(3, left, right),
            IntExpr::Sub(left, right) => self.binary_expr(4, left, right),
            IntExpr::Mul(left, right) => self.binary_expr(5, left, right),
            IntExpr::Div(left, right) => self.binary_expr(6, left, right),
            IntExpr::RoundDiv(left, right) => self.binary_expr(7, left, right),
            IntExpr::Log2Ceil(value) => self.record(8, |out| out.int_expr(value)),
        }
    }

    fn binary_expr(
        &mut self,
        tag: u8,
        left: &IntExpr,
        right: &IntExpr,
    ) -> Result<(), BinaryEncodeError> {
        self.record(tag, |out| {
            out.int_expr(left)?;
            out.int_expr(right)
        })
    }

    fn real_expr(&mut self, value: &RealExpr) -> Result<(), BinaryEncodeError> {
        match value {
            RealExpr::Rational(value) => self.record(0, |out| {
                out.int(value.numerator())?;
                out.int(value.denominator())
            }),
            RealExpr::Var(name) => self.record(1, |out| out.string(name)),
            RealExpr::FromInt(value) => self.record(2, |out| out.int_expr(value)),
            RealExpr::Add(left, right) => self.binary_real_expr(3, left, right),
            RealExpr::Sub(left, right) => self.binary_real_expr(4, left, right),
            RealExpr::Mul(left, right) => self.binary_real_expr(5, left, right),
            RealExpr::Div(left, right) => self.binary_real_expr(6, left, right),
            RealExpr::Sqrt(value) => self.record(7, |out| out.real_expr(value)),
        }
    }

    fn binary_real_expr(
        &mut self,
        tag: u8,
        left: &RealExpr,
        right: &RealExpr,
    ) -> Result<(), BinaryEncodeError> {
        self.record(tag, |out| {
            out.real_expr(left)?;
            out.real_expr(right)
        })
    }

    fn matrix_type(&mut self, value: &MatrixType) -> Result<(), BinaryEncodeError> {
        self.int_expr(&value.modulus)?;
        self.int_expr(&value.ring_dimension)?;
        self.int_expr(&value.rows)?;
        self.int_expr(&value.columns)
    }

    fn wire_type(&mut self, value: &WireType) -> Result<(), BinaryEncodeError> {
        match value {
            WireType::ConstantInt => self.record(0, |_| Ok(())),
            WireType::ConstantReal => self.record(1, |_| Ok(())),
            WireType::ConstantBool => self.record(2, |_| Ok(())),
            WireType::Int => self.record(3, |_| Ok(())),
            WireType::Real => self.record(4, |_| Ok(())),
            WireType::Bool => self.record(5, |_| Ok(())),
            WireType::Bytes { length } => self.record(6, |out| out.int_expr(length)),
            WireType::TypedBlob { type_name, schema_hash } => self.record(7, |out| {
                out.string(type_name)?;
                out.u32(schema_hash.len(), "schema hash")?;
                out.bytes.extend_from_slice(schema_hash);
                Ok(())
            }),
            WireType::Matrix(value) => self.record(8, |out| out.matrix_type(value)),
            WireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => self.record(9, |out| {
                out.matrix_type(matrix)?;
                out.real_expr(sigma)?;
                out.int_expr(gadget_base)?;
                out.int_expr(digit_count)?;
                out.int_expr(preimage_max_coefficient_bound)
            }),
            WireType::Preimage(value) => self.record(10, |out| out.matrix_type(value)),
            WireType::IndexedFamily { element, count } => self.record(11, |out| {
                out.wire_type(element)?;
                out.int_expr(count)
            }),
        }
    }

    fn wire(&mut self, value: &mxx_ir_core::WireRef) -> Result<(), BinaryEncodeError> {
        let node = u32::try_from(value.node.0)
            .map_err(|_| BinaryEncodeError::U32Overflow { field: "wire node" })?;
        self.raw_u32(node);
        self.raw_u32(value.port.0);
        Ok(())
    }

    fn option_expr(&mut self, value: &Option<IntExpr>) -> Result<(), BinaryEncodeError> {
        if let Some(value) = value {
            self.bytes.push(1);
            self.int_expr(value)
        } else {
            self.bytes.push(0);
            Ok(())
        }
    }

    fn bindings(&mut self, values: &[(String, IntExpr)]) -> Result<(), BinaryEncodeError> {
        self.array(values, |out, (name, value)| {
            out.string(name)?;
            out.int_expr(value)
        })
    }

    fn node_kind(
        &mut self,
        graph: &Graph,
        scope_id: &FrozenGraphScopeId,
        node_id: mxx_ir_core::NodeId,
        node: &mxx_ir_core::NodeHandle,
    ) -> Result<(), BinaryEncodeError> {
        let kind = node.kind();
        match kind {
            NodeKind::Input { name, .. } => self.record(0, |out| out.string(name)),
            NodeKind::ConstantInt(value) => self.record(1, |out| out.int(value)),
            NodeKind::EvaluateInt(value) => self.record(2, |out| out.int_expr(value)),
            NodeKind::ConstantReal(value) => self.record(3, |out| out.real_expr(value)),
            NodeKind::ConstantBool(value) => self.record(4, |out| {
                out.bytes.push(u8::from(*value));
                Ok(())
            }),
            NodeKind::ConstantMatrix { matrix_type, value } => match value {
                ConstantMatrix::Zero => self.record(5, |out| out.matrix_type(matrix_type)),
                ConstantMatrix::Identity => self.record(6, |out| out.matrix_type(matrix_type)),
                ConstantMatrix::Polynomial { coefficients } => self.record(7, |out| {
                    out.matrix_type(matrix_type)?;
                    out.array(coefficients, |out, value| out.int_expr(value))
                }),
                ConstantMatrix::UnitRow { index } => self.record(8, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(index)
                }),
                ConstantMatrix::UnitColumn { index } => self.record(9, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(index)
                }),
                ConstantMatrix::Gadget { base, small: false } => self.record(10, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(base)
                }),
                ConstantMatrix::Gadget { base, small: true } => self.record(11, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(base)
                }),
                ConstantMatrix::PowerOfBase { base, exponent } => self.record(12, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(base)?;
                    out.int_expr(exponent)
                }),
                ConstantMatrix::Rotation { exponent } => self.record(13, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(exponent)
                }),
            },
            NodeKind::GadgetTrapdoor { matrix_type, base } => self.record(14, |out| {
                out.matrix_type(matrix_type)?;
                out.int_expr(base)
            }),
            NodeKind::BoolToInt => self.record(15, |_| Ok(())),
            NodeKind::IntToReal => self.record(16, |_| Ok(())),
            NodeKind::IntBinary(op) => self.record(17, |out| {
                out.bytes.push(*op as u8);
                Ok(())
            }),
            NodeKind::RealBinary(op) => self.record(18, |out| {
                out.bytes.push(*op as u8);
                Ok(())
            }),
            NodeKind::RealSqrt => self.record(19, |_| Ok(())),
            NodeKind::IntCompare(op) => self.record(20, |out| {
                out.bytes.push(*op as u8);
                Ok(())
            }),
            NodeKind::BitExtract { bit } => self.record(21, |out| out.int_expr(bit)),
            NodeKind::ExtractCoefficient { position } => {
                self.record(22, |out| out.int_expr(position))
            }
            NodeKind::ConstantCoefficient { position } => {
                self.record(23, |out| out.int_expr(position))
            }
            NodeKind::Select { .. } => self.record(24, |_| Ok(())),
            NodeKind::UniformResidueSample { matrix_type } => {
                self.record(25, |out| out.matrix_type(matrix_type))
            }
            NodeKind::UniformIntervalSample { matrix_type, range } => self.record(26, |out| {
                out.matrix_type(matrix_type)?;
                out.int_expr(&range.minimum)?;
                out.int_expr(&range.maximum)
            }),
            NodeKind::GaussianSample { matrix_type, max_coefficient_bound, .. } => {
                self.record(27, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(max_coefficient_bound)
                })
            }
            NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
            } => self.record(28, |out| {
                out.matrix_type(matrix_type)?;
                out.bytes.push(*variant as u8);
                out.u32(tag_prefix.len(), "tag prefix")?;
                out.bytes.extend_from_slice(tag_prefix);
                out.array(tag_expressions, |o, v| o.int_expr(v))?;
                out.array(tag_decimal_expressions, |o, v| o.int_expr(v))?;
                out.array(tag_u64_le_expressions, |o, v| o.int_expr(v))?;
                out.option_expr(base)?;
                out.option_expr(digit_count)
            }),
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let matrix = node
                    .output_types()
                    .first()
                    .and_then(|value| match value {
                        WireType::Matrix(m) | WireType::Preimage(m) => Some(m),
                        _ => None,
                    })
                    .ok_or(BinaryEncodeError::MissingMatrixType)?;
                self.record(29, |out| {
                    out.matrix_type(matrix)?;
                    out.int_expr(base)?;
                    out.bytes.push(u8::from(*small));
                    out.int_expr(digit_count)
                })
            }
            NodeKind::TrapdoorSample { matrix_type, preimage_max_coefficient_bound, .. } => self
                .record(30, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(preimage_max_coefficient_bound)
                }),
            NodeKind::TrapdoorPublic => self.record(31, |_| Ok(())),
            NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => {
                self.record(32, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(max_coefficient_bound)
                })
            }
            NodeKind::MatrixBinary(op) => self.record(
                match op {
                    MatrixBinaryOp::Add => 33,
                    MatrixBinaryOp::Subtract => 34,
                    MatrixBinaryOp::Multiply => 35,
                },
                |_| Ok(()),
            ),
            NodeKind::MatrixNegate => self.record(36, |_| Ok(())),
            NodeKind::MatrixScale { scalar } => self.record(37, |out| out.int_expr(scalar)),
            NodeKind::Transpose => self.record(38, |_| Ok(())),
            NodeKind::Slice { rows, columns } => self.record(39, |out| {
                for range in [rows, columns] {
                    if let Some(range) = range {
                        out.bytes.push(1);
                        out.int_expr(&range.start)?;
                        out.int_expr(&range.end)?;
                    } else {
                        out.bytes.push(0);
                    }
                }
                Ok(())
            }),
            NodeKind::Tensor => self.record(40, |_| Ok(())),
            NodeKind::Reshape { rows, columns } => self.record(41, |out| {
                out.int_expr(rows)?;
                out.int_expr(columns)
            }),
            NodeKind::Concat { axis } => self.record(42, |out| {
                out.bytes.push(*axis as u8);
                Ok(())
            }),
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let scope = graph.scope(scope_id).ok_or(BinaryEncodeError::MissingScope)?;
                let modulus = scope
                    .arguments(node)
                    .and_then(|a| a.first().copied())
                    .and_then(|w| scope.node(w.node))
                    .and_then(|n| n.output_types().first())
                    .and_then(|t| match t {
                        WireType::Matrix(m) | WireType::Preimage(m) => Some(&m.modulus),
                        WireType::Trapdoor { matrix, .. } => Some(&matrix.modulus),
                        _ => None,
                    })
                    .ok_or(BinaryEncodeError::MissingMatrixType)?;
                self.record(if *output_bool { 43 } else { 44 }, |out| {
                    out.int_expr(modulus)?;
                    out.int_expr(plaintext_modulus)?;
                    out.int_expr(length)
                })
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => self
                .record(45, |out| {
                    out.array(plaintext_moduli, |o, v| o.int_expr(v))?;
                    out.array(reconstruction_coefficients, |o, v| o.int_expr(v))
                }),
            NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => {
                self.record(46, |out| {
                    out.matrix_type(matrix_type)?;
                    out.int_expr(coefficient_bits)
                })
            }
            NodeKind::FamilyPack { .. } => self.record(47, |_| Ok(())),
            NodeKind::FamilyGetStatic { index } => self.record(48, |out| out.int_expr(index)),
            NodeKind::FamilyGetDynamic => self.record(49, |_| Ok(())),
            NodeKind::SubgraphCall(call) => {
                let child = graph
                    .child_scope_id(scope_id, node_id)
                    .ok_or(BinaryEncodeError::MissingScope)?;
                self.record(50, |out| {
                    out.string(&scope_name(&child))?;
                    out.bindings(&call.bindings)
                })
            }
            NodeKind::ParallelLoop(spec) => {
                let child = graph
                    .child_scope_id(scope_id, node_id)
                    .ok_or(BinaryEncodeError::MissingScope)?;
                self.record(51, |out| {
                    out.string(&scope_name(&child))?;
                    out.int_expr(&spec.count)?;
                    out.raw_u32(spec.index_slot);
                    out.bindings(&spec.bindings)?;
                    out.array(&spec.input_modes, |o, m| {
                        match m {
                            LoopInputMode::Broadcast => o.bytes.push(0),
                            LoopInputMode::Zip => o.bytes.push(1),
                            LoopInputMode::ZipOffset { offset } => {
                                o.bytes.push(2);
                                o.u32(*offset, "zip offset")?;
                            }
                        }
                        Ok(())
                    })
                })
            }
            NodeKind::SequentialLoop(spec) => {
                let child = graph
                    .child_scope_id(scope_id, node_id)
                    .ok_or(BinaryEncodeError::MissingScope)?;
                self.record(52, |out| {
                    out.string(&scope_name(&child))?;
                    out.int_expr(&spec.count)?;
                    out.raw_u32(spec.index_slot);
                    out.bindings(&spec.bindings)?;
                    out.u32(spec.carried_count, "carried count")
                })
            }
        }
    }

    fn scope(
        &mut self,
        graph: &Graph,
        id: &FrozenGraphScopeId,
        scope: &GraphScope,
    ) -> Result<(), BinaryEncodeError> {
        self.u32(scope.nodes().len(), "node count")?;
        for (index, node) in scope.nodes().iter().enumerate() {
            self.node_kind(graph, id, mxx_ir_core::NodeId(index as u64), node)?;
            let args = scope.arguments(node).ok_or(BinaryEncodeError::MissingScope)?;
            self.array(&args, |out, value| out.wire(value))?;
            self.u32(node.output_types().len(), "output count")?;
            self.array(node.output_types(), |out, value| out.wire_type(value))?;
        }
        if matches!(id, FrozenGraphScopeId::Root) {
            self.u32(graph.outputs().len(), "output count")?;
            for (name, output) in graph.outputs() {
                self.string(name)?;
                self.wire(&output.value)?;
            }
        } else {
            self.u32(scope.outputs().len(), "output count")?;
            for (index, output) in scope.outputs().iter().enumerate() {
                self.string(&format!("output-{index}"))?;
                self.wire(output)?;
            }
        }
        let inputs = if matches!(id, FrozenGraphScopeId::Root) {
            scope
                .nodes()
                .iter()
                .filter_map(|n| match n.kind() {
                    NodeKind::Input { name, .. } => Some(name),
                    _ => None,
                })
                .collect::<Vec<_>>()
        } else {
            scope
                .inputs()
                .iter()
                .map(|wire| match scope.node(wire.node).expect("validated scope input").kind() {
                    NodeKind::Input { name, .. } => name,
                    _ => unreachable!("scope input node"),
                })
                .collect()
        };
        self.u32(inputs.len(), "input count")?;
        for input in inputs {
            self.string(input)?;
        }
        Ok(())
    }

    fn derivation_rule(
        &mut self,
        node: &mxx_ir_core::NodeHandle,
        scope: &GraphScope,
    ) -> Result<(), BinaryEncodeError> {
        let (tag, relation) = match node.kind() {
            NodeKind::Input { .. } => (0, None),
            NodeKind::ConstantInt(_) => (1, None),
            NodeKind::EvaluateInt(_) => (2, None),
            NodeKind::ConstantReal(_) => (3, None),
            NodeKind::ConstantBool(_) => (4, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::Zero, .. } => (5, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. } => (6, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::Polynomial { .. }, .. } => (7, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::UnitRow { .. }, .. } => (8, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::UnitColumn { .. }, .. } => (9, None),
            NodeKind::ConstantMatrix {
                value: ConstantMatrix::Gadget { small: false, .. }, ..
            } => (10, None),
            NodeKind::ConstantMatrix {
                value: ConstantMatrix::Gadget { small: true, .. }, ..
            } => (11, None),
            NodeKind::ConstantMatrix { value: ConstantMatrix::PowerOfBase { .. }, .. } => {
                (12, None)
            }
            NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { .. }, .. } => (13, None),
            NodeKind::GadgetTrapdoor { .. } => (14, None),
            NodeKind::IntToReal => (15, None),
            NodeKind::BoolToInt => (16, None),
            NodeKind::IntBinary(_) => (17, None),
            NodeKind::RealBinary(_) => (18, None),
            NodeKind::RealSqrt => (19, None),
            NodeKind::IntCompare(_) => (20, None),
            NodeKind::BitExtract { .. } => (21, None),
            NodeKind::ExtractCoefficient { .. } => (22, None),
            NodeKind::ConstantCoefficient { .. } => (23, None),
            NodeKind::Select { .. } => (24, None),
            NodeKind::UniformResidueSample { .. } => (25, None),
            NodeKind::UniformIntervalSample { .. } => (26, None),
            NodeKind::GaussianSample { .. } => (27, None),
            NodeKind::HashSample { .. } => (28, None),
            NodeKind::GadgetDecompose { .. } => (29, None),
            NodeKind::TrapdoorSample { .. } => (30, None),
            NodeKind::TrapdoorPublic => (31, None),
            NodeKind::PreimageSample { .. } => (32, None),
            NodeKind::MatrixBinary(MatrixBinaryOp::Add) => (33, None),
            NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => (34, None),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
                let arguments = scope.arguments(node).expect("validated multiplication arguments");
                let left = arguments[0];
                let right = arguments[1];
                let relation = match scope.node(right.node).map(|producer| producer.kind()) {
                    Some(NodeKind::GadgetDecompose { .. }) => {
                        scope.node(left.node).is_some_and(|producer| {
                            matches!(
                                producer.kind(),
                                NodeKind::ConstantMatrix {
                                    value: ConstantMatrix::Gadget { .. },
                                    ..
                                }
                            )
                        })
                    }
                    Some(NodeKind::PreimageSample { .. }) => {
                        scope
                            .node(right.node)
                            .and_then(|producer| scope.arguments(producer))
                            .and_then(|arguments| arguments.first().copied()) ==
                            Some(left)
                    }
                    _ => false,
                };
                if relation { (36, Some(right)) } else { (35, None) }
            }
            NodeKind::MatrixNegate => (37, None),
            NodeKind::MatrixScale { .. } => (38, None),
            NodeKind::Transpose => (39, None),
            NodeKind::Slice { .. } => (40, None),
            NodeKind::Tensor => (41, None),
            NodeKind::Reshape { .. } => (42, None),
            NodeKind::Concat { .. } => (43, None),
            NodeKind::ThresholdDecode { output_bool: true, .. } => (44, None),
            NodeKind::ThresholdDecode { output_bool: false, .. } => (45, None),
            NodeKind::CrtRecompose { .. } => (46, None),
            NodeKind::PackPolynomialCoefficients { .. } => (47, None),
            NodeKind::FamilyPack { .. } => (48, None),
            NodeKind::FamilyGetStatic { .. } => (49, None),
            NodeKind::FamilyGetDynamic => (50, None),
            NodeKind::SubgraphCall(_) => (51, None),
            NodeKind::ParallelLoop(_) => (52, None),
            NodeKind::SequentialLoop(_) => (53, None),
        };
        self.record(tag, |out| if let Some(wire) = relation { out.wire(&wire) } else { Ok(()) })
    }

    fn scope_derivation(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        attachments: Option<&mxx_dsl::FrozenDerivationAttachments>,
    ) -> Result<(), BinaryEncodeError> {
        self.u32(scope.nodes().len(), "derivation step count")?;
        for (source_node, node) in scope.nodes().iter().enumerate() {
            self.u32(source_node, "source node")?;
            self.derivation_rule(node, scope)?;
            let arguments = scope.arguments(node).ok_or(BinaryEncodeError::MissingScope)?;
            self.array(&arguments, |out, wire| out.wire(wire))?;
        }
        let entries = attachments
            .into_iter()
            .flat_map(|values| values.iter())
            .filter(|entry| entry.roles.first().is_some_and(|(_, wire)| &wire.scope == scope_id))
            .collect::<Vec<_>>();
        self.u32(entries.len(), "attachment count")?;
        for entry in entries {
            if entry.roles.iter().any(|(_, wire)| &wire.scope != scope_id) {
                return Err(BinaryEncodeError::MissingScope);
            }
            self.string(&entry.namespace)?;
            self.string(&entry.rule)?;
            self.u32(entry.roles.len(), "attachment role count")?;
            for (role, wire) in &entry.roles {
                self.string(role)?;
                self.wire(&wire.wire)?;
            }
        }
        Ok(())
    }
}

fn scope_name(id: &FrozenGraphScopeId) -> String {
    match id {
        FrozenGraphScopeId::Root => "__root".to_owned(),
        FrozenGraphScopeId::Subgraph { canonical_name } => format!("subgraph:{canonical_name}"),
        FrozenGraphScopeId::ParallelBody { parent, owner } => {
            format!("parallel:{}:{}", scope_name(parent), owner.0)
        }
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            format!("sequential:{}:{}", scope_name(parent), owner.0)
        }
    }
}

fn finish_document(encoder: Encoder, kind: u8) -> Result<Vec<u8>, BinaryEncodeError> {
    let payload = encoder.bytes;
    let mut output = vec![IR_BINARY_FORMAT_VERSION, kind];
    output.extend_from_slice(
        &u32::try_from(payload.len())
            .map_err(|_| BinaryEncodeError::U32Overflow { field: "document payload" })?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u32::try_from(encoder.strings.len())
            .map_err(|_| BinaryEncodeError::U32Overflow { field: "string count" })?
            .to_le_bytes(),
    );
    let blob_len = encoder.strings.iter().map(String::len).sum::<usize>();
    output.extend_from_slice(
        &u32::try_from(blob_len)
            .map_err(|_| BinaryEncodeError::U32Overflow { field: "string blob" })?
            .to_le_bytes(),
    );
    let mut offset = 0_u32;
    output.extend_from_slice(&offset.to_le_bytes());
    for value in &encoder.strings {
        offset = offset
            .checked_add(
                u32::try_from(value.len())
                    .map_err(|_| BinaryEncodeError::U32Overflow { field: "string" })?,
            )
            .ok_or(BinaryEncodeError::U32Overflow { field: "string offsets" })?;
        output.extend_from_slice(&offset.to_le_bytes());
    }
    for value in &encoder.strings {
        output.extend_from_slice(value.as_bytes());
    }
    output.extend(payload);
    Ok(output)
}

pub fn encode_prog(graph: &Graph) -> Result<Vec<u8>, BinaryEncodeError> {
    let mut encoder = Encoder::default();
    encoder.scope(graph, &FrozenGraphScopeId::Root, graph.root_scope())?;
    let definitions = graph
        .scopes()
        .iter()
        .filter(|(id, _)| !matches!(id, FrozenGraphScopeId::Root))
        .collect::<Vec<_>>();
    encoder.u32(definitions.len(), "definition count")?;
    for (id, scope) in definitions {
        encoder.string(&scope_name(id))?;
        encoder.scope(graph, id, scope)?;
    }
    finish_document(encoder, DOCUMENT_PROG)
}

pub fn encode_program_derivation(
    graph: &Graph,
    attachments: Option<&mxx_dsl::FrozenDerivationAttachments>,
) -> Result<Vec<u8>, BinaryEncodeError> {
    let mut encoder = Encoder::default();
    encoder.scope_derivation(&FrozenGraphScopeId::Root, graph.root_scope(), attachments)?;
    let definitions = graph
        .scopes()
        .iter()
        .filter(|(id, _)| !matches!(id, FrozenGraphScopeId::Root))
        .collect::<Vec<_>>();
    encoder.u32(definitions.len(), "derivation definition count")?;
    for (id, scope) in definitions {
        encoder.string(&scope_name(id))?;
        encoder.scope_derivation(id, scope, attachments)?;
    }
    finish_document(encoder, DOCUMENT_PROGRAM_DERIVATION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_program_encoding_is_deterministic_and_versioned() {
        let protocol = crate::toy_example::protocol();
        for stage in &protocol.bundle.workflow.stages {
            let first = encode_prog(&stage.graph).unwrap();
            let second = encode_prog(&stage.graph).unwrap();
            assert_eq!(first, second);
            assert_eq!(first[0], IR_BINARY_FORMAT_VERSION);
            assert_eq!(first[1], DOCUMENT_PROG);

            let first =
                encode_program_derivation(&stage.graph, Some(&stage.derivation_attachments))
                    .unwrap();
            let second =
                encode_program_derivation(&stage.graph, Some(&stage.derivation_attachments))
                    .unwrap();
            assert_eq!(first, second);
            assert_eq!(first[0], IR_BINARY_FORMAT_VERSION);
            assert_eq!(first[1], DOCUMENT_PROGRAM_DERIVATION);
        }
    }
}
