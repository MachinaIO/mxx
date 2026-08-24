use crate::{
    Graph, IntExpr, ParamEnv, RealExpr, ValidationError, WireType,
    expr::ExprError,
    node::{ConcatAxis, NodeKind},
    types::MatrixType,
};
use num_bigint::BigInt;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// A decidable parameter condition shared by concrete validation and operational checking.
///
/// These constraints intentionally contain only compile-time expressions. Scheduling,
/// liveness, concrete wire flow, and manifest checks remain validation-only concerns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ParamConstraint {
    IntPositive { value: IntExpr, label: String },
    IntNonnegative { value: IntExpr, label: String },
    IntGreaterThan { left: IntExpr, right: IntExpr, label: String },
    IntLessEqual { left: IntExpr, right: IntExpr, label: String },
    IntEqual { left: IntExpr, right: IntExpr, label: String },
    RealPositive { value: RealExpr, label: String },
    RealNonnegative { value: RealExpr, label: String },
}

impl ParamConstraint {
    pub fn evaluate(&self, env: &ParamEnv) -> Result<bool, ExprError> {
        Ok(match self {
            Self::IntPositive { value, .. } => value.evaluate(env)? > BigInt::zero(),
            Self::IntNonnegative { value, .. } => value.evaluate(env)? >= BigInt::zero(),
            Self::IntGreaterThan { left, right, .. } => {
                left.evaluate(env)? > right.evaluate(env)?
            }
            Self::IntLessEqual { left, right, .. } => left.evaluate(env)? <= right.evaluate(env)?,
            Self::IntEqual { left, right, .. } => left.evaluate(env)? == right.evaluate(env)?,
            Self::RealPositive { value, .. } => value.evaluate_f64(env)? > 0.0,
            Self::RealNonnegative { value, .. } => value.evaluate_f64(env)? >= 0.0,
        })
    }

    pub fn label(&self) -> &str {
        match self {
            Self::IntPositive { label, .. } |
            Self::IntNonnegative { label, .. } |
            Self::IntGreaterThan { label, .. } |
            Self::IntLessEqual { label, .. } |
            Self::IntEqual { label, .. } |
            Self::RealPositive { label, .. } |
            Self::RealNonnegative { label, .. } => label,
        }
    }

    fn contains_loop_index(&self) -> bool {
        match self {
            Self::IntPositive { value, .. } | Self::IntNonnegative { value, .. } => {
                int_contains_loop_index(value)
            }
            Self::IntGreaterThan { left, right, .. } |
            Self::IntLessEqual { left, right, .. } |
            Self::IntEqual { left, right, .. } => {
                int_contains_loop_index(left) || int_contains_loop_index(right)
            }
            Self::RealPositive { value, .. } | Self::RealNonnegative { value, .. } => {
                real_contains_loop_index(value)
            }
        }
    }
}

pub fn derive_param_constraints(graph: &Graph) -> Result<Vec<ParamConstraint>, ValidationError> {
    crate::validate::validate_structure(graph)?;
    let mut constraints = Vec::new();
    for (scope, graph_scope) in graph.scopes() {
        for (node_index, node) in graph_scope.nodes().iter().enumerate() {
            for wire_type in node.output_types() {
                derive_wire_constraints(wire_type, &mut constraints);
            }
            let prefix = format!("scope {scope:?}, node {node_index}");
            match node.kind() {
                NodeKind::UniformIntervalSample { range, .. } => {
                    constraints.push(ParamConstraint::IntLessEqual {
                        left: range.minimum.clone(),
                        right: range.maximum.clone(),
                        label: format!("{prefix}: uniform range must be nonempty"),
                    })
                }
                NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                    nonnegative_real(&mut constraints, sigma, format!("{prefix}: Gaussian sigma"));
                    nonnegative(
                        &mut constraints,
                        max_coefficient_bound,
                        format!("{prefix}: Gaussian coefficient bound"),
                    );
                }
                NodeKind::TrapdoorSample {
                    sigma,
                    gadget_base,
                    digit_count,
                    preimage_max_coefficient_bound,
                    ..
                } => {
                    positive_real(&mut constraints, sigma, format!("{prefix}: trapdoor sigma"));
                    constraints.push(ParamConstraint::IntGreaterThan {
                        left: gadget_base.clone(),
                        right: IntExpr::constant(1),
                        label: format!("{prefix}: gadget base must exceed one"),
                    });
                    positive(&mut constraints, digit_count, format!("{prefix}: digit count"));
                    nonnegative(
                        &mut constraints,
                        preimage_max_coefficient_bound,
                        format!("{prefix}: preimage coefficient bound"),
                    );
                }
                NodeKind::PreimageSample { max_coefficient_bound, .. } => nonnegative(
                    &mut constraints,
                    max_coefficient_bound,
                    format!("{prefix}: preimage coefficient bound"),
                ),
                NodeKind::GadgetTrapdoor { base, .. } | NodeKind::GadgetDecompose { base, .. } => {
                    constraints.push(ParamConstraint::IntGreaterThan {
                        left: base.clone(),
                        right: IntExpr::constant(1),
                        label: format!("{prefix}: gadget base must exceed one"),
                    })
                }
                NodeKind::Slice { rows, columns } => {
                    for (axis, range) in [("row", rows), ("column", columns)] {
                        if let Some(range) = range {
                            nonnegative(
                                &mut constraints,
                                &range.start,
                                format!("{prefix}: {axis} slice start"),
                            );
                            constraints.push(ParamConstraint::IntGreaterThan {
                                left: range.end.clone(),
                                right: range.start.clone(),
                                label: format!("{prefix}: {axis} slice must be nonempty"),
                            });
                        }
                    }
                }
                NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                    constraints.push(ParamConstraint::IntGreaterThan {
                        left: plaintext_modulus.clone(),
                        right: IntExpr::constant(1),
                        label: format!("{prefix}: plaintext modulus must exceed one"),
                    });
                    positive(&mut constraints, length, format!("{prefix}: decode length"));
                }
                NodeKind::FamilyPack { count } | NodeKind::Select { count } => {
                    positive(&mut constraints, count, format!("{prefix}: family count"));
                }
                NodeKind::FamilyGetStatic { index } |
                NodeKind::ExtractCoefficient { position: index, .. } |
                NodeKind::BitExtract { bit: index } => {
                    nonnegative(&mut constraints, index, format!("{prefix}: index"));
                }
                NodeKind::ParallelLoop(loop_spec) => {
                    nonnegative(
                        &mut constraints,
                        &loop_spec.count,
                        format!("{prefix}: loop count"),
                    );
                }
                NodeKind::SequentialLoop(loop_spec) => {
                    nonnegative(
                        &mut constraints,
                        &loop_spec.count,
                        format!("{prefix}: sequential loop count"),
                    );
                }
                NodeKind::Concat {
                    axis: ConcatAxis::Rows | ConcatAxis::Columns | ConcatAxis::Diagonal,
                } |
                NodeKind::LiftIntegerToConstantPolynomial { .. } |
                NodeKind::Input { .. } |
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::ConstantMatrix { .. } |
                NodeKind::TrapdoorPublic |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::IntToReal |
                NodeKind::BoolToInt |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt |
                NodeKind::MatrixBinary(_) |
                NodeKind::MatrixMulAccumulate { .. } |
                NodeKind::MatrixNegate |
                NodeKind::MatrixScale { .. } |
                NodeKind::Transpose |
                NodeKind::Tensor |
                NodeKind::UniformResidueSample { .. } |
                NodeKind::HashSample { .. } |
                NodeKind::CrtRecompose { .. } |
                NodeKind::PackPolynomialCoefficients { .. } |
                NodeKind::SubgraphCall(_) |
                NodeKind::FamilyGetDynamic => {}
            }
        }
    }
    // Loop indices are runtime binders rather than protocol parameters. Their range-dependent
    // checks remain in concrete scope validation and must not leak into `ParamsValid`.
    constraints.retain(|constraint| !constraint.contains_loop_index());
    Ok(constraints)
}

pub(crate) fn evaluate_param_constraints(
    graph: &Graph,
    env: &ParamEnv,
) -> Result<(), ValidationError> {
    for constraint in derive_param_constraints(graph)? {
        if !constraint.evaluate(env)? {
            return Err(ValidationError::ParameterConstraint(constraint.label().to_owned()));
        }
    }
    Ok(())
}

fn derive_wire_constraints(wire_type: &WireType, output: &mut Vec<ParamConstraint>) {
    match wire_type {
        WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
            constraints_for_matrix(matrix, output);
        }
        WireType::Trapdoor {
            matrix,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            ..
        } => {
            constraints_for_matrix(matrix, output);
            output.push(ParamConstraint::IntGreaterThan {
                left: gadget_base.clone(),
                right: IntExpr::constant(1),
                label: "trapdoor gadget base must exceed one".to_owned(),
            });
            positive(output, digit_count, "trapdoor digit count".to_owned());
            nonnegative(
                output,
                preimage_max_coefficient_bound,
                "trapdoor preimage coefficient bound".to_owned(),
            );
        }
        WireType::Bytes { length } => nonnegative(output, length, "byte length".to_owned()),
        WireType::TypedBlob { .. } => {}
        WireType::IndexedFamily { element, count } => {
            nonnegative(output, count, "family count".to_owned());
            derive_wire_constraints(element, output);
        }
        WireType::ConstantInt |
        WireType::Int |
        WireType::ConstantReal |
        WireType::Real |
        WireType::ConstantBool |
        WireType::Bool => {}
    }
}

fn constraints_for_matrix(matrix: &MatrixType, output: &mut Vec<ParamConstraint>) {
    constraints_for_positive(
        output,
        [
            (&matrix.modulus, "matrix modulus"),
            (&matrix.ring_dimension, "ring dimension"),
            (&matrix.rows, "matrix rows"),
            (&matrix.columns, "matrix columns"),
        ],
    );
    output.push(ParamConstraint::IntGreaterThan {
        left: matrix.modulus.clone(),
        right: IntExpr::constant(1),
        label: "matrix modulus must exceed one".to_owned(),
    });
}

fn constraints_for_positive<'a>(
    output: &mut Vec<ParamConstraint>,
    values: impl IntoIterator<Item = (&'a IntExpr, &'a str)>,
) {
    for (value, label) in values {
        positive(output, value, label.to_owned());
    }
}

fn positive(output: &mut Vec<ParamConstraint>, value: &IntExpr, label: String) {
    output.push(ParamConstraint::IntPositive { value: value.clone(), label });
}

fn nonnegative(output: &mut Vec<ParamConstraint>, value: &IntExpr, label: String) {
    output.push(ParamConstraint::IntNonnegative { value: value.clone(), label });
}

fn positive_real(output: &mut Vec<ParamConstraint>, value: &RealExpr, label: String) {
    output.push(ParamConstraint::RealPositive { value: value.clone(), label });
}

fn nonnegative_real(output: &mut Vec<ParamConstraint>, value: &RealExpr, label: String) {
    output.push(ParamConstraint::RealNonnegative { value: value.clone(), label });
}

fn int_contains_loop_index(value: &IntExpr) -> bool {
    match value {
        IntExpr::LoopIndex(_) => true,
        IntExpr::Add(left, right) |
        IntExpr::Sub(left, right) |
        IntExpr::Mul(left, right) |
        IntExpr::Div(left, right) |
        IntExpr::RoundDiv(left, right) => {
            int_contains_loop_index(left) || int_contains_loop_index(right)
        }
        IntExpr::Log2Ceil(value) => int_contains_loop_index(value),
        IntExpr::Const(_) | IntExpr::Var(_) => false,
    }
}

fn real_contains_loop_index(value: &RealExpr) -> bool {
    match value {
        RealExpr::FromInt(value) => int_contains_loop_index(value),
        RealExpr::Add(left, right) |
        RealExpr::Sub(left, right) |
        RealExpr::Mul(left, right) |
        RealExpr::Div(left, right) => {
            real_contains_loop_index(left) || real_contains_loop_index(right)
        }
        RealExpr::Sqrt(value) => real_contains_loop_index(value),
        RealExpr::Rational(_) | RealExpr::Var(_) => false,
    }
}
