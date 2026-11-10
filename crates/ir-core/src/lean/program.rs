//! IR graphs as Lean data.
//!
//! [`program_term`] writes a validated graph as a `MxxSecurity.IR.Program`, the deep embedding of
//! the IR in `crates/ir-core/lean/security/MxxSecurity/IR`: one `Op` per node, in an order where
//! every argument precedes its use, with every compile parameter replaced by its value. Integer,
//! real, and ring expressions that depend on a loop index stay symbolic. A subgraph called with
//! different bindings becomes one scope per binding, so a call carries no bindings.
//!
//! Unlike the relational export of [`super::export`], the term has no meaning by itself; Lean
//! gives it one with `MxxSecurity.IR.Program.run` and a cost with `MxxSecurity.IR.Program.cost`.

use super::{ExportError, PACKED_POLYNOMIAL_MIN, packed_table};
use crate::{
    ParamEnv,
    expr::{IntExpr, RealExpr},
    graph::{FrozenGraphScopeId, Graph, GraphScope},
    node::{
        ConcatAxis, ConstantMatrix, HashTagComponent, HashVariant, IntBinaryOp, IntCompareOp,
        LoopInputMode, MatrixBinaryOp, NodeKind, RealBinaryOp,
    },
    ring::{RingExpr, RingRef},
    types::{CoefficientBoundDomain, MatrixType, NodeId, TrapdoorInstantiation, WireRef, WireType},
    validate::ValidatedGraph,
};
use num_bigint::BigUint;
use std::collections::BTreeMap;

/// The Lean terms of the variables in scope: compile parameters bound by a loop or a call.
type Vars = BTreeMap<String, String>;

struct Writer<'a> {
    graph: &'a Graph,
    bindings: &'a ParamEnv,
    /// Scope instances in order of creation, by scope and the bindings of its variables.
    instances: Vec<(FrozenGraphScopeId, Vars)>,
    scopes: Vec<Option<String>>,
}

fn encoding(message: impl Into<String>) -> ExportError {
    ExportError::Encoding(message.into())
}

fn list(items: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", items.into_iter().collect::<Vec<_>>().join(", "))
}

fn option(item: Option<String>) -> String {
    item.map_or("none".into(), |item| format!("(some {item})"))
}

fn bytes(values: &[u8]) -> String {
    list(values.iter().map(|value| value.to_string()))
}

impl Writer<'_> {
    fn int(&self, expr: &IntExpr, vars: &Vars) -> Result<String, ExportError> {
        if let Ok(value) = expr.evaluate(self.bindings) {
            return Ok(format!("(.const ({value}))"));
        }
        let pair = |tag: &str, l: &IntExpr, r: &IntExpr| -> Result<String, ExportError> {
            Ok(format!("(.{tag} {} {})", self.int(l, vars)?, self.int(r, vars)?))
        };
        match expr {
            IntExpr::Const(value) => Ok(format!("(.const ({value}))")),
            IntExpr::Var(name) => vars
                .get(name)
                .cloned()
                .ok_or_else(|| encoding(format!("unbound integer parameter {name}"))),
            IntExpr::LoopIndex(slot) => Ok(format!("(.loopIndex {slot})")),
            IntExpr::Add(l, r) => pair("add", l, r),
            IntExpr::Sub(l, r) => pair("sub", l, r),
            IntExpr::Mul(l, r) => pair("mul", l, r),
            IntExpr::Div(l, r) => pair("div", l, r),
            IntExpr::FloorDiv(l, r) => pair("floorDiv", l, r),
            IntExpr::Rem(l, r) => pair("rem", l, r),
            IntExpr::RoundDiv(l, r) => pair("roundDiv", l, r),
            IntExpr::Log2Ceil(value) => Ok(format!("(.log2Ceil {})", self.int(value, vars)?)),
            IntExpr::Select { selector, branches } => Ok(format!(
                "(.select {} {})",
                self.int(selector, vars)?,
                list(branches.iter().map(|b| self.int(b, vars)).collect::<Result<Vec<_>, _>>()?)
            )),
            IntExpr::RingModulus(_) | IntExpr::RingCrtDepth(_) | IntExpr::RingCrtModulus { .. } => {
                Err(encoding(format!("ring property {expr:?} depends on a loop index")))
            }
        }
    }

    fn real(&self, expr: &RealExpr, vars: &Vars) -> Result<String, ExportError> {
        let pair = |tag: &str, l: &RealExpr, r: &RealExpr| -> Result<String, ExportError> {
            Ok(format!("(.{tag} {} {})", self.real(l, vars)?, self.real(r, vars)?))
        };
        match expr {
            RealExpr::Rational(value) => {
                let (numerator, denominator) = (value.numerator(), value.denominator());
                Ok(format!("(.rational (({numerator} : Rat) / {denominator}))"))
            }
            RealExpr::Var(name) => Err(encoding(format!("unbound real parameter {name}"))),
            RealExpr::FromInt(value) => Ok(format!("(.ofInt {})", self.int(value, vars)?)),
            RealExpr::Add(l, r) => pair("add", l, r),
            RealExpr::Sub(l, r) => pair("sub", l, r),
            RealExpr::Mul(l, r) => pair("mul", l, r),
            RealExpr::Div(l, r) => pair("div", l, r),
            RealExpr::Sqrt(value) => Ok(format!("(.sqrt {})", self.real(value, vars)?)),
        }
    }

    fn ring(&self, ring: &RingRef, vars: &Vars) -> Result<String, ExportError> {
        if let Ok(concrete) = ring.resolve(self.bindings) {
            return Ok(format!(
                "(.explicit {} {})",
                list(concrete.crt_moduli().iter().map(|modulus| format!("(.const {modulus})"))),
                concrete.ring_dimension()
            ));
        }
        match ring.expression() {
            RingExpr::Explicit { crt_moduli, ring_dimension } => Ok(format!(
                "(.explicit {} {ring_dimension})",
                list(crt_moduli.iter().map(|m| self.int(m, vars)).collect::<Result<Vec<_>, _>>()?)
            )),
            RingExpr::Slice { source, start, end } => Ok(format!(
                "(.slice {} {} {})",
                self.ring(source, vars)?,
                self.int(start, vars)?,
                self.int(end, vars)?
            )),
            RingExpr::Select { source, indices } => Ok(format!(
                "(.select {} {})",
                self.ring(source, vars)?,
                list(indices.iter().map(|i| self.int(i, vars)).collect::<Result<Vec<_>, _>>()?)
            )),
            RingExpr::Concat { left, right } => {
                Ok(format!("(.concat {} {})", self.ring(left, vars)?, self.ring(right, vars)?))
            }
            RingExpr::Generated { .. } => Err(encoding("a generated ring depends on a loop index")),
        }
    }

    fn matrix_type(&self, matrix: &MatrixType, vars: &Vars) -> Result<String, ExportError> {
        Ok(format!(
            "⟨{}, {}, {}⟩",
            self.ring(&matrix.ring, vars)?,
            self.int(&matrix.rows, vars)?,
            self.int(&matrix.columns, vars)?
        ))
    }

    fn wire_type(&self, ty: &WireType, vars: &Vars) -> Result<String, ExportError> {
        let domain = |domain: &CoefficientBoundDomain| match domain {
            CoefficientBoundDomain::Global => ".global",
            CoefficientBoundDomain::PerCrtLimb => ".perCrtLimb",
        };
        Ok(match ty {
            WireType::ConstantInt | WireType::Int => ".int".into(),
            WireType::ConstantReal | WireType::Real => ".real".into(),
            WireType::ConstantBool | WireType::Bool => ".bool".into(),
            WireType::Bytes { length } => format!("(.bytes {})", self.int(length, vars)?),
            WireType::TypedBlob { type_name, .. } => format!("(.blob {type_name:?})"),
            WireType::Matrix(matrix) => format!("(.matrix {})", self.matrix_type(matrix, vars)?),
            WireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                instantiation,
                preimage_sigma,
            } => format!(
                "(.trapdoor {} {} {} {} {} {} {})",
                self.matrix_type(matrix, vars)?,
                self.real(sigma, vars)?,
                self.int(gadget_base, vars)?,
                self.int(digit_count, vars)?,
                self.int(preimage_max_coefficient_bound, vars)?,
                statistical(instantiation),
                option(preimage_sigma.as_ref().map(|s| self.real(s, vars)).transpose()?)
            ),
            WireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain } => format!(
                "(.small {} {} {})",
                self.matrix_type(matrix, vars)?,
                self.int(max_coefficient_bound, vars)?,
                domain(bound_domain)
            ),
            WireType::Preimage { matrix, max_coefficient_bound, bound_domain } => format!(
                "(.preimage {} {} {})",
                self.matrix_type(matrix, vars)?,
                self.int(max_coefficient_bound, vars)?,
                domain(bound_domain)
            ),
            WireType::IndexedFamily { element, count } => {
                format!("(.family {} {})", self.wire_type(element, vars)?, self.int(count, vars)?)
            }
        })
    }

    fn constant_matrix(&self, value: &ConstantMatrix, vars: &Vars) -> Result<String, ExportError> {
        Ok(match value {
            ConstantMatrix::Zero => ".zero".into(),
            ConstantMatrix::Identity => ".identity".into(),
            ConstantMatrix::UnitRow { index } => format!("(.unitRow {})", self.int(index, vars)?),
            ConstantMatrix::UnitColumn { index } => {
                format!("(.unitColumn {})", self.int(index, vars)?)
            }
            ConstantMatrix::Gadget { base, small } => {
                format!("(.gadget {} {small})", self.int(base, vars)?)
            }
            ConstantMatrix::PowerOfBase { base, exponent } => {
                format!("(.powerOfBase {} {})", self.int(base, vars)?, self.int(exponent, vars)?)
            }
            ConstantMatrix::Rotation { exponent } => {
                format!("(.rotation {})", self.int(exponent, vars)?)
            }
            ConstantMatrix::AllCoefficients { value } => {
                format!("(.allCoefficients {})", self.int(value, vars)?)
            }
            ConstantMatrix::Polynomial { coefficients } => {
                let values = coefficients
                    .iter()
                    .map(|c| c.evaluate(self.bindings).ok().and_then(|v| BigUint::try_from(v).ok()))
                    .collect::<Option<Vec<_>>>();
                match values {
                    Some(values) if values.len() >= PACKED_POLYNOMIAL_MIN => {
                        let (width, table) = packed_table(&values);
                        format!("(.packed {width} {} {table})", values.len())
                    }
                    _ => format!(
                        "(.polynomial {})",
                        list(
                            coefficients
                                .iter()
                                .map(|c| self.int(c, vars))
                                .collect::<Result<Vec<_>, _>>()?
                        )
                    ),
                }
            }
        })
    }

    fn tag(&self, parts: &[HashTagComponent], vars: &Vars) -> Result<String, ExportError> {
        Ok(list(
            parts
                .iter()
                .map(|part| {
                    Ok(match part {
                        HashTagComponent::Bytes(value) => format!("(.bytes {})", bytes(value)),
                        HashTagComponent::Integer(value) => {
                            format!("(.integer {})", self.int(value, vars)?)
                        }
                        HashTagComponent::Decimal(value) => {
                            format!("(.decimal {})", self.int(value, vars)?)
                        }
                        HashTagComponent::U64Le(value) => {
                            format!("(.u64le {})", self.int(value, vars)?)
                        }
                        HashTagComponent::Operand(index) => format!("(.operand {index})"),
                        HashTagComponent::OperandU64Le(index) => {
                            format!("(.operandU64le {index})")
                        }
                    })
                })
                .collect::<Result<Vec<_>, ExportError>>()?,
        ))
    }

    /// The variables of a child scope: those of the parent and the child's bindings.
    fn child_vars(&self, vars: &Vars, bindings: &[(String, IntExpr)]) -> Result<Vars, ExportError> {
        let mut child = vars.clone();
        for (name, value) in bindings {
            child.insert(name.clone(), self.int(value, vars)?);
        }
        Ok(child)
    }

    /// The index of the instance of `scope` with `vars`, written on first use.
    fn instance(&mut self, scope: FrozenGraphScopeId, vars: Vars) -> Result<usize, ExportError> {
        if let Some(index) = self.instances.iter().position(|(s, v)| *s == scope && *v == vars) {
            return Ok(index);
        }
        let index = self.instances.len();
        self.instances.push((scope.clone(), vars.clone()));
        self.scopes.push(None);
        let term = self.scope(&scope, &vars)?;
        self.scopes[index] = Some(term);
        Ok(index)
    }

    fn op(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        id: NodeId,
        kind: &NodeKind,
        vars: &Vars,
    ) -> Result<String, ExportError> {
        let int = |w: &Self, e: &IntExpr| w.int(e, vars);
        Ok(match kind {
            NodeKind::Input { name, .. } => {
                // The root reads its inputs by name, in name order; a body reads them in the
                // order of the scope's inputs.
                let position = if *scope_id == FrozenGraphScopeId::Root {
                    root_inputs(scope).iter().position(|input| input == name)
                } else {
                    scope.inputs().iter().position(|wire| wire.node == id)
                };
                let position = position
                    .ok_or_else(|| encoding(format!("input {id:?} is not a scope input")))?;
                format!(".input {position}")
            }
            NodeKind::ConstantInt(value) => format!(".constInt ({value})"),
            NodeKind::EvaluateInt(value) => format!(".evaluateInt {}", int(self, value)?),
            NodeKind::Sha256BitColumns { seed, rows, columns, index_bits } => {
                format!(".sha256BitColumns {} {rows} {columns} {index_bits}", bytes(seed))
            }
            NodeKind::ConstantReal(value) => format!(".constReal {}", self.real(value, vars)?),
            NodeKind::ConstantBool(value) => format!(".constBool {value}"),
            NodeKind::ConstantMatrix { matrix_type, value } => format!(
                ".constMatrix {} {}",
                self.matrix_type(matrix_type, vars)?,
                self.constant_matrix(value, vars)?
            ),
            NodeKind::GadgetTrapdoor { matrix_type, base } => format!(
                ".gadgetTrapdoor {} {}",
                self.matrix_type(matrix_type, vars)?,
                int(self, base)?
            ),
            NodeKind::TrapdoorPublic => ".trapdoorPublic".into(),
            NodeKind::IntBinary(op) => format!(
                ".intBinary .{}",
                match op {
                    IntBinaryOp::Add => "add",
                    IntBinaryOp::Subtract => "sub",
                    IntBinaryOp::Multiply => "mul",
                    IntBinaryOp::Divide => "div",
                    IntBinaryOp::Remainder => "rem",
                }
            ),
            NodeKind::IntCompare(op) => format!(
                ".intCompare .{}",
                match op {
                    IntCompareOp::Equal => "eq",
                    IntCompareOp::Less => "lt",
                    IntCompareOp::LessEqual => "le",
                }
            ),
            NodeKind::BitExtract { bit } => format!(".bitExtract {}", int(self, bit)?),
            NodeKind::IntToReal => ".intToReal".into(),
            NodeKind::BoolToInt => ".boolToInt".into(),
            NodeKind::RealBinary(op) => format!(
                ".realBinary .{}",
                match op {
                    RealBinaryOp::Add => "add",
                    RealBinaryOp::Subtract => "sub",
                    RealBinaryOp::Multiply => "mul",
                    RealBinaryOp::Divide => "div",
                }
            ),
            NodeKind::RealSqrt => ".realSqrt".into(),
            NodeKind::MatrixBinary(op) => format!(
                ".matrixBinary .{}",
                match op {
                    MatrixBinaryOp::Add => "add",
                    MatrixBinaryOp::Subtract => "sub",
                    MatrixBinaryOp::Multiply => "mul",
                }
            ),
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => format!(
                ".matrixMulAccumulate {} {has_bias}",
                list(coefficients.iter().map(|c| int(self, c)).collect::<Result<Vec<_>, _>>()?)
            ),
            NodeKind::MatrixMulSmallRhs => ".matrixMulSmallRhs".into(),
            NodeKind::MatrixNegate => ".matrixNegate".into(),
            NodeKind::MatrixScale { scalar } => format!(".matrixScale {}", int(self, scalar)?),
            NodeKind::RingAutomorphism { index } => {
                format!(".ringAutomorphism {}", int(self, index)?)
            }
            NodeKind::MultiplyMonomial => ".multiplyMonomial".into(),
            NodeKind::ModulusSwitch { destination } => {
                format!(".modulusSwitch {}", self.ring(destination, vars)?)
            }
            NodeKind::ModulusReduce { destination } => {
                format!(".modulusReduce {}", self.ring(destination, vars)?)
            }
            NodeKind::CenteredRebase { destination } => {
                format!(".centeredRebase {}", self.ring(destination, vars)?)
            }
            NodeKind::CenteredRoundDivide { divisor } => {
                format!(".centeredRoundDivide {}", int(self, divisor)?)
            }
            NodeKind::RnsModUp { destination, digit_size, normalize } => {
                // The ordered basis of the argument fixes the grouping of limbs into digits.
                let node = scope.node(id).ok_or_else(|| encoding("missing node"))?;
                let wire = scope
                    .arguments(node)
                    .and_then(|arguments| arguments.first().copied())
                    .ok_or_else(|| encoding("rnsModUp has no argument"))?;
                let source = match &scope.nodes()[wire.node.0 as usize].output_types()
                    [wire.port.0 as usize]
                {
                    WireType::Matrix(matrix) |
                    WireType::SmallMatrix { matrix, .. } |
                    WireType::Preimage { matrix, .. } => self.ring(&matrix.ring, vars)?,
                    _ => return Err(encoding("rnsModUp argument is not a matrix")),
                };
                format!(
                    ".rnsModUp {source} {} {digit_size} {normalize}",
                    self.ring(destination, vars)?
                )
            }
            NodeKind::RnsModDown { destination, plaintext_modulus } => format!(
                ".rnsModDown {} {}",
                self.ring(destination, vars)?,
                int(self, plaintext_modulus)?
            ),
            NodeKind::BlockModSwitch { destination, plaintext_modulus } => format!(
                ".blockModSwitch {} {}",
                self.ring(destination, vars)?,
                int(self, plaintext_modulus)?
            ),
            NodeKind::Transpose => ".transpose".into(),
            NodeKind::Slice { rows, columns } => {
                let range = |w: &Self, range: &Option<crate::node::IndexRange>| {
                    range
                        .as_ref()
                        .map(|r| -> Result<String, ExportError> {
                            Ok(format!("({}, {})", w.int(&r.start, vars)?, w.int(&r.end, vars)?))
                        })
                        .transpose()
                };
                format!(".slice {} {}", option(range(self, rows)?), option(range(self, columns)?))
            }
            NodeKind::Tensor => ".tensor".into(),
            NodeKind::BlockDiagonal { count } => format!(".blockDiagonal {count}"),
            NodeKind::Concat { axis } => format!(
                ".concat .{}",
                match axis {
                    ConcatAxis::Rows => "rows",
                    ConcatAxis::Columns => "columns",
                    ConcatAxis::Diagonal => "diagonal",
                }
            ),
            NodeKind::UniformResidueSample { matrix_type } => {
                format!(".uniformResidueSample {}", self.matrix_type(matrix_type, vars)?)
            }
            NodeKind::UniformIntervalSample { matrix_type, range } => format!(
                ".uniformIntervalSample {} {} {}",
                self.matrix_type(matrix_type, vars)?,
                int(self, &range.minimum)?,
                int(self, &range.maximum)?
            ),
            NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound, compact } => {
                format!(
                    ".gaussianSample {} {} {} {compact}",
                    self.matrix_type(matrix_type, vars)?,
                    self.real(sigma, vars)?,
                    int(self, max_coefficient_bound)?
                )
            }
            NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix,
                tag_components,
                base,
                digit_count,
            } => format!(
                ".hashSample {} .{} {} {} {} {}",
                self.matrix_type(matrix_type, vars)?,
                match variant {
                    HashVariant::Plain => "plain",
                    HashVariant::Decomposed => "decomposed",
                    HashVariant::SmallDecomposed => "smallDecomposed",
                },
                bytes(tag_prefix),
                self.tag(tag_components, vars)?,
                option(base.as_ref().map(|b| int(self, b)).transpose()?),
                option(digit_count.as_ref().map(|d| int(self, d)).transpose()?)
            ),
            NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } => format!(
                ".hashIntFamily {} {} {} {}",
                int(self, count)?,
                int(self, modulus)?,
                bytes(tag_prefix),
                self.tag(tag_components, vars)?
            ),
            NodeKind::TrapdoorSample {
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                instantiation,
                preimage_sigma,
            } => format!(
                ".trapdoorSample {} {} {} {} {} {} {}",
                self.matrix_type(matrix_type, vars)?,
                self.real(sigma, vars)?,
                int(self, gadget_base)?,
                int(self, digit_count)?,
                int(self, preimage_max_coefficient_bound)?,
                statistical(instantiation),
                option(preimage_sigma.as_ref().map(|s| self.real(s, vars)).transpose()?)
            ),
            NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => format!(
                ".preimageSample {} {}",
                self.matrix_type(matrix_type, vars)?,
                int(self, max_coefficient_bound)?
            ),
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                format!(".gadgetDecompose {} {small} {}", int(self, base)?, int(self, digit_count)?)
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                format!(".extractCoefficient {}", int(self, position)?)
            }
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
                format!(".liftIntegerToConstantPolynomial {}", self.matrix_type(matrix_type, vars)?)
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool, family } => {
                format!(
                    ".thresholdDecode {} {} {output_bool} {family}",
                    int(self, plaintext_modulus)?,
                    int(self, length)?
                )
            }
            NodeKind::CrtRecompose { modulus, plaintext_moduli, reconstruction_coefficients } => {
                format!(
                    ".crtRecompose {} {} {}",
                    int(self, modulus)?,
                    list(
                        plaintext_moduli
                            .iter()
                            .map(|m| int(self, m))
                            .collect::<Result<Vec<_>, _>>()?
                    ),
                    list(
                        reconstruction_coefficients.iter().map(|c| int(self, c)).collect::<Result<
                            Vec<_>,
                            _,
                        >>(
                        )?
                    )
                )
            }
            NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => format!(
                ".packPolynomialCoefficients {} {}",
                self.matrix_type(matrix_type, vars)?,
                int(self, coefficient_bits)?
            ),
            NodeKind::PolynomialFromValues { matrix_type, evaluation } => format!(
                ".polynomialFromValues {} {evaluation}",
                self.matrix_type(matrix_type, vars)?
            ),
            NodeKind::PolynomialValues { evaluation } => format!(".polynomialValues {evaluation}"),
            NodeKind::IntMatrixVectorProduct { transpose } => {
                format!(".intMatrixVectorProduct {transpose}")
            }
            NodeKind::SubgraphCall(call) => {
                let child = self.child(scope_id, id)?;
                let vars = self.child_vars(vars, &call.bindings)?;
                format!(".call {}", self.instance(child, vars)?)
            }
            NodeKind::ParallelLoop(spec) => {
                let child = self.child(scope_id, id)?;
                let count = int(self, &spec.count)?;
                let modes = list(spec.input_modes.iter().map(|mode| match mode {
                    LoopInputMode::Broadcast => ".broadcast".into(),
                    LoopInputMode::Zip => ".zip".into(),
                    LoopInputMode::ZipOffset { offset } => format!("(.zipOffset {offset})"),
                    LoopInputMode::Gather { index_argument } => {
                        format!("(.gather {index_argument})")
                    }
                    LoopInputMode::ZipQuotient { divisor } => {
                        format!("(.zipQuotient {divisor})")
                    }
                }));
                let vars = self.child_vars(vars, &spec.bindings)?;
                format!(
                    ".parallelLoop {} {count} {} {modes}",
                    self.instance(child, vars)?,
                    spec.index_slot
                )
            }
            NodeKind::SequentialLoop(spec) => {
                let child = self.child(scope_id, id)?;
                let count = int(self, &spec.count)?;
                let vars = self.child_vars(vars, &spec.bindings)?;
                format!(
                    ".sequentialLoop {} {count} {} {}",
                    self.instance(child, vars)?,
                    spec.index_slot,
                    spec.carried_count
                )
            }
            NodeKind::FamilyPack { count } => format!(".familyPack {}", int(self, count)?),
            NodeKind::FamilyGetStatic { index } => {
                format!(".familyGetStatic {}", int(self, index)?)
            }
            NodeKind::FamilyGetDynamic => ".familyGetDynamic".into(),
            NodeKind::Select { count } => format!(".select {}", int(self, count)?),
        })
    }

    fn child(
        &self,
        scope: &FrozenGraphScopeId,
        id: NodeId,
    ) -> Result<FrozenGraphScopeId, ExportError> {
        self.graph
            .child_scope_id(scope, id)
            .ok_or_else(|| ExportError::MissingChildScope { scope: scope.clone(), node: id })
    }

    /// The term of one scope instance, its nodes in an order where arguments come first.
    fn scope(&mut self, id: &FrozenGraphScopeId, vars: &Vars) -> Result<String, ExportError> {
        let scope =
            self.graph.scope(id).ok_or_else(|| encoding(format!("missing scope {id:?}")))?;
        let count = scope.nodes().len();
        let arguments = (0..count)
            .map(|i| {
                scope.arguments(&scope.nodes()[i]).ok_or(ExportError::MissingArguments {
                    scope: id.clone(),
                    node: NodeId(i as u64),
                })
            })
            .collect::<Result<Vec<Vec<WireRef>>, _>>()?;
        // Depth-first topological order.
        let mut order = Vec::with_capacity(count);
        let mut state = vec![0u8; count];
        fn visit(
            node: usize,
            arguments: &[Vec<WireRef>],
            state: &mut [u8],
            order: &mut Vec<usize>,
        ) -> Result<(), ExportError> {
            match state[node] {
                2 => return Ok(()),
                1 => return Err(encoding("cyclic scope")),
                _ => {}
            }
            state[node] = 1;
            for wire in &arguments[node] {
                visit(wire.node.0 as usize, arguments, state, order)?;
            }
            state[node] = 2;
            order.push(node);
            Ok(())
        }
        for node in 0..count {
            visit(node, &arguments, &mut state, &mut order)?;
        }
        let mut position = vec![0usize; count];
        for (index, &node) in order.iter().enumerate() {
            position[node] = index;
        }
        let reference =
            |wire: &WireRef| format!("⟨{}, {}⟩", position[wire.node.0 as usize], wire.port.0);
        let mut nodes = Vec::with_capacity(count);
        for &node in &order {
            let handle = &scope.nodes()[node];
            let op = self.op(id, scope, NodeId(node as u64), handle.kind(), vars)?;
            let outputs = handle
                .output_types()
                .iter()
                .map(|ty| self.wire_type(ty, vars))
                .collect::<Result<Vec<_>, _>>()?;
            nodes.push(format!(
                "⟨{op}, {}, {}⟩",
                list(arguments[node].iter().map(reference)),
                list(outputs)
            ));
        }
        let outputs = if *id == FrozenGraphScopeId::Root {
            self.graph.outputs().values().map(|root| reference(&root.value)).collect::<Vec<_>>()
        } else {
            scope.outputs().iter().map(reference).collect()
        };
        Ok(format!("⟨[\n      {}],\n    {}⟩", nodes.join(",\n      "), list(outputs)))
    }
}

fn statistical(instantiation: &TrapdoorInstantiation) -> String {
    match instantiation {
        TrapdoorInstantiation::Computational => "none".into(),
        TrapdoorInstantiation::Statistical { m_bar } => format!("(some {m_bar})"),
    }
}

/// The names of the root's inputs, in name order.
fn root_inputs(root: &GraphScope) -> Vec<String> {
    let mut names = root
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

/// The `MxxSecurity.IR.Program` term of `validated`. The root's inputs are in name order.
pub fn program_term(validated: &ValidatedGraph) -> Result<String, ExportError> {
    let graph = &validated.source;
    let mut writer =
        Writer { graph, bindings: &validated.bindings, instances: Vec::new(), scopes: Vec::new() };
    writer.instance(FrozenGraphScopeId::Root, Vars::new())?;
    let inputs = root_inputs(graph.root_scope()).into_iter().map(|name| format!("{name:?}"));
    let outputs = graph.outputs().keys().map(|name| format!("{name:?}"));
    Ok(format!(
        "⟨[\n    {}],\n  {},\n  {}⟩",
        writer.scopes.into_iter().map(Option::unwrap).collect::<Vec<_>>().join(",\n    "),
        list(inputs),
        list(outputs)
    ))
}
