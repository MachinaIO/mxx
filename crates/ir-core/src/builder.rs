use crate::{
    Graph, NodeId, Port, WireRef, WireType,
    artifact::{ArtifactConfidentiality, ProductionId},
    expr::RealExpr,
    graph::CompileParameter,
    node::{
        ArtifactInput, ConcatAxis, ConstantMatrix, HashVariant, IndexRange, IntBinaryOp,
        IntCompareOp, LoopInputMode, MatrixBinaryOp, Node, NodeKind, ParallelLoop, SampleRange,
        SubgraphCall,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use std::collections::BTreeMap;
use thiserror::Error;

/// A matrix wire together with its statically known type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixWire {
    pub wire: WireRef,
    pub matrix_type: MatrixType,
}

/// A first-class indexed family of homogeneous matrix wires.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixFamilyWire {
    pub wire: WireRef,
    pub matrix_type: MatrixType,
    pub count: crate::IntExpr,
}

/// A first-class indexed family whose element type is inferred by validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueFamilyWire {
    pub wire: WireRef,
    pub count: crate::IntExpr,
}

/// A first-class indexed family of homogeneous trapdoor wires.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapdoorFamilyWire {
    pub wire: WireRef,
    pub matrix_type: MatrixType,
    pub count: crate::IntExpr,
    pub sigma: RealExpr,
    pub gadget_base: crate::IntExpr,
    pub digit_count: crate::IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapdoorWire {
    pub wire: WireRef,
    pub public: MatrixWire,
    pub sigma: RealExpr,
    pub gadget_base: crate::IntExpr,
    pub digit_count: crate::IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum OutputFamilyError {
    #[error("an output family must contain at least one member")]
    Empty,
    #[error("all output-family members must have the same matrix type")]
    TypeMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SubgraphBuildError {
    #[error("subgraph template {name} was registered with a different definition")]
    ConflictingTemplate { name: String },
    #[error("parallel-loop input modes do not match its arguments")]
    LoopInputModeMismatch,
    #[error("parallel-loop body output count does not match the declared output types")]
    LoopOutputCountMismatch,
}

/// Deterministic builder for executable Graph IR producers.
#[derive(Debug)]
pub struct GraphBuilder {
    graph: Graph,
    next_node: u64,
}

impl GraphBuilder {
    pub fn new(name: impl Into<String>, parameters: Vec<CompileParameter>) -> Self {
        Self {
            graph: Graph {
                name: name.into(),
                parameters,
                input_types: BTreeMap::new(),
                nodes: Vec::new(),
                outputs: BTreeMap::new(),
                subgraphs: BTreeMap::new(),
                real_constants: BTreeMap::new(),
            },
            next_node: 0,
        }
    }

    pub fn from_graph(graph: Graph) -> Self {
        let next_node =
            graph.nodes.iter().map(|node| node.id.0).max().map_or(0, |id| id.saturating_add(1));
        Self { graph, next_node }
    }

    pub fn input(&mut self, name: impl Into<String>, matrix_type: MatrixType) -> MatrixWire {
        let name = name.into();
        let wire_type = WireType::Matrix(matrix_type.clone());
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new());
        MatrixWire { wire, matrix_type }
    }

    pub fn preimage_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
    ) -> MatrixWire {
        let name = name.into();
        let wire_type = WireType::Preimage(matrix_type.clone());
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new());
        MatrixWire { wire, matrix_type }
    }

    pub fn family_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        count: crate::IntExpr,
    ) -> MatrixFamilyWire {
        let name = name.into();
        let wire_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix_type.clone())),
            count: count.clone(),
        };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new());
        MatrixFamilyWire { wire, matrix_type, count }
    }

    pub fn artifact_family_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: crate::IntExpr,
        confidentiality: ArtifactConfidentiality,
    ) -> MatrixFamilyWire {
        let name = name.into();
        let wire_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix_type.clone())),
            count: count.clone(),
        };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        );
        MatrixFamilyWire { wire, matrix_type, count }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn artifact_trapdoor_family_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        sigma: RealExpr,
        gadget_base: crate::IntExpr,
        digit_count: crate::IntExpr,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: crate::IntExpr,
        confidentiality: ArtifactConfidentiality,
    ) -> TrapdoorFamilyWire {
        let name = name.into();
        let element = WireType::Trapdoor {
            matrix: matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
        };
        let wire_type =
            WireType::IndexedFamily { element: Box::new(element), count: count.clone() };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        );
        TrapdoorFamilyWire { wire, matrix_type, count, sigma, gadget_base, digit_count }
    }

    pub fn artifact_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> MatrixWire {
        let name = name.into();
        let wire_type = WireType::Matrix(matrix_type.clone());
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        );
        MatrixWire { wire, matrix_type }
    }

    pub fn integer_input(&mut self, name: impl Into<String>) -> WireRef {
        let name = name.into();
        self.graph.input_types.insert(name.clone(), WireType::Int);
        self.push(NodeKind::Input { name, wire_type: WireType::Int, artifact: None }, Vec::new())
    }

    pub fn boolean_input(&mut self, name: impl Into<String>) -> WireRef {
        let name = name.into();
        self.graph.input_types.insert(name.clone(), WireType::Bool);
        self.push(NodeKind::Input { name, wire_type: WireType::Bool, artifact: None }, Vec::new())
    }

    pub fn bytes_input(&mut self, name: impl Into<String>, length: usize) -> WireRef {
        let name = name.into();
        let wire_type = WireType::Bytes { length: crate::IntExpr::constant(length) };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new())
    }

    pub fn typed_blob_input(
        &mut self,
        name: impl Into<String>,
        type_name: impl Into<String>,
        schema_hash: [u8; 32],
    ) -> WireRef {
        let name = name.into();
        let wire_type = WireType::TypedBlob { type_name: type_name.into(), schema_hash };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new())
    }

    pub fn artifact_bytes_input(
        &mut self,
        name: impl Into<String>,
        length: crate::IntExpr,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> WireRef {
        let name = name.into();
        let wire_type = WireType::Bytes { length };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        )
    }

    pub fn artifact_typed_blob_input(
        &mut self,
        name: impl Into<String>,
        type_name: impl Into<String>,
        schema_hash: [u8; 32],
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> WireRef {
        let name = name.into();
        let wire_type = WireType::TypedBlob { type_name: type_name.into(), schema_hash };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        )
    }

    pub fn artifact_trapdoor_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        sigma: RealExpr,
        gadget_base: crate::IntExpr,
        digit_count: crate::IntExpr,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> TrapdoorWire {
        let name = name.into();
        let wire_type = WireType::Trapdoor {
            matrix: matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
        };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    confidentiality,
                }),
            },
            Vec::new(),
        );
        let public = self.push(NodeKind::TrapdoorPublic, vec![wire]);
        TrapdoorWire {
            wire,
            public: MatrixWire { wire: public, matrix_type },
            sigma,
            gadget_base,
            digit_count,
        }
    }

    pub fn trapdoor_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        sigma: RealExpr,
        gadget_base: crate::IntExpr,
        digit_count: crate::IntExpr,
    ) -> TrapdoorWire {
        let name = name.into();
        let wire_type = WireType::Trapdoor {
            matrix: matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
        };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let wire = self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new());
        let public = self.push(NodeKind::TrapdoorPublic, vec![wire]);
        TrapdoorWire {
            wire,
            public: MatrixWire { wire: public, matrix_type },
            sigma,
            gadget_base,
            digit_count,
        }
    }

    pub fn bool_to_int(&mut self, value: WireRef) -> WireRef {
        self.push(NodeKind::BoolToInt, vec![value])
    }

    pub fn constant_int(&mut self, value: impl Into<BigInt>) -> WireRef {
        self.push(NodeKind::ConstantInt(value.into()), Vec::new())
    }

    pub fn evaluate_int(&mut self, value: crate::IntExpr) -> WireRef {
        self.push(NodeKind::EvaluateInt(value), Vec::new())
    }

    pub fn select_wire(&mut self, index: WireRef, branches: &[WireRef]) -> WireRef {
        assert!(!branches.is_empty(), "select requires at least one branch");
        let mut args = Vec::with_capacity(branches.len() + 1);
        args.push(index);
        args.extend_from_slice(branches);
        self.push(NodeKind::Select { count: crate::IntExpr::constant(branches.len()) }, args)
    }

    pub fn int_binary(&mut self, operation: IntBinaryOp, lhs: WireRef, rhs: WireRef) -> WireRef {
        self.push(NodeKind::IntBinary(operation), vec![lhs, rhs])
    }

    pub fn int_compare(&mut self, operation: IntCompareOp, lhs: WireRef, rhs: WireRef) -> WireRef {
        self.push(NodeKind::IntCompare(operation), vec![lhs, rhs])
    }

    pub fn bit_extract(&mut self, value: WireRef, bit: crate::IntExpr) -> WireRef {
        self.push(NodeKind::BitExtract { bit }, vec![value])
    }

    pub fn extract_coefficient(&mut self, value: &MatrixWire, position: crate::IntExpr) -> WireRef {
        self.push(NodeKind::ExtractCoefficient { position }, vec![value.wire])
    }

    pub fn constant_coefficient(
        &mut self,
        value: &MatrixWire,
        position: crate::IntExpr,
    ) -> MatrixWire {
        let wire = self.push(NodeKind::ConstantCoefficient { position }, vec![value.wire]);
        MatrixWire { wire, matrix_type: value.matrix_type.clone() }
    }

    pub fn matrix_binary(
        &mut self,
        operation: MatrixBinaryOp,
        lhs: &MatrixWire,
        rhs: &MatrixWire,
        output_type: MatrixType,
    ) -> MatrixWire {
        let wire = self.push(NodeKind::MatrixBinary(operation), vec![lhs.wire, rhs.wire]);
        MatrixWire { wire, matrix_type: output_type }
    }

    pub fn matrix_scale(&mut self, input: &MatrixWire, scalar: crate::IntExpr) -> MatrixWire {
        let wire = self.push(NodeKind::MatrixScale { scalar }, vec![input.wire]);
        MatrixWire { wire, matrix_type: input.matrix_type.clone() }
    }

    pub fn matrix_negate(&mut self, input: &MatrixWire) -> MatrixWire {
        let wire = self.push(NodeKind::MatrixNegate, vec![input.wire]);
        MatrixWire { wire, matrix_type: input.matrix_type.clone() }
    }

    pub fn slice(
        &mut self,
        input: &MatrixWire,
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
        output_type: MatrixType,
    ) -> MatrixWire {
        let wire = self.push(NodeKind::Slice { rows, columns }, vec![input.wire]);
        MatrixWire { wire, matrix_type: output_type }
    }

    pub fn constant_matrix(
        &mut self,
        matrix_type: MatrixType,
        value: ConstantMatrix,
    ) -> MatrixWire {
        let wire = self
            .push(NodeKind::ConstantMatrix { matrix_type: matrix_type.clone(), value }, Vec::new());
        MatrixWire { wire, matrix_type }
    }

    pub fn gaussian_sample(&mut self, matrix_type: MatrixType, sigma: RealExpr) -> MatrixWire {
        let wire = self
            .push(NodeKind::GaussianSample { matrix_type: matrix_type.clone(), sigma }, Vec::new());
        MatrixWire { wire, matrix_type }
    }

    pub fn uniform_sample(&mut self, matrix_type: MatrixType, range: SampleRange) -> MatrixWire {
        let wire = self
            .push(NodeKind::UniformSample { matrix_type: matrix_type.clone(), range }, Vec::new());
        MatrixWire { wire, matrix_type }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hash_sample(
        &mut self,
        key: WireRef,
        matrix_type: MatrixType,
        variant: HashVariant,
        tag_prefix: Vec<u8>,
        tag_expressions: Vec<crate::IntExpr>,
        base: Option<crate::IntExpr>,
        digit_count: Option<crate::IntExpr>,
    ) -> MatrixWire {
        self.hash_sample_with_encoded_tags(
            key,
            matrix_type,
            variant,
            tag_prefix,
            tag_expressions,
            Vec::new(),
            Vec::new(),
            base,
            digit_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hash_sample_with_encoded_tags(
        &mut self,
        key: WireRef,
        matrix_type: MatrixType,
        variant: HashVariant,
        tag_prefix: Vec<u8>,
        tag_expressions: Vec<crate::IntExpr>,
        tag_decimal_expressions: Vec<crate::IntExpr>,
        tag_u64_le_expressions: Vec<crate::IntExpr>,
        base: Option<crate::IntExpr>,
        digit_count: Option<crate::IntExpr>,
    ) -> MatrixWire {
        let wire = self.push(
            NodeKind::HashSample {
                matrix_type: matrix_type.clone(),
                variant,
                tag_prefix,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
            },
            vec![key],
        );
        MatrixWire { wire, matrix_type }
    }

    pub fn trapdoor_sample(
        &mut self,
        matrix_type: MatrixType,
        sigma: RealExpr,
        gadget_base: crate::IntExpr,
        digit_count: crate::IntExpr,
    ) -> TrapdoorWire {
        let public_wire = self.push(
            NodeKind::TrapdoorSample {
                matrix_type: matrix_type.clone(),
                sigma: sigma.clone(),
                gadget_base: gadget_base.clone(),
                digit_count: digit_count.clone(),
            },
            Vec::new(),
        );
        TrapdoorWire {
            wire: WireRef { node: public_wire.node, port: Port(1) },
            public: MatrixWire { wire: public_wire, matrix_type },
            sigma,
            gadget_base,
            digit_count,
        }
    }

    pub fn preimage_sample(
        &mut self,
        trapdoor: &TrapdoorWire,
        target: &MatrixWire,
        matrix_type: MatrixType,
    ) -> MatrixWire {
        let wire = self.push(
            NodeKind::PreimageSample { matrix_type: matrix_type.clone() },
            vec![trapdoor.wire, target.wire],
        );
        MatrixWire { wire, matrix_type }
    }

    pub fn concat(
        &mut self,
        axis: ConcatAxis,
        inputs: &[MatrixWire],
        output: MatrixType,
    ) -> MatrixWire {
        let wire =
            self.push(NodeKind::Concat { axis }, inputs.iter().map(|input| input.wire).collect());
        MatrixWire { wire, matrix_type: output }
    }

    pub fn tensor(&mut self, lhs: &MatrixWire, rhs: &MatrixWire, output: MatrixType) -> MatrixWire {
        let wire = self.push(NodeKind::Tensor, vec![lhs.wire, rhs.wire]);
        MatrixWire { wire, matrix_type: output }
    }

    pub fn select(&mut self, index: WireRef, branches: &[MatrixWire]) -> MatrixWire {
        let first = branches.first().expect("select requires at least one branch");
        debug_assert!(branches.iter().all(|branch| branch.matrix_type == first.matrix_type));
        let wires = branches.iter().map(|branch| branch.wire).collect::<Vec<_>>();
        let wire = self.select_wire(index, &wires);
        MatrixWire { wire, matrix_type: first.matrix_type.clone() }
    }

    pub fn family_get_static(
        &mut self,
        family: &MatrixFamilyWire,
        index: crate::IntExpr,
    ) -> MatrixWire {
        let wire = self.push(NodeKind::FamilyGetStatic { index }, vec![family.wire]);
        MatrixWire { wire, matrix_type: family.matrix_type.clone() }
    }

    pub fn family_get_dynamic(&mut self, family: &MatrixFamilyWire, index: WireRef) -> MatrixWire {
        let wire = self.push(NodeKind::FamilyGetDynamic, vec![family.wire, index]);
        MatrixWire { wire, matrix_type: family.matrix_type.clone() }
    }

    pub fn value_family_get_static(
        &mut self,
        family: &ValueFamilyWire,
        index: crate::IntExpr,
    ) -> WireRef {
        self.push(NodeKind::FamilyGetStatic { index }, vec![family.wire])
    }

    pub fn value_family_get_dynamic(
        &mut self,
        family: &ValueFamilyWire,
        index: WireRef,
    ) -> WireRef {
        self.push(NodeKind::FamilyGetDynamic, vec![family.wire, index])
    }

    pub fn trapdoor_family_get_static(
        &mut self,
        family: &TrapdoorFamilyWire,
        index: crate::IntExpr,
    ) -> TrapdoorWire {
        let wire = self.push(NodeKind::FamilyGetStatic { index }, vec![family.wire]);
        self.trapdoor_family_member(family, wire)
    }

    pub fn trapdoor_family_get_dynamic(
        &mut self,
        family: &TrapdoorFamilyWire,
        index: WireRef,
    ) -> TrapdoorWire {
        let wire = self.push(NodeKind::FamilyGetDynamic, vec![family.wire, index]);
        self.trapdoor_family_member(family, wire)
    }

    fn trapdoor_family_member(
        &mut self,
        family: &TrapdoorFamilyWire,
        wire: WireRef,
    ) -> TrapdoorWire {
        let public = self.push(NodeKind::TrapdoorPublic, vec![wire]);
        TrapdoorWire {
            wire,
            public: MatrixWire { wire: public, matrix_type: family.matrix_type.clone() },
            sigma: family.sigma.clone(),
            gadget_base: family.gadget_base.clone(),
            digit_count: family.digit_count.clone(),
        }
    }

    /// Builds the 1×1 polynomial matrix `sum_i coefficients[i] * X^i`.
    pub fn constant_polynomial(
        &mut self,
        matrix_type: MatrixType,
        coefficients: impl IntoIterator<Item = BigInt>,
    ) -> MatrixWire {
        let mut nonzero = coefficients
            .into_iter()
            .enumerate()
            .filter(|(_, coefficient)| coefficient != &BigInt::from(0));
        let Some((exponent, coefficient)) = nonzero.next() else {
            return self.constant_matrix(matrix_type, ConstantMatrix::Zero);
        };
        let monomial = self.constant_matrix(
            matrix_type.clone(),
            ConstantMatrix::Rotation { exponent: crate::IntExpr::constant(exponent) },
        );
        let mut sum = self.matrix_scale(&monomial, crate::IntExpr::constant(coefficient));
        for (exponent, coefficient) in nonzero {
            let monomial = self.constant_matrix(
                matrix_type.clone(),
                ConstantMatrix::Rotation { exponent: crate::IntExpr::constant(exponent) },
            );
            let term = self.matrix_scale(&monomial, crate::IntExpr::constant(coefficient));
            sum = self.matrix_binary(MatrixBinaryOp::Add, &sum, &term, matrix_type.clone());
        }
        sum
    }

    pub fn gadget_decompose(
        &mut self,
        input: &MatrixWire,
        base: crate::IntExpr,
        output_type: MatrixType,
    ) -> MatrixWire {
        self.gadget_decompose_with_layout(input, base, false, None, output_type)
    }

    pub fn gadget_decompose_with_layout(
        &mut self,
        input: &MatrixWire,
        base: crate::IntExpr,
        small: bool,
        digit_count: Option<crate::IntExpr>,
        output_type: MatrixType,
    ) -> MatrixWire {
        let wire =
            self.push(NodeKind::GadgetDecompose { base, small, digit_count }, vec![input.wire]);
        MatrixWire { wire, matrix_type: output_type }
    }

    pub fn output(
        &mut self,
        name: impl Into<String>,
        value: &MatrixWire,
        confidentiality: ArtifactConfidentiality,
    ) {
        self.output_wire(name, value.wire, confidentiality);
    }

    pub fn output_wire(
        &mut self,
        name: impl Into<String>,
        value: WireRef,
        confidentiality: ArtifactConfidentiality,
    ) {
        let name = name.into();
        let output = self.push(
            NodeKind::Output {
                name: name.clone(),
                artifact_confidentiality: Some(confidentiality),
            },
            vec![value],
        );
        self.graph.outputs.insert(name, output);
    }

    pub fn value_output_wire(&mut self, name: impl Into<String>, value: WireRef) {
        let name = name.into();
        let output = self.push(
            NodeKind::Output { name: name.clone(), artifact_confidentiality: None },
            vec![value],
        );
        self.graph.outputs.insert(name, output);
    }

    pub fn subgraph_call(
        &mut self,
        graph: Graph,
        args: Vec<WireRef>,
        output_types: &[MatrixType],
    ) -> Result<Vec<MatrixWire>, SubgraphBuildError> {
        let name = graph.name.clone();
        if let Some(existing) = self.graph.subgraphs.get(&name) {
            if existing.as_ref() != &graph {
                return Err(SubgraphBuildError::ConflictingTemplate { name });
            }
        } else {
            self.graph.subgraphs.insert(name.clone(), Box::new(graph));
        }
        let first = self
            .push(NodeKind::SubgraphCall(SubgraphCall { graph: name, bindings: Vec::new() }), args);
        Ok(output_types
            .iter()
            .enumerate()
            .map(|(port, matrix_type)| MatrixWire {
                wire: WireRef { node: first.node, port: Port(port as u32) },
                matrix_type: matrix_type.clone(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn parallel_loop(
        &mut self,
        graph: Graph,
        count: crate::IntExpr,
        index_variable: impl Into<String>,
        bindings: Vec<(String, crate::IntExpr)>,
        args: Vec<WireRef>,
        input_modes: Vec<LoopInputMode>,
        output_types: &[MatrixType],
    ) -> Result<Vec<MatrixFamilyWire>, SubgraphBuildError> {
        let first = self.push_parallel_loop(
            graph,
            count.clone(),
            0,
            index_variable,
            bindings,
            args,
            input_modes,
            output_types.len(),
        )?;
        Ok(output_types
            .iter()
            .enumerate()
            .map(|(port, matrix_type)| MatrixFamilyWire {
                wire: WireRef { node: first.node, port: Port(port as u32) },
                matrix_type: matrix_type.clone(),
                count: count.clone(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn parallel_value_loop(
        &mut self,
        graph: Graph,
        count: crate::IntExpr,
        index_variable: impl Into<String>,
        bindings: Vec<(String, crate::IntExpr)>,
        args: Vec<WireRef>,
        input_modes: Vec<LoopInputMode>,
        output_count: usize,
    ) -> Result<Vec<ValueFamilyWire>, SubgraphBuildError> {
        let first = self.push_parallel_loop(
            graph,
            count.clone(),
            0,
            index_variable,
            bindings,
            args,
            input_modes,
            output_count,
        )?;
        Ok((0..output_count)
            .map(|port| ValueFamilyWire {
                wire: WireRef { node: first.node, port: Port(port as u32) },
                count: count.clone(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn nonempty_parallel_loop(
        &mut self,
        graph: Graph,
        count: crate::IntExpr,
        index_variable: impl Into<String>,
        bindings: Vec<(String, crate::IntExpr)>,
        args: Vec<WireRef>,
        input_modes: Vec<LoopInputMode>,
        output_types: &[MatrixType],
    ) -> Result<Vec<MatrixFamilyWire>, SubgraphBuildError> {
        let first = self.push_parallel_loop(
            graph,
            count.clone(),
            1,
            index_variable,
            bindings,
            args,
            input_modes,
            output_types.len(),
        )?;
        Ok(output_types
            .iter()
            .enumerate()
            .map(|(port, matrix_type)| MatrixFamilyWire {
                wire: WireRef { node: first.node, port: Port(port as u32) },
                matrix_type: matrix_type.clone(),
                count: count.clone(),
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn parallel_trapdoor_loop(
        &mut self,
        graph: Graph,
        count: crate::IntExpr,
        index_variable: impl Into<String>,
        bindings: Vec<(String, crate::IntExpr)>,
        args: Vec<WireRef>,
        input_modes: Vec<LoopInputMode>,
        matrix_type: MatrixType,
        sigma: RealExpr,
        gadget_base: crate::IntExpr,
        digit_count: crate::IntExpr,
    ) -> Result<TrapdoorFamilyWire, SubgraphBuildError> {
        let wire = self.push_parallel_loop(
            graph,
            count.clone(),
            0,
            index_variable,
            bindings,
            args,
            input_modes,
            1,
        )?;
        Ok(TrapdoorFamilyWire { wire, matrix_type, count, sigma, gadget_base, digit_count })
    }

    #[allow(clippy::too_many_arguments)]
    fn push_parallel_loop(
        &mut self,
        graph: Graph,
        count: crate::IntExpr,
        minimum_count: usize,
        index_variable: impl Into<String>,
        bindings: Vec<(String, crate::IntExpr)>,
        args: Vec<WireRef>,
        input_modes: Vec<LoopInputMode>,
        output_count: usize,
    ) -> Result<WireRef, SubgraphBuildError> {
        if args.len() != input_modes.len() {
            return Err(SubgraphBuildError::LoopInputModeMismatch);
        }
        if graph.outputs.len() != output_count {
            return Err(SubgraphBuildError::LoopOutputCountMismatch);
        }
        let name = graph.name.clone();
        if let Some(existing) = self.graph.subgraphs.get(&name) {
            if existing.as_ref() != &graph {
                return Err(SubgraphBuildError::ConflictingTemplate { name });
            }
        } else {
            self.graph.subgraphs.insert(name.clone(), Box::new(graph));
        }
        Ok(self.push(
            NodeKind::ParallelLoop(ParallelLoop {
                graph: name,
                count,
                minimum_count,
                index_variable: index_variable.into(),
                bindings,
                input_modes,
            }),
            args,
        ))
    }

    pub fn remove_output(&mut self, name: &str) {
        self.graph.outputs.remove(name);
    }

    pub fn threshold_decode(
        &mut self,
        input: &MatrixWire,
        plaintext_modulus: crate::IntExpr,
        length: crate::IntExpr,
        output_bool: bool,
    ) -> WireRef {
        self.push(
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool },
            vec![input.wire],
        )
    }

    pub fn crt_recompose(
        &mut self,
        levels: &[MatrixWire],
        plaintext_moduli: Vec<crate::IntExpr>,
        reconstruction_coefficients: Vec<crate::IntExpr>,
    ) -> MatrixWire {
        assert!(!levels.is_empty(), "CRT recomposition requires at least one level");
        assert_eq!(levels.len(), plaintext_moduli.len());
        assert_eq!(levels.len(), reconstruction_coefficients.len());
        let matrix_type = levels[0].matrix_type.clone();
        let wire = self.push(
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients },
            levels.iter().map(|level| level.wire).collect(),
        );
        MatrixWire { wire, matrix_type }
    }

    /// Declares an indexed artifact family. The family members must have one
    /// identical matrix type; elaboration validates this again.
    pub fn output_family(
        &mut self,
        name: impl Into<String>,
        values: &[MatrixWire],
        confidentiality: ArtifactConfidentiality,
    ) -> Result<WireRef, OutputFamilyError> {
        let family = self.family_pack(values)?;
        let name = name.into();
        self.output_family_wire(name.clone(), &family, confidentiality);
        Ok(self.graph.outputs[&name])
    }

    pub fn family_pack(
        &mut self,
        values: &[MatrixWire],
    ) -> Result<MatrixFamilyWire, OutputFamilyError> {
        let first = values.first().ok_or(OutputFamilyError::Empty)?;
        if values.iter().any(|value| value.matrix_type != first.matrix_type) {
            return Err(OutputFamilyError::TypeMismatch);
        }
        let count = crate::IntExpr::constant(values.len());
        let wire = self.push(
            NodeKind::FamilyPack { count: count.clone() },
            values.iter().map(|value| value.wire).collect(),
        );
        Ok(MatrixFamilyWire { wire, matrix_type: first.matrix_type.clone(), count })
    }

    pub fn trapdoor_family_pack(
        &mut self,
        values: &[TrapdoorWire],
    ) -> Result<TrapdoorFamilyWire, OutputFamilyError> {
        let first = values.first().ok_or(OutputFamilyError::Empty)?;
        if values.iter().any(|value| {
            value.public.matrix_type != first.public.matrix_type ||
                value.sigma != first.sigma ||
                value.gadget_base != first.gadget_base ||
                value.digit_count != first.digit_count
        }) {
            return Err(OutputFamilyError::TypeMismatch);
        }
        let count = crate::IntExpr::constant(values.len());
        let wire = self.push(
            NodeKind::FamilyPack { count: count.clone() },
            values.iter().map(|value| value.wire).collect(),
        );
        Ok(TrapdoorFamilyWire {
            wire,
            matrix_type: first.public.matrix_type.clone(),
            count,
            sigma: first.sigma.clone(),
            gadget_base: first.gadget_base.clone(),
            digit_count: first.digit_count.clone(),
        })
    }

    pub fn output_family_wire(
        &mut self,
        name: impl Into<String>,
        family: &MatrixFamilyWire,
        confidentiality: ArtifactConfidentiality,
    ) {
        self.output_wire(name, family.wire, confidentiality);
    }

    pub fn push(&mut self, kind: NodeKind, args: Vec<WireRef>) -> WireRef {
        let id = NodeId(self.next_node);
        self.next_node += 1;
        self.graph.nodes.push(Node { id, kind, args });
        WireRef { node: id, port: Port(0) }
    }

    pub fn finish(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ParamEnv,
        artifact::{Manifest, ManifestArtifact, SpecHash},
        types::{ConcreteMatrixType, ConcreteWireType, WireId},
        validate, validate_with_manifests,
    };

    #[test]
    fn core_builder_emits_a_valid_compound_matrix_expression() {
        let matrix_type = MatrixType {
            modulus: crate::IntExpr::constant(17),
            ring_dimension: crate::IntExpr::constant(8),
            rows: crate::IntExpr::constant(1),
            columns: crate::IntExpr::constant(1),
        };
        let range = SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) };
        let mut builder = GraphBuilder::new("reduce-compound", Vec::new());
        let left = builder.uniform_sample(matrix_type.clone(), range.clone());
        let right = builder.uniform_sample(matrix_type.clone(), range);
        let sum = builder.matrix_binary(MatrixBinaryOp::Add, &left, &right, matrix_type.clone());
        builder.output("out", &sum, ArtifactConfidentiality::Public);

        validate(&builder.finish(), &ParamEnv::default()).expect("compound graph validates");
    }

    #[test]
    fn value_loop_preserves_scalar_element_types() {
        let mut body = GraphBuilder::new("integer-loop-body", Vec::new());
        let value = body.evaluate_int(crate::IntExpr::Var("index".to_owned()));
        body.value_output_wire("0_value", value);

        let mut builder = GraphBuilder::new("integer-loop", Vec::new());
        let family = builder
            .parallel_value_loop(
                body.finish(),
                crate::IntExpr::constant(4),
                "index",
                Vec::new(),
                Vec::new(),
                Vec::new(),
                1,
            )
            .expect("value loop")
            .remove(0);
        let selected = builder.value_family_get_static(&family, crate::IntExpr::constant(2));
        builder.value_output_wire("selected", selected);
        validate(&builder.finish(), &ParamEnv::default()).expect("integer family graph validates");
    }

    #[test]
    fn value_loop_rejects_a_family_valued_body_output() {
        let matrix_type = MatrixType {
            modulus: crate::IntExpr::constant(17),
            ring_dimension: crate::IntExpr::constant(8),
            rows: crate::IntExpr::constant(1),
            columns: crate::IntExpr::constant(1),
        };
        let mut body = GraphBuilder::new("nested-family-loop-body", Vec::new());
        let first = body.constant_matrix(matrix_type.clone(), ConstantMatrix::Zero);
        let second = body.constant_matrix(matrix_type, ConstantMatrix::Identity);
        let family = body.family_pack(&[first, second]).expect("inner family");
        body.value_output_wire("0_family", family.wire);

        let mut builder = GraphBuilder::new("nested-family-loop", Vec::new());
        let family = builder
            .parallel_value_loop(
                body.finish(),
                crate::IntExpr::constant(2),
                "index",
                Vec::new(),
                Vec::new(),
                Vec::new(),
                1,
            )
            .expect("loop construction")
            .remove(0);
        builder.value_output_wire("nested", family.wire);
        let error = validate(&builder.finish(), &ParamEnv::default()).expect_err("nested family");
        assert!(error.to_string().contains("nested indexed families are not supported"));
    }

    #[test]
    fn artifact_family_cardinality_does_not_expand_graph_or_wire_metadata() {
        let count = 1_000_000usize;
        let matrix_type = MatrixType {
            modulus: crate::IntExpr::constant(17),
            ring_dimension: crate::IntExpr::constant(8),
            rows: crate::IntExpr::constant(1),
            columns: crate::IntExpr::constant(1),
        };
        let concrete_type = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        };
        let production_id =
            ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [11; 32] };
        let manifest = Manifest {
            ir_version: crate::encoding::IR_VERSION,
            production_id: production_id.clone(),
            artifacts: BTreeMap::from([(
                "family".to_owned(),
                ManifestArtifact {
                    artifact_type: crate::artifact::ArtifactType::Matrix(concrete_type.clone()),
                    family_count: Some(count),
                    confidentiality: ArtifactConfidentiality::Public,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        let mut builder = GraphBuilder::new("constant-size-family", Vec::new());
        let family = builder.artifact_family_input(
            "family",
            matrix_type,
            production_id.clone(),
            "family",
            crate::IntExpr::constant(count),
            ArtifactConfidentiality::Public,
        );
        let index = builder.integer_input("index");
        let selected = builder.family_get_dynamic(&family, index);
        builder.output("out", &selected, ArtifactConfidentiality::Public);
        let graph = builder.finish();
        assert_eq!(graph.nodes.len(), 4);

        let validated = validate_with_manifests(
            &graph,
            &ParamEnv::default(),
            &BTreeMap::from([(production_id.clone(), manifest.clone())]),
        )
        .expect("first-class family validates");
        assert_eq!(validated.wires.len(), 4);
        assert!(matches!(
            validated.wires.get(&WireId {
                instantiation_path: Vec::new(),
                wire: WireRef { node: NodeId(0), port: Port(0) },
            }),
            Some(ConcreteWireType::IndexedFamily { element, count: actual_count })
                if **element == ConcreteWireType::Matrix(concrete_type) &&
                    *actual_count == count
        ));

        let mut wrong_version = manifest.clone();
        wrong_version.ir_version = crate::encoding::IR_VERSION.saturating_sub(1);
        let error = validate_with_manifests(
            &graph,
            &ParamEnv::default(),
            &BTreeMap::from([(production_id.clone(), wrong_version)]),
        )
        .expect_err("old manifest version must be rejected");
        assert!(error.to_string().contains("requires version"));

        let mut wrong_production = manifest;
        wrong_production.production_id =
            ProductionId { spec_hash: SpecHash([9; 32]), execution_nonce: [13; 32] };
        let error = validate_with_manifests(
            &graph,
            &ParamEnv::default(),
            &BTreeMap::from([(production_id, wrong_production)]),
        )
        .expect_err("manifest production mismatch must be rejected");
        assert!(error.to_string().contains("does not match the manifest production id"));
    }

    #[test]
    fn persisted_outputs_reject_unsupported_values_and_skip_ephemeral_values_in_manifests() {
        let mut invalid = GraphBuilder::new("invalid-persisted-output", Vec::new());
        let boolean = invalid.push(NodeKind::ConstantBool(true), Vec::new());
        invalid.output_wire("boolean", boolean, ArtifactConfidentiality::Public);
        let error = validate(&invalid.finish(), &ParamEnv::default())
            .expect_err("persisted booleans are not artifact-compatible");
        assert!(error.to_string().contains("unsupported artifact type"));

        let matrix_type = MatrixType {
            modulus: crate::IntExpr::constant(17),
            ring_dimension: crate::IntExpr::constant(8),
            rows: crate::IntExpr::constant(1),
            columns: crate::IntExpr::constant(1),
        };
        let mut mixed = GraphBuilder::new("mixed-output-kinds", Vec::new());
        let boolean = mixed.push(NodeKind::ConstantBool(true), Vec::new());
        mixed.value_output_wire("boolean", boolean);
        let matrix = mixed.constant_matrix(matrix_type, ConstantMatrix::Zero);
        mixed.output("matrix", &matrix, ArtifactConfidentiality::Private);
        let validated =
            validate(&mixed.finish(), &ParamEnv::default()).expect("mixed outputs validate");
        let production_id = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [5; 32] };
        let manifest = crate::artifact::export_validated_manifest(production_id, &validated)
            .expect("only persisted outputs are exported");
        assert_eq!(manifest.artifacts.len(), 1);
        assert_eq!(manifest.artifacts["matrix"].confidentiality, ArtifactConfidentiality::Private);

        let mut mismatched = GraphBuilder::new("mismatched-output-name", Vec::new());
        let matrix = mismatched.constant_matrix(
            MatrixType {
                modulus: crate::IntExpr::constant(17),
                ring_dimension: crate::IntExpr::constant(8),
                rows: crate::IntExpr::constant(1),
                columns: crate::IntExpr::constant(1),
            },
            ConstantMatrix::Zero,
        );
        mismatched.output("declared", &matrix, ArtifactConfidentiality::Public);
        let mut mismatched = mismatched.finish();
        let output = mismatched.outputs.remove("declared").expect("declared output");
        mismatched.outputs.insert("different".to_owned(), output);
        let error = validate(&mismatched, &ParamEnv::default())
            .expect_err("output key and persisted node name must agree");
        assert!(error.to_string().contains("does not match Output node name"));
    }
}
