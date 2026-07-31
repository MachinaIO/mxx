use mxx_ir_core::{
    Graph, NodeId, Port, WireRef, WireType,
    artifact::ProductionId,
    expr::RealExpr,
    graph::CompileParameter,
    node::{
        ArtifactInput, ConcatAxis, ConstantMatrix, HashVariant, IndexRange, MatrixBinaryOp, Node,
        NodeKind, SampleRange, SubgraphCall,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapdoorWire {
    pub wire: WireRef,
    pub public: MatrixWire,
    pub sigma: RealExpr,
    pub gadget_base: mxx_ir_core::IntExpr,
    pub digit_count: mxx_ir_core::IntExpr,
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
}

/// Deterministic builder used by the BGG+ compilers.
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

    pub fn artifact_family_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
        count: mxx_ir_core::IntExpr,
        concrete_count: usize,
    ) -> Vec<MatrixWire> {
        let name = name.into();
        let wire_type = WireType::Matrix(matrix_type.clone());
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        let first = self.push(
            NodeKind::Input {
                name,
                wire_type,
                artifact: Some(ArtifactInput {
                    production_id,
                    artifact_name: artifact_name.into(),
                    family_count: Some(count),
                }),
            },
            Vec::new(),
        );
        (0..concrete_count)
            .map(|port| MatrixWire {
                wire: WireRef { node: first.node, port: Port(port as u32) },
                matrix_type: matrix_type.clone(),
            })
            .collect()
    }

    pub fn artifact_input(
        &mut self,
        name: impl Into<String>,
        matrix_type: MatrixType,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
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
                    family_count: None,
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
        let wire_type = WireType::Bytes { length: mxx_ir_core::IntExpr::constant(length) };
        self.graph.input_types.insert(name.clone(), wire_type.clone());
        self.push(NodeKind::Input { name, wire_type, artifact: None }, Vec::new())
    }

    pub fn bool_to_int(&mut self, value: WireRef) -> WireRef {
        self.push(NodeKind::BoolToInt, vec![value])
    }

    pub fn bit_extract(&mut self, value: WireRef, bit: mxx_ir_core::IntExpr) -> WireRef {
        self.push(NodeKind::BitExtract { bit }, vec![value])
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

    pub fn matrix_scale(&mut self, input: &MatrixWire, scalar: mxx_ir_core::IntExpr) -> MatrixWire {
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
        tag_expressions: Vec<mxx_ir_core::IntExpr>,
        base: Option<mxx_ir_core::IntExpr>,
        digit_count: Option<mxx_ir_core::IntExpr>,
    ) -> MatrixWire {
        let wire = self.push(
            NodeKind::HashSample {
                matrix_type: matrix_type.clone(),
                variant,
                tag_prefix,
                tag_expressions,
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
        gadget_base: mxx_ir_core::IntExpr,
        digit_count: mxx_ir_core::IntExpr,
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

    pub fn select(&mut self, index: WireRef, branches: &[MatrixWire]) -> MatrixWire {
        let first = branches.first().expect("select requires at least one branch");
        debug_assert!(branches.iter().all(|branch| branch.matrix_type == first.matrix_type));
        let mut args = Vec::with_capacity(branches.len() + 1);
        args.push(index);
        args.extend(branches.iter().map(|branch| branch.wire));
        let wire = self
            .push(NodeKind::Select { count: mxx_ir_core::IntExpr::constant(branches.len()) }, args);
        MatrixWire { wire, matrix_type: first.matrix_type.clone() }
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
            ConstantMatrix::Rotation { exponent: mxx_ir_core::IntExpr::constant(exponent) },
        );
        let mut sum = self.matrix_scale(&monomial, mxx_ir_core::IntExpr::constant(coefficient));
        for (exponent, coefficient) in nonzero {
            let monomial = self.constant_matrix(
                matrix_type.clone(),
                ConstantMatrix::Rotation { exponent: mxx_ir_core::IntExpr::constant(exponent) },
            );
            let term = self.matrix_scale(&monomial, mxx_ir_core::IntExpr::constant(coefficient));
            sum = self.matrix_binary(MatrixBinaryOp::Add, &sum, &term, matrix_type.clone());
        }
        sum
    }

    pub fn gadget_decompose(
        &mut self,
        input: &MatrixWire,
        base: mxx_ir_core::IntExpr,
        output_type: MatrixType,
    ) -> MatrixWire {
        self.gadget_decompose_with_layout(input, base, false, None, output_type)
    }

    pub fn gadget_decompose_with_layout(
        &mut self,
        input: &MatrixWire,
        base: mxx_ir_core::IntExpr,
        small: bool,
        digit_count: Option<mxx_ir_core::IntExpr>,
        output_type: MatrixType,
    ) -> MatrixWire {
        let wire =
            self.push(NodeKind::GadgetDecompose { base, small, digit_count }, vec![input.wire]);
        MatrixWire { wire, matrix_type: output_type }
    }

    pub fn output(&mut self, name: impl Into<String>, value: &MatrixWire) {
        self.graph.outputs.insert(name.into(), value.wire);
    }

    pub fn output_wire(&mut self, name: impl Into<String>, value: WireRef) {
        self.graph.outputs.insert(name.into(), value);
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

    pub fn remove_output(&mut self, name: &str) {
        self.graph.outputs.remove(name);
    }

    pub fn threshold_decode(
        &mut self,
        input: &MatrixWire,
        plaintext_modulus: mxx_ir_core::IntExpr,
        length: mxx_ir_core::IntExpr,
        output_bool: bool,
    ) -> WireRef {
        self.push(
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool },
            vec![input.wire],
        )
    }

    /// Declares an indexed artifact family. The family members must have one
    /// identical matrix type; elaboration validates this again.
    pub fn output_family(
        &mut self,
        name: impl Into<String>,
        values: &[MatrixWire],
    ) -> Result<WireRef, OutputFamilyError> {
        let first = values.first().ok_or(OutputFamilyError::Empty)?;
        if values.iter().any(|value| value.matrix_type != first.matrix_type) {
            return Err(OutputFamilyError::TypeMismatch);
        }
        let name = name.into();
        let wire = self.push(
            NodeKind::Output { name: name.clone() },
            values.iter().map(|value| value.wire).collect(),
        );
        self.graph.outputs.insert(name, wire);
        Ok(wire)
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
    use mxx_ir_core::{ParamEnv, validate};

    #[test]
    fn core_builder_emits_a_valid_compound_matrix_expression() {
        let matrix_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let range = SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) };
        let mut builder = GraphBuilder::new("reduce-compound", Vec::new());
        let left = builder.uniform_sample(matrix_type.clone(), range.clone());
        let right = builder.uniform_sample(matrix_type.clone(), range);
        let sum = builder.matrix_binary(MatrixBinaryOp::Add, &left, &right, matrix_type.clone());
        builder.output("out", &sum);

        validate(&builder.finish(), &ParamEnv::default()).expect("compound graph validates");
    }
}
