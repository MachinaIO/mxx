//! Dual-use BGG+ attribute evaluation.
//!
//! This is the matrix-evaluation operation used by the BTVW17 dual-use
//! technique.  The evaluator deliberately returns the two products consumed
//! by protocols, rather than materializing the generally much larger
//! transformation matrices `H_C` and `H_{C,x}`:
//!
//! ```text
//! public evaluation:    A_att H_C
//! encoded evaluation:   c_att H_{C,x}
//! ```
//!
//! For `c_att = s(A_att - x tensor G) + e`, both evaluations satisfy
//!
//! ```text
//! (A_att - x tensor G) H_{C,x} = A_att H_C - C(x).
//! ```

use mxx_dsl::{Mat, Ring};
use mxx_gadgets::{
    Poly,
    circuit::{
        ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
        PolyCircuit, PolyGateKind, PublicLookupLowering, SlotOperationLowering, lower_circuit,
    },
};
use mxx_ir_core::node::{ConcatAxis, ConstantMatrix};
use num_bigint::BigUint;
use thiserror::Error;

/// One block of a dual-use attribute encoding.
///
/// `attribute` is public auxiliary data (for AKY24 it is a bit of the public
/// GSW ciphertext `X`), not the hidden plaintext encrypted by `X`.
#[derive(Clone)]
pub struct AttributeEncodingWire {
    pub vector: Mat,
    pub public_matrix: Mat,
    pub attribute: Mat,
}

#[derive(Clone)]
pub struct AttributeEncodingCompiler {
    pub ring: Ring,
    pub gadget_base: mxx_ir_core::IntExpr,
    pub digit_count: mxx_ir_core::IntExpr,
}

#[derive(Clone)]
pub struct AttributeMatrixEvaluation {
    pub vector: Mat,
    pub public_matrix: Mat,
    pub value: Mat,
}

#[derive(Debug, Error)]
pub enum AttributeEvaluationError {
    #[error("attribute evaluation does not support gate {gate}: {feature}")]
    Unsupported { gate: usize, feature: &'static str },
    #[error("attribute evaluation circuit structure is invalid: {0}")]
    Structure(String),
}

impl AttributeEncodingCompiler {
    /// `MEvalC` in product form: returns `A_att H_C`.
    pub fn evaluate_public<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: Mat,
        inputs: impl IntoIterator<Item = Mat>,
    ) -> Result<Vec<Mat>, AttributeEvaluationError> {
        let mut lowering = PublicAttributeLowering { compiler: self };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    /// `MEvalCX` in product form: returns `c_att H_{C,x}` together with the
    /// matching public product and the public value `C(x)`.
    pub fn evaluate_encoded<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: AttributeEncodingWire,
        inputs: impl IntoIterator<Item = AttributeEncodingWire>,
    ) -> Result<Vec<AttributeEncodingWire>, AttributeEvaluationError> {
        let mut lowering = EncodedAttributeLowering { compiler: self };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    /// Evaluates a circuit whose inputs are ordered as encoded attributes
    /// followed by dynamic public values.
    ///
    /// A public value `p` is represented by the exact zero-noise relation
    /// `(vector, public_matrix, attribute) = (0, Gp, p)`.  This lets a circuit
    /// key be bound to public matrices sampled by the surrounding DSL graph
    /// without pretending those runtime values are compile-time constants.
    pub fn evaluate_public_mixed<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: Mat,
        encoded_inputs: impl IntoIterator<Item = Mat>,
        public_inputs: impl IntoIterator<Item = Mat>,
    ) -> Result<Vec<Mat>, AttributeEvaluationError> {
        let rows = one.matrix_type().rows.clone();
        let gadget = self.ring.gadget(rows, self.gadget_base.clone(), self.digit_count.clone());
        let inputs = encoded_inputs
            .into_iter()
            .chain(public_inputs.into_iter().map(|value| gadget.clone() * value));
        self.evaluate_public(circuit, one, inputs)
    }

    /// Encoded counterpart of [`Self::evaluate_public_mixed`]. Circuit inputs
    /// use the same ordering: hidden encoded attributes first, then dynamic
    /// public values.
    pub fn evaluate_encoded_mixed<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: AttributeEncodingWire,
        encoded_inputs: impl IntoIterator<Item = AttributeEncodingWire>,
        public_inputs: impl IntoIterator<Item = Mat>,
    ) -> Result<Vec<AttributeEncodingWire>, AttributeEvaluationError> {
        let rows = one.public_matrix.matrix_type().rows.clone();
        let gadget =
            self.ring.gadget(rows.clone(), self.gadget_base.clone(), self.digit_count.clone());
        let zero_rows = one.vector.matrix_type().rows.clone();
        let public_inputs = public_inputs.into_iter().map(|attribute| AttributeEncodingWire {
            // A public attribute a is represented by the carrier-preserving
            // pair (0 * G, G * a), so it enters the same algebra as a sampled
            // encoding without losing gadget columns.
            vector: self.ring.zero((zero_rows.clone(), rows.clone())) * gadget.clone(),
            public_matrix: gadget.clone() * attribute.clone(),
            attribute,
        });
        self.evaluate_encoded(circuit, one, encoded_inputs.into_iter().chain(public_inputs))
    }

    /// Matrix-valued BTVW17 evaluation. Circuit outputs are flattened in
    /// row-major order and embedded with the corresponding unit column before
    /// being accumulated. This is the construction from BTVW17 Section 4.1.
    pub fn evaluate_public_matrix<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: Mat,
        inputs: impl IntoIterator<Item = Mat>,
        output_rows: usize,
    ) -> Result<Mat, AttributeEvaluationError> {
        let outputs = self.evaluate_public(circuit, one, inputs)?;
        let output_columns = matrix_output_columns(outputs.len(), output_rows)?;
        let mut columns = Vec::with_capacity(output_columns);
        for column in 0..output_columns {
            let terms = (0..output_rows)
                .map(|row| {
                    let target = self.unit_column(output_rows, row);
                    outputs[row * output_columns + column]
                        .clone()
                        .mul_small_rhs(self.decompose(target))
                })
                .collect::<Vec<_>>();
            columns.push(sum_matrices(terms));
        }
        Ok(Mat::concat(ConcatAxis::Columns, columns))
    }

    pub fn evaluate_public_matrix_mixed<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: Mat,
        encoded_inputs: impl IntoIterator<Item = Mat>,
        public_inputs: impl IntoIterator<Item = Mat>,
        output_rows: usize,
    ) -> Result<Mat, AttributeEvaluationError> {
        let outputs = self.evaluate_public_mixed(circuit, one, encoded_inputs, public_inputs)?;
        let output_columns = matrix_output_columns(outputs.len(), output_rows)?;
        let mut columns = Vec::with_capacity(output_columns);
        for column in 0..output_columns {
            let terms = (0..output_rows)
                .map(|row| {
                    let target = self.unit_column(output_rows, row);
                    outputs[row * output_columns + column]
                        .clone()
                        .mul_small_rhs(self.decompose(target))
                })
                .collect::<Vec<_>>();
            columns.push(sum_matrices(terms));
        }
        Ok(Mat::concat(ConcatAxis::Columns, columns))
    }

    pub fn evaluate_encoded_matrix<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: AttributeEncodingWire,
        inputs: impl IntoIterator<Item = AttributeEncodingWire>,
        output_rows: usize,
    ) -> Result<AttributeMatrixEvaluation, AttributeEvaluationError> {
        let outputs = self.evaluate_encoded(circuit, one, inputs)?;
        let output_columns = matrix_output_columns(outputs.len(), output_rows)?;
        let mut vectors = Vec::with_capacity(output_columns);
        let mut public_columns = Vec::with_capacity(output_columns);
        let mut value_columns = Vec::with_capacity(output_columns);
        for column in 0..output_columns {
            let mut vector_terms = Vec::with_capacity(output_rows);
            let mut public_terms = Vec::with_capacity(output_rows);
            let mut value_terms = Vec::with_capacity(output_rows);
            for row in 0..output_rows {
                let output = &outputs[row * output_columns + column];
                let target = self.unit_column(output_rows, row);
                let decomposed = self.decompose(target.clone());
                // This unit-column target is an arbitrary output projection:
                // its decomposition is consumed on the right, but the target
                // is not thereby a canonical G encoding.
                vector_terms.push(output.vector.clone().mul_small_rhs(decomposed.clone()));
                public_terms.push(output.public_matrix.clone().mul_small_rhs(decomposed));
                value_terms.push(target * output.attribute.clone());
            }
            vectors.push(sum_matrices(vector_terms));
            public_columns.push(sum_matrices(public_terms));
            value_columns.push(sum_matrices(value_terms));
        }
        Ok(AttributeMatrixEvaluation {
            vector: Mat::concat(ConcatAxis::Columns, vectors),
            public_matrix: Mat::concat(ConcatAxis::Columns, public_columns),
            value: Mat::concat(ConcatAxis::Columns, value_columns),
        })
    }

    pub fn evaluate_encoded_matrix_mixed<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: AttributeEncodingWire,
        encoded_inputs: impl IntoIterator<Item = AttributeEncodingWire>,
        public_inputs: impl IntoIterator<Item = Mat>,
        output_rows: usize,
    ) -> Result<AttributeMatrixEvaluation, AttributeEvaluationError> {
        let outputs = self.evaluate_encoded_mixed(circuit, one, encoded_inputs, public_inputs)?;
        let output_columns = matrix_output_columns(outputs.len(), output_rows)?;
        let mut vectors = Vec::with_capacity(output_columns);
        let mut public_columns = Vec::with_capacity(output_columns);
        let mut value_columns = Vec::with_capacity(output_columns);
        for column in 0..output_columns {
            let mut vector_terms = Vec::with_capacity(output_rows);
            let mut public_terms = Vec::with_capacity(output_rows);
            let mut value_terms = Vec::with_capacity(output_rows);
            for row in 0..output_rows {
                let output = &outputs[row * output_columns + column];
                let target = self.unit_column(output_rows, row);
                let decomposed = self.decompose(target.clone());
                vector_terms.push(output.vector.clone().mul_small_rhs(decomposed.clone()));
                public_terms.push(output.public_matrix.clone().mul_small_rhs(decomposed));
                value_terms.push(target * output.attribute.clone());
            }
            vectors.push(sum_matrices(vector_terms));
            public_columns.push(sum_matrices(public_terms));
            value_columns.push(sum_matrices(value_terms));
        }
        Ok(AttributeMatrixEvaluation {
            vector: Mat::concat(ConcatAxis::Columns, vectors),
            public_matrix: Mat::concat(ConcatAxis::Columns, public_columns),
            value: Mat::concat(ConcatAxis::Columns, value_columns),
        })
    }

    fn decompose(&self, matrix: Mat) -> mxx_dsl::Preimage {
        matrix.decompose(self.gadget_base.clone(), self.digit_count.clone())
    }

    fn scalar(&self, coefficients: impl IntoIterator<Item = mxx_ir_core::IntExpr>) -> Mat {
        self.ring.polynomial(coefficients)
    }

    fn large_scalar_decomposition(&self, public_matrix: &Mat, scalar: Mat) -> mxx_dsl::Preimage {
        let rows = public_matrix.matrix_type().rows.clone();
        let gadget = self.ring.gadget(rows, self.gadget_base.clone(), self.digit_count.clone());
        self.decompose(scalar * gadget)
    }

    fn unit_column(&self, rows: usize, row: usize) -> Mat {
        self.ring.constant(
            (rows, 1),
            ConstantMatrix::UnitColumn { index: mxx_ir_core::IntExpr::constant(row) },
        )
    }
}

fn matrix_output_columns(
    output_count: usize,
    output_rows: usize,
) -> Result<usize, AttributeEvaluationError> {
    if output_rows == 0 || output_count == 0 || !output_count.is_multiple_of(output_rows) {
        return Err(AttributeEvaluationError::Structure(
            "matrix-valued circuit outputs must be a nonempty row-major rectangle".to_owned(),
        ));
    }
    Ok(output_count / output_rows)
}

fn sum_matrices(mut values: Vec<Mat>) -> Mat {
    let first = values.remove(0);
    values.into_iter().fold(first, |sum, value| sum + value)
}

struct PublicAttributeLowering<'a> {
    compiler: &'a AttributeEncodingCompiler,
}

impl CircuitLoweringTypes for PublicAttributeLowering<'_> {
    type Wire = Mat;
    type Error = AttributeEvaluationError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for PublicAttributeLowering<'_> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Mat,
        rhs: &Mat,
        gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(lhs.clone() + rhs.clone()),
            PolyGateKind::Sub => Ok(lhs.clone() - rhs.clone()),
            PolyGateKind::Mul => {
                Ok(lhs.clone().mul_small_rhs(self.compiler.decompose(rhs.clone())))
            }
            _ => unsupported(gate, "non-arithmetic gate"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Mat,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        Ok(input.clone() *
            self.compiler.scalar(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant)))
    }

    fn large_scalar_mul(
        &mut self,
        input: &Mat,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        let scalar = self.compiler.scalar(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        Ok(input.clone().mul_small_rhs(self.compiler.large_scalar_decomposition(input, scalar)))
    }
}

impl<P: Poly> PublicLookupLowering<P> for PublicAttributeLowering<'_> {
    fn public_lookup(
        &mut self,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &Mat,
        gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        unsupported(gate, "public lookup")
    }
}

impl<P: Poly> SlotOperationLowering<P> for PublicAttributeLowering<'_> {
    fn slot_transfer(
        &mut self,
        _input: &Mat,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        unsupported(gate, "slot transfer")
    }

    fn slot_reduce(
        &mut self,
        _inputs: &[Mat],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        unsupported(gate, "slot reduction")
    }

    fn slot_anchor_reduce(
        &mut self,
        _input: &Mat,
        _num_blocks: u32,
        _lane_scalars: &[num_bigint::BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Mat, Self::Error> {
        unsupported(gate, "anchor reduction")
    }
}

struct EncodedAttributeLowering<'a> {
    compiler: &'a AttributeEncodingCompiler,
}

impl CircuitLoweringTypes for EncodedAttributeLowering<'_> {
    type Wire = AttributeEncodingWire;
    type Error = AttributeEvaluationError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for EncodedAttributeLowering<'_> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &AttributeEncodingWire,
        rhs: &AttributeEncodingWire,
        gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(AttributeEncodingWire {
                // Addition keeps the synchronized relation
                // (C_L + C_R, A_L + A_R, x_L + x_R).
                vector: lhs.vector.clone() + rhs.vector.clone(),
                public_matrix: lhs.public_matrix.clone() + rhs.public_matrix.clone(),
                attribute: lhs.attribute.clone() + rhs.attribute.clone(),
            }),
            PolyGateKind::Sub => Ok(AttributeEncodingWire {
                // Subtraction applies the same component-wise relation with
                // differences in the carrier, public matrix, and attribute.
                vector: lhs.vector.clone() - rhs.vector.clone(),
                public_matrix: lhs.public_matrix.clone() - rhs.public_matrix.clone(),
                attribute: lhs.attribute.clone() - rhs.attribute.clone(),
            }),
            PolyGateKind::Mul => {
                // BTVW17 dual-use multiplication.  Expanding the two terms
                // cancels `s * lhs.attribute * rhs.public_matrix`, leaving an
                // encoding of the product under the public product matrix.
                let decomposed_rhs = self.compiler.decompose(rhs.public_matrix.clone());
                Ok(AttributeEncodingWire {
                    // This is C_L K_R + x_L C_R with G K_R=A_R; the two
                    // expanded encodings cancel their cross term.
                    vector: lhs.vector.clone().mul_small_rhs(decomposed_rhs.clone()) +
                        lhs.attribute.clone() * rhs.vector.clone(),
                    public_matrix: lhs.public_matrix.clone().mul_small_rhs(decomposed_rhs),
                    attribute: lhs.attribute.clone() * rhs.attribute.clone(),
                })
            }
            _ => unsupported(gate, "non-arithmetic gate"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &AttributeEncodingWire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        let scalar =
            self.compiler.scalar(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        // A small scalar t acts directly as tC and tA, while metadata records
        // the ordinary attribute product t x.
        Ok(AttributeEncodingWire {
            vector: scalar.clone() * input.vector.clone(),
            public_matrix: input.public_matrix.clone() * scalar.clone(),
            attribute: input.attribute.clone() * scalar,
        })
    }

    fn large_scalar_mul(
        &mut self,
        input: &AttributeEncodingWire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        let scalar = self.compiler.scalar(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        let decomposed =
            self.compiler.large_scalar_decomposition(&input.public_matrix, scalar.clone());
        // A large scalar is carried by tG: K_t is decomposed from tG and is
        // applied on the right of both vector and public-matrix relations.
        Ok(AttributeEncodingWire {
            vector: input.vector.clone().mul_small_rhs(decomposed.clone()),
            public_matrix: input.public_matrix.clone().mul_small_rhs(decomposed),
            attribute: input.attribute.clone() * scalar,
        })
    }
}

impl<P: Poly> PublicLookupLowering<P> for EncodedAttributeLowering<'_> {
    fn public_lookup(
        &mut self,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &AttributeEncodingWire,
        gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        unsupported(gate, "public lookup")
    }
}

impl<P: Poly> SlotOperationLowering<P> for EncodedAttributeLowering<'_> {
    fn slot_transfer(
        &mut self,
        _input: &AttributeEncodingWire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        unsupported(gate, "slot transfer")
    }

    fn slot_reduce(
        &mut self,
        _inputs: &[AttributeEncodingWire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        unsupported(gate, "slot reduction")
    }

    fn slot_anchor_reduce(
        &mut self,
        _input: &AttributeEncodingWire,
        _num_blocks: u32,
        _lane_scalars: &[num_bigint::BigUint],
        gate: GateInstance<'_>,
    ) -> Result<AttributeEncodingWire, Self::Error> {
        unsupported(gate, "anchor reduction")
    }
}

fn unsupported<T>(
    gate: GateInstance<'_>,
    feature: &'static str,
) -> Result<T, AttributeEvaluationError> {
    Err(AttributeEvaluationError::Unsupported { gate: gate.local_gate().index(), feature })
}

fn map_lower_error(error: CircuitLowerError<AttributeEvaluationError>) -> AttributeEvaluationError {
    match error {
        CircuitLowerError::Operation { source, .. } => source,
        other => AttributeEvaluationError::Structure(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::DslContext;
    use mxx_ir_core::node::{MatrixBinaryOp, NodeKind};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::RuntimeValue;
    use std::collections::BTreeMap;

    #[test]
    fn public_zero_attribute_encoding_is_a_zero_gain_gadget_product() {
        let ring = Ring::new(257, 8);
        let compiler = AttributeEncodingCompiler {
            ring: ring.clone(),
            gadget_base: 4.into(),
            digit_count: 4.into(),
        };
        let one = AttributeEncodingWire {
            vector: ring.input("one-vector", (1, 8)),
            public_matrix: ring.input("one-public", (2, 8)),
            attribute: ring.identity(1),
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        circuit.output([input]);
        let output = compiler
            .evaluate_encoded_mixed(&circuit, one, [], [ring.identity(1)])
            .unwrap()
            .remove(0);
        let built = DslContext::new("attribute-public-zero-carrier")
            .output("vector", output.vector)
            .unwrap()
            .build()
            .unwrap();
        let nodes = built.graph.scopes().values().flat_map(|scope| scope.nodes());
        assert!(nodes.into_iter().any(|node| {
            matches!(node.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) &&
                node.arguments().get(1).is_some_and(|right| {
                    matches!(
                        right.node().kind(),
                        NodeKind::ConstantMatrix {
                            value: mxx_ir_core::node::ConstantMatrix::Gadget { .. },
                            ..
                        }
                    )
                })
        }));
    }

    #[test]
    fn mixed_public_input_satisfies_the_btvw17_key_equation_at_runtime() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let ring = Ring::new(
            num_bigint::BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = AttributeEncodingCompiler {
            ring: ring.clone(),
            gadget_base: num_bigint::BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count: digit_count.into(),
        };
        let public_columns = 2 * digit_count;
        let a_one = ring.input("a-one", (2, public_columns));
        let a_left = ring.input("a-left", (2, public_columns));
        let a_right = ring.input("a-right", (2, public_columns));
        let secret = ring.input("secret", (1, 2));
        let x_left = ring.input("x-left", (1, 1));
        let x_right = ring.input("x-right", (1, 1));
        let public_bit = ring.input("public-bit", (1, 1));
        let one = ring.identity(1);
        let gadget = ring.gadget(2, compiler.gadget_base.clone(), digit_count);
        let encoded = |public_matrix: Mat, attribute: Mat| AttributeEncodingWire {
            vector: secret.clone() * (public_matrix.clone() - gadget.clone() * attribute.clone()),
            public_matrix,
            attribute,
        };

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3);
        let hidden_product = circuit.mul_gate(inputs.at(0), inputs.at(1)).as_single_wire();
        let mixed_product = circuit.mul_gate(inputs.at(0), inputs.at(2)).as_single_wire();
        let output = circuit.add_gate(hidden_product, mixed_product).as_single_wire();
        circuit.output([output]);
        let public = compiler
            .evaluate_public_mixed(
                &circuit,
                a_one.clone(),
                [a_left.clone(), a_right.clone()],
                [public_bit.clone()],
            )
            .expect("public MEvalC product")
            .remove(0);
        let evaluated = compiler
            .evaluate_encoded_mixed(
                &circuit,
                encoded(a_one, one.clone()),
                [encoded(a_left, x_left), encoded(a_right, x_right)],
                [public_bit],
            )
            .expect("encoded MEvalCX product")
            .remove(0);
        let expected = secret * (public.clone() - gadget * evaluated.attribute.clone());
        let graph = DslContext::new("btvw17-dual-use-key-equation")
            .output("difference", evaluated.vector - expected)
            .unwrap()
            .output("public-difference", evaluated.public_matrix - public)
            .unwrap()
            .build()
            .unwrap();

        let one_value = DCRTPolyMatrix::identity(&parameters, 1, None);
        let public_value = |offset| {
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, public_columns, offset).get_row(0),
                    row(&parameters, public_columns, offset + 2).get_row(0),
                ],
            )
        };
        let secret_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![one_value.entry(0, 0), one_value.entry(0, 0)],
        );
        let result = execute_graph(
            graph,
            parameters.clone(),
            BTreeMap::from([
                ("a-one".to_owned(), RuntimeValue::matrix(public_value(0))),
                ("a-left".to_owned(), RuntimeValue::matrix(public_value(1))),
                ("a-right".to_owned(), RuntimeValue::matrix(public_value(3))),
                ("secret".to_owned(), RuntimeValue::matrix(secret_value)),
                ("x-left".to_owned(), RuntimeValue::matrix(one_value.clone())),
                ("x-right".to_owned(), RuntimeValue::matrix(one_value.clone())),
                ("public-bit".to_owned(), RuntimeValue::matrix(one_value)),
            ]),
        );
        let zero = DCRTPolyMatrix::zero(&parameters, 1, public_columns);
        assert_eq!(matrix_output(&result, "difference"), &zero);
        assert_eq!(
            matrix_output(&result, "public-difference"),
            &DCRTPolyMatrix::zero(&parameters, 2, public_columns)
        );
    }
}
