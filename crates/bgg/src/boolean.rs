//! BGG+ handlers for parameterized dynamic Boolean circuit families.

use crate::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{
    DerivationAttachmentValue, DslError, Family, GraphValue, GraphValueSchema, Mat, MatType,
    Pending, iterate, parallel, select,
};
use mxx_gadgets::circuit::{
    BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanLayerGate,
    BooleanMatrixLayerGate, GateSlot, evaluate_boolean_matrix_family,
};
use mxx_ir_core::{ValueHandle, WireType};
use thiserror::Error;

pub type BggPublicKeyFamily = Family<BggPublicKeyWire>;
pub type BggEncodingFamily = Family<CircuitEncoding>;

/// One revealed Boolean-circuit encoding with its preprocessing public projection.
/// This circuit value is separate from the public-key-free online encoding carrier.
#[derive(Clone)]
pub struct CircuitEncoding {
    pub vector: Mat,
    pub public_key: Mat,
    pub plaintext: Mat,
}

#[derive(Clone, PartialEq)]
pub struct CircuitEncodingType {
    pub vector: MatType,
    pub public_key: MatType,
    pub plaintext: MatType,
}

impl GraphValue for CircuitEncoding {
    type Schema = CircuitEncodingType;

    fn flatten(&self) -> Vec<ValueHandle> {
        (self.vector.clone(), self.public_key.clone(), self.plaintext.clone()).flatten()
    }

    fn pending(&self) -> Pending {
        Pending::merge([self.vector.pending(), self.public_key.pending(), self.plaintext.pending()])
    }

    fn schema(&self) -> Self::Schema {
        CircuitEncodingType {
            vector: self.vector.schema(),
            public_key: self.public_key.schema(),
            plaintext: self.plaintext.schema(),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let (vector, public_key, plaintext) = <(Mat, Mat, Mat)>::from_values(
            &(schema.vector.clone(), schema.public_key.clone(), schema.plaintext.clone()),
            values,
            pending,
        )?;
        Ok(Self { vector, public_key, plaintext })
    }
}

impl GraphValueSchema for CircuitEncodingType {
    type Value = CircuitEncoding;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        CircuitEncoding {
            vector: self.vector.placeholders_from(next),
            public_key: self.public_key.placeholders_from(next),
            plaintext: self.plaintext.placeholders_from(next),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        (self.vector.clone(), self.public_key.clone(), self.plaintext.clone()).wire_types()
    }
}

#[derive(Debug, Error)]
pub enum DynamicBooleanBggError {
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error("dynamic Boolean BGG evaluation requires revealed plaintexts for every input")]
    PlaintextRequired,
    #[error("dynamic Boolean BGG input families have incompatible counts")]
    FamilyLayout,
}

fn attach_public_key_signal_group<T: GraphValue>(value: T) -> Result<T, DslError> {
    let wire = value.flatten().into_iter().next().ok_or(DslError::Schema)?;
    value.derivation_attachment(
        "mxx-bgg",
        "public-key-signal-grouping",
        vec![("value".to_owned(), wire)],
    )
}

/// Retains executable vector/public-key/plaintext family wires for checked analysis.
fn attach_operational_pairing(family: BggEncodingFamily) -> Result<BggEncodingFamily, DslError> {
    let values = family.flatten();
    let [vector, public_key, plaintext] = values.as_slice() else { return Err(DslError::Schema) };
    family
        .derivation_attachment(
            "mxx-bgg",
            "encoding-family-pairing",
            vec![
                ("vector".to_owned(), vector.clone()),
                ("public-key".to_owned(), public_key.clone()),
                ("plaintext".to_owned(), plaintext.clone()),
            ],
        )?
        .derivation_attachment(
            "mxx-bgg",
            "public-key-signal-grouping",
            vec![("value".to_owned(), public_key.clone())],
        )
}

pub fn evaluate_boolean_public_key_layers(
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggPublicKeyFamily,
    one: BggPublicKeyWire,
    compiler: BggPublicKeyCompiler,
) -> Result<BggPublicKeyFamily, DynamicBooleanBggError> {
    if !preceding.schema().element.reveal_plaintext || !one.reveal_plaintext {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let matrices = evaluate_boolean_matrix_family(
        params,
        circuit,
        preceding.field(|key| key.matrix)?,
        PublicKeyBooleanGate { compiler, one },
    )?;
    Ok(matrices.field(|matrix| BggPublicKeyWire { matrix, reveal_plaintext: true })?)
}

pub fn evaluate_boolean_encoding_layers(
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggEncodingFamily,
    one: BggEncodingWire,
    one_public_key: BggPublicKeyWire,
    public_key_compiler: BggPublicKeyCompiler,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    if preceding.count() != &params.max_layer_width {
        return Err(DynamicBooleanBggError::FamilyLayout);
    }
    let one = CircuitEncoding {
        vector: one.vector,
        public_key: one_public_key.matrix,
        plaintext: one.plaintext.ok_or(DynamicBooleanBggError::PlaintextRequired)?,
    };
    let preceding = attach_operational_pairing(preceding)?;
    Ok(iterate(params.depth.clone(), preceding, |layer, preceding| {
        let active_count = circuit.active_gate_counts.at(&layer);
        let output = parallel(params.max_layer_width.clone(), |slot| {
            let flat = &layer * &params.max_layer_width + &slot;
            let left = preceding.at(circuit.left_sources.at(&flat));
            let right = preceding.at(circuit.right_sources.at(&flat));
            let zero = encoding_binary(&public_key_compiler, &one, &one, EncodingOp::Sub);
            let not = encoding_binary(&public_key_compiler, &one, &left, EncodingOp::Sub);
            let decomposition = right.public_key.clone().decompose(
                public_key_compiler.base.clone(),
                public_key_compiler.digit_count.clone(),
            );
            let product = CircuitEncoding {
                vector: decomposition.clone().mul_small_rhs(left.vector.clone()) +
                    &right.vector * &left.plaintext,
                public_key: decomposition.mul_small_rhs(left.public_key.clone()),
                plaintext: &left.plaintext * &right.plaintext,
            };
            let sum = encoding_binary(&public_key_compiler, &left, &right, EncodingOp::Add);
            let two = public_key_compiler.ring.polynomial([2.into()]);
            let two_product = CircuitEncoding {
                vector: &product.vector * &two,
                public_key: public_key_compiler
                    .small_scalar_mul(
                        &BggPublicKeyWire {
                            matrix: product.public_key.clone(),
                            reveal_plaintext: true,
                        },
                        &two,
                    )
                    .matrix,
                plaintext: &product.plaintext * two,
            };
            let xor = encoding_binary(&public_key_compiler, &sum, &two_product, EncodingOp::Sub);
            let selected = select(
                circuit.gate_kinds.at(flat),
                vec![zero.clone(), one.clone(), left, not, product, xor],
            )?;
            let active = slot.less_equal(&active_count - 1).to_int();
            select(active, vec![zero, selected])
        })?;
        attach_operational_pairing(output)
    })?)
}

#[derive(Clone)]
struct PublicKeyBooleanGate {
    compiler: BggPublicKeyCompiler,
    one: BggPublicKeyWire,
}

impl BooleanLayerGate<Mat> for PublicKeyBooleanGate {
    fn candidates(&self, _slot: GateSlot, left: Mat, right: Mat) -> Result<[Mat; 6], DslError> {
        let left = BggPublicKeyWire { matrix: left, reveal_plaintext: true };
        let right = BggPublicKeyWire { matrix: right, reveal_plaintext: true };
        let zero = self.compiler.sub(&self.one, &self.one);
        let not = self.compiler.sub(&self.one, &left);
        let right_decomposition = right
            .matrix
            .clone()
            .decompose(self.compiler.base.clone(), self.compiler.digit_count.clone());
        let product = self.compiler.mul_with_decomposition(&left, &right, right_decomposition);
        let sum = self.compiler.add(&left, &right);
        let two_scalar = self.compiler.ring.polynomial([2.into()]);
        let two_product = self.compiler.small_scalar_mul(&product, &two_scalar);
        let xor = self.compiler.sub(&sum, &two_product);
        Ok([
            zero.matrix,
            self.one.matrix.clone(),
            left.matrix,
            not.matrix,
            product.matrix,
            xor.matrix,
        ])
    }
}

impl BooleanMatrixLayerGate for PublicKeyBooleanGate {
    fn retain_initial_family(&self, family: Family<Mat>) -> Result<Family<Mat>, DslError> {
        attach_public_key_signal_group(family)
    }
    fn retain_selected_value(&self, value: Mat) -> Result<Mat, DslError> {
        attach_public_key_signal_group(value)
    }
}

#[derive(Clone, Copy)]
enum EncodingOp {
    Add,
    Sub,
}

fn encoding_binary(
    compiler: &BggPublicKeyCompiler,
    left: &CircuitEncoding,
    right: &CircuitEncoding,
    operation: EncodingOp,
) -> CircuitEncoding {
    let left_key = BggPublicKeyWire { matrix: left.public_key.clone(), reveal_plaintext: true };
    let right_key = BggPublicKeyWire { matrix: right.public_key.clone(), reveal_plaintext: true };
    match operation {
        EncodingOp::Add => CircuitEncoding {
            vector: &left.vector + &right.vector,
            public_key: compiler.add(&left_key, &right_key).matrix,
            plaintext: &left.plaintext + &right.plaintext,
        },
        EncodingOp::Sub => CircuitEncoding {
            vector: &left.vector - &right.vector,
            public_key: compiler.sub(&left_key, &right_key).matrix,
            plaintext: &left.plaintext - &right.plaintext,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, node::NodeKind};

    #[test]
    fn public_key_and_encoding_candidates_have_uniform_selected_schemas() {
        let ring = Ring::new(257, 8);
        let public_key =
            BggPublicKeyCompiler { ring: ring.clone(), base: 2.into(), digit_count: 4.into() };

        let (public_context, public_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("dynamic-bgg-public-key"));
        let public_circuit =
            BooleanCircuitFamilyInputs::protocol_inputs(&public_context, &public_params);
        let one_key =
            BggPublicKeyWire { matrix: ring.input("one-key", (1, 4)), reveal_plaintext: true };
        let public_inputs = ring
            .input_family("public-key-inputs", public_params.max_layer_width.clone(), (1, 4))
            .field(|matrix| BggPublicKeyWire { matrix, reveal_plaintext: true })
            .unwrap();
        let public_output = evaluate_boolean_public_key_layers(
            &public_params,
            public_circuit,
            public_inputs,
            one_key.clone(),
            public_key.clone(),
        )
        .unwrap();
        let public_graph = public_context
            .family_output("output", public_output.field(|key| key.matrix).unwrap())
            .unwrap()
            .build()
            .unwrap();
        public_graph.validate(&bindings()).unwrap();

        let (encoding_context, encoding_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("dynamic-bgg-encoding"));
        let encoding_circuit =
            BooleanCircuitFamilyInputs::protocol_inputs(&encoding_context, &encoding_params);
        let one_encoding = BggEncodingWire {
            vector: ring.input("one-vector", (1, 4)),
            plaintext: Some(ring.input("one-plaintext", (1, 1))),
        };
        let vectors = ring.input_family(
            "encoding-input-vectors",
            encoding_params.max_layer_width.clone(),
            (1, 4),
        );
        let keys = ring.input_family(
            "encoding-input-public-keys",
            encoding_params.max_layer_width.clone(),
            (1, 4),
        );
        let plaintexts = ring.input_family(
            "encoding-input-plaintexts",
            encoding_params.max_layer_width.clone(),
            (1, 1),
        );
        let encoding_inputs = parallel(encoding_params.max_layer_width.clone(), |i| {
            Ok(CircuitEncoding {
                vector: vectors.at(&i),
                public_key: keys.at(&i),
                plaintext: plaintexts.at(i),
            })
        })
        .unwrap();
        let encoding_result = evaluate_boolean_encoding_layers(
            &encoding_params,
            encoding_circuit,
            encoding_inputs,
            one_encoding,
            one_key,
            public_key,
        )
        .unwrap();
        let encoding_output = encoding_result;
        let encoding_graph = encoding_context
            .family_output("vector", encoding_output.field(|value| value.vector).unwrap())
            .unwrap()
            .family_output("public-key", encoding_output.field(|value| value.public_key).unwrap())
            .unwrap()
            .family_output("plaintext", encoding_output.field(|value| value.plaintext).unwrap())
            .unwrap()
            .build()
            .unwrap();
        encoding_graph.validate(&bindings()).unwrap();
        let decomposition_count = encoding_graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .count();
        assert_eq!(
            decomposition_count, 1,
            "the encoding family reuses one deterministic right-key decomposition"
        );
        let small_rhs_count = encoding_graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs))
            .count();
        assert_eq!(small_rhs_count, 2);
    }

    fn bindings() -> ParamEnv {
        ParamEnv {
            integers: std::collections::BTreeMap::from([
                (BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(), 2.into()),
            ]),
            ..ParamEnv::default()
        }
    }
}
