//! BGG+ handlers for parameterized dynamic Boolean circuit families.

use crate::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslContext, DslError, Family, Int, LoopIndex, Mat, Parallel, Sequential};
use mxx_gadgets::circuit::{
    BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanLayerGate, GateSlot,
    evaluate_boolean_matrix_family,
};
use thiserror::Error;

#[derive(Clone)]
pub struct BggPublicKeyFamily {
    pub matrices: Family<Mat>,
    pub reveal_plaintext: bool,
}

#[derive(Clone)]
pub struct BggEncodingFamily {
    pub vectors: Family<Mat>,
    pub public_keys: BggPublicKeyFamily,
    pub plaintexts: Family<Mat>,
}

#[derive(Debug, Error)]
pub enum DynamicBooleanBggError {
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error("dynamic Boolean BGG evaluation requires revealed plaintexts for every input")]
    PlaintextRequired,
    #[error("dynamic Boolean BGG input component families have different counts")]
    FamilyLayout,
}

impl BggPublicKeyFamily {
    pub fn pack(values: Vec<BggPublicKeyWire>) -> Result<Self, DynamicBooleanBggError> {
        let reveal_plaintext = values.iter().all(|value| value.reveal_plaintext);
        Ok(Self {
            matrices: Family::pack(values.into_iter().map(|value| value.matrix).collect())?,
            reveal_plaintext,
        })
    }
}

impl BggEncodingFamily {
    pub fn pack(values: Vec<BggEncodingWire>) -> Result<Self, DynamicBooleanBggError> {
        if values.iter().any(|value| !value.pubkey.reveal_plaintext || value.plaintext.is_none()) {
            return Err(DynamicBooleanBggError::PlaintextRequired);
        }
        let vectors = Family::pack(values.iter().map(|value| value.vector.clone()).collect())?;
        let public_keys =
            BggPublicKeyFamily::pack(values.iter().map(|value| value.pubkey.clone()).collect())?;
        let plaintexts = Family::pack(
            values.into_iter().map(|value| value.plaintext.expect("checked above")).collect(),
        )?;
        Ok(Self { vectors, public_keys, plaintexts })
    }

    fn validate(&self) -> Result<(), DynamicBooleanBggError> {
        if self.vectors.count() != self.public_keys.matrices.count() ||
            self.vectors.count() != self.plaintexts.count() ||
            !self.public_keys.reveal_plaintext
        {
            return Err(DynamicBooleanBggError::FamilyLayout);
        }
        Ok(())
    }

    fn gather(self, indices: Family<mxx_dsl::Int>) -> Result<Self, DynamicBooleanBggError> {
        self.validate()?;
        let vectors = self.vectors.parallel_gather(indices.clone())?;
        let public_keys = self.public_keys.matrices.parallel_gather(indices.clone())?;
        let plaintexts = self.plaintexts.parallel_gather(indices)?;
        Ok(Self {
            vectors,
            public_keys: BggPublicKeyFamily {
                matrices: public_keys,
                reveal_plaintext: self.public_keys.reveal_plaintext,
            },
            plaintexts,
        })
    }
}

pub fn evaluate_boolean_public_key_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggPublicKeyFamily,
    one: BggPublicKeyWire,
    compiler: BggPublicKeyCompiler,
) -> Result<BggPublicKeyFamily, DynamicBooleanBggError> {
    if !preceding.reveal_plaintext || !one.reveal_plaintext {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let matrices = evaluate_boolean_matrix_family(
        context,
        params,
        circuit,
        preceding.matrices,
        PublicKeyBooleanGate { compiler, one },
    )?;
    Ok(BggPublicKeyFamily { matrices, reveal_plaintext: true })
}

pub fn evaluate_boolean_encoding_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggEncodingFamily,
    one: BggEncodingWire,
    compiler: BggEncodingCompiler,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    preceding.validate()?;
    if !one.pubkey.reveal_plaintext || one.plaintext.is_none() {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let BooleanCircuitFamilyInputs {
        active_gate_counts,
        gate_kinds,
        left_sources,
        right_sources,
        output_sources: _,
    } = circuit;
    let invariants = (active_gate_counts, (gate_kinds, (left_sources, right_sources)));
    let initial = (preceding.vectors, preceding.public_keys.matrices, preceding.plaintexts);
    let (vectors, public_keys, plaintexts) = Sequential::range(params.depth.clone()).scan(
        initial,
        invariants,
        |layer,
         (vectors, public_keys, plaintexts),
         (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
            let preceding = BggEncodingFamily {
                vectors,
                public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
                plaintexts,
            };
            let active_count = active_gate_counts.get(layer.as_int());
            let (_, kinds, left_indices, right_indices) =
                layer_metadata(context, params, &layer, gate_kinds, left_sources, right_sources)?;
            let left = scan_result(preceding.clone().gather(left_indices))?;
            let right = scan_result(preceding.gather(right_indices))?;
            let one_family = scan_result(repeated_encoding(params, &one))?;
            let zero =
                scan_result(encoding_binary(&compiler, &one_family, &one_family, EncodingOp::Sub))?;
            let not = scan_result(encoding_binary(&compiler, &one_family, &left, EncodingOp::Sub))?;
            let product = scan_result(encoding_multiply(&compiler, &left, &right))?;
            let sum = scan_result(encoding_binary(&compiler, &left, &right, EncodingOp::Add))?;
            let two_product = scan_result(encoding_scalar(
                &compiler,
                &product,
                compiler.public_key.ring.polynomial([2.into()]),
            ))?;
            let xor = scan_result(encoding_binary(&compiler, &sum, &two_product, EncodingOp::Sub))?;
            let active = Parallel::range(params.max_layer_width.clone()).map_values(|slot| {
                slot.as_int().less_equal(active_count.clone().sub(Int::constant(1))).to_int()
            })?;

            let selected_vectors = kinds.clone().parallel_select_mats(vec![
                zero.vectors.clone(),
                one_family.vectors.clone(),
                left.vectors.clone(),
                not.vectors.clone(),
                product.vectors.clone(),
                xor.vectors.clone(),
            ])?;
            let selected_public_keys = kinds.clone().parallel_select_mats(vec![
                zero.public_keys.matrices.clone(),
                one_family.public_keys.matrices.clone(),
                left.public_keys.matrices.clone(),
                not.public_keys.matrices.clone(),
                product.public_keys.matrices.clone(),
                xor.public_keys.matrices.clone(),
            ])?;
            let selected_plaintexts = kinds.clone().parallel_select_mats(vec![
                zero.plaintexts.clone(),
                one_family.plaintexts.clone(),
                left.plaintexts.clone(),
                not.plaintexts.clone(),
                product.plaintexts.clone(),
                xor.plaintexts.clone(),
            ])?;
            let output_vectors = active
                .clone()
                .parallel_select_mats(vec![zero.vectors.clone(), selected_vectors.clone()])?;
            let output_public_keys = active.clone().parallel_select_mats(vec![
                zero.public_keys.matrices.clone(),
                selected_public_keys.clone(),
            ])?;
            let output_plaintexts = active
                .clone()
                .parallel_select_mats(vec![zero.plaintexts.clone(), selected_plaintexts.clone()])?;
            Ok((output_vectors, output_public_keys, output_plaintexts))
        },
    )?;
    Ok(BggEncodingFamily {
        vectors,
        public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
        plaintexts,
    })
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
        let right_decomposition = right_decomposition.as_mat();
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

fn layer_metadata(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    layer: &LoopIndex,
    gate_kinds: Family<Int>,
    left_sources: Family<Int>,
    right_sources: Family<Int>,
) -> Result<(Family<Int>, Family<Int>, Family<Int>, Family<Int>), DslError> {
    let flattened = Parallel::range(params.max_layer_width.clone()).map_values(|slot| {
        context.evaluate_int(mxx_ir_core::IntExpr::Add(
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(layer.expression()),
                Box::new(params.max_layer_width.clone()),
            )),
            Box::new(slot.expression()),
        ))
    })?;
    let kinds = gate_kinds.parallel_gather(flattened.clone())?;
    let left = left_sources.parallel_gather(flattened.clone())?;
    let right = right_sources.parallel_gather(flattened.clone())?;
    Ok((flattened, kinds, left, right))
}

fn repeated_encoding(
    params: &BooleanCircuitFamilyParams,
    one: &BggEncodingWire,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    let plaintext = one.plaintext.clone().ok_or(DynamicBooleanBggError::PlaintextRequired)?;
    let vectors =
        Parallel::range(params.max_layer_width.clone()).map_values(|_| one.vector.clone())?;
    let public_keys = Parallel::range(params.max_layer_width.clone())
        .map_values(|_| one.pubkey.matrix.clone())?;
    let plaintexts =
        Parallel::range(params.max_layer_width.clone()).map_values(|_| plaintext.clone())?;
    Ok(BggEncodingFamily {
        vectors,
        public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
        plaintexts,
    })
}

fn scan_result<T>(result: Result<T, DynamicBooleanBggError>) -> Result<T, DslError> {
    result.map_err(|error| match error {
        DynamicBooleanBggError::Dsl(error) => error,
        DynamicBooleanBggError::PlaintextRequired | DynamicBooleanBggError::FamilyLayout => {
            DslError::Schema
        }
    })
}

#[derive(Clone, Copy)]
enum KeyOp {
    Add,
    Sub,
}

fn key_binary(
    compiler: &BggPublicKeyCompiler,
    left: &BggPublicKeyFamily,
    right: &BggPublicKeyFamily,
    operation: KeyOp,
) -> Result<BggPublicKeyFamily, DslError> {
    let compiler = compiler.clone();
    let matrices = mxx_dsl::parallel_zip_bundle_result(
        (left.matrices.clone(), right.matrices.clone()),
        move |_, (left_matrix, right_matrix)| {
            let left = BggPublicKeyWire { matrix: left_matrix, reveal_plaintext: true };
            let right = BggPublicKeyWire { matrix: right_matrix, reveal_plaintext: true };
            let output = match operation {
                KeyOp::Add => compiler.add(&left, &right),
                KeyOp::Sub => compiler.sub(&left, &right),
            }
            .matrix;
            Ok(output)
        },
    )?;
    Ok(BggPublicKeyFamily { matrices, reveal_plaintext: true })
}

fn key_scalar(
    compiler: &BggPublicKeyCompiler,
    input: &BggPublicKeyFamily,
    scalar: Mat,
) -> Result<BggPublicKeyFamily, DslError> {
    let compiler = compiler.clone();
    let matrices = input.matrices.clone().parallel_map_values(move |_, matrix| {
        let output = compiler
            .small_scalar_mul(&BggPublicKeyWire { matrix, reveal_plaintext: true }, &scalar)
            .matrix;
        output
    })?;
    Ok(BggPublicKeyFamily { matrices, reveal_plaintext: true })
}

#[derive(Clone, Copy)]
enum EncodingOp {
    Add,
    Sub,
}

fn encoding_binary(
    compiler: &BggEncodingCompiler,
    left: &BggEncodingFamily,
    right: &BggEncodingFamily,
    operation: EncodingOp,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let plaintexts = mxx_dsl::parallel_zip_bundle_result(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        move |_, (left_value, right_value)| {
            let output = match operation {
                EncodingOp::Add => left_value + right_value,
                EncodingOp::Sub => left_value - right_value,
            };
            Ok(output)
        },
    )?;
    let (vectors, public_keys) = match operation {
        EncodingOp::Add => {
            let vectors = mxx_dsl::parallel_zip_bundle_result(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| Ok(left_value + right_value),
            )?;
            let public_keys = key_binary(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Add,
            )?;
            (vectors, public_keys)
        }
        EncodingOp::Sub => {
            let vectors = mxx_dsl::parallel_zip_bundle_result(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| Ok(left_value - right_value),
            )?;
            let public_keys = key_binary(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Sub,
            )?;
            (vectors, public_keys)
        }
    };
    Ok(BggEncodingFamily { vectors, public_keys, plaintexts })
}

fn encoding_multiply(
    compiler: &BggEncodingCompiler,
    left: &BggEncodingFamily,
    right: &BggEncodingFamily,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let base = compiler.public_key.base.clone();
    let digits = compiler.public_key.digit_count.clone();
    let decomposed_right = right
        .public_keys
        .matrices
        .clone()
        .parallel_map_values(move |_, key| key.decompose(base, digits).as_mat())?;
    let public_keys = mxx_dsl::parallel_zip_bundle_result(
        (left.public_keys.matrices.clone(), decomposed_right.clone()),
        |_, (key, decomposition)| Ok(key * decomposition),
    )?;
    let first = mxx_dsl::parallel_zip_bundle_result(
        (left.vectors.clone(), decomposed_right.clone()),
        |_, (vector, key)| Ok(vector * key),
    )?;
    let second = mxx_dsl::parallel_zip_bundle_result(
        (right.vectors.clone(), left.plaintexts.clone()),
        |_, (vector, plaintext)| Ok(vector * plaintext),
    )?;
    let vectors = mxx_dsl::parallel_zip_bundle_result(
        (first.clone(), second.clone()),
        |_, (left_value, right_value)| Ok(left_value + right_value),
    )?;
    let plaintexts = mxx_dsl::parallel_zip_bundle_result(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        |_, (left_value, right_value)| Ok(left_value * right_value),
    )?;
    Ok(BggEncodingFamily {
        vectors,
        public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
        plaintexts,
    })
}

fn encoding_scalar(
    compiler: &BggEncodingCompiler,
    input: &BggEncodingFamily,
    scalar: Mat,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    input.validate()?;
    let public_keys = key_scalar(&compiler.public_key, &input.public_keys, scalar.clone())?;
    let vectors = input.vectors.clone().parallel_map_values({
        let scalar = scalar.clone();
        move |_, value| value * scalar
    })?;
    let plaintexts =
        input.plaintexts.clone().parallel_map_values(move |_, value| value * scalar)?;
    Ok(BggEncodingFamily { vectors, public_keys, plaintexts })
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
        let public_inputs = BggPublicKeyFamily {
            matrices: ring.input_family(
                "public-key-inputs",
                public_params.max_layer_width.clone(),
                (1, 4),
            ),
            reveal_plaintext: true,
        };
        let public_output = evaluate_boolean_public_key_layers(
            &public_context,
            &public_params,
            public_circuit,
            public_inputs,
            one_key.clone(),
            public_key.clone(),
        )
        .unwrap();
        let public_graph = public_context
            .family_output("output", public_output.matrices)
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
            pubkey: one_key.clone(),
            plaintext: Some(ring.input("one-plaintext", (1, 1))),
        };
        let encoding_inputs = BggEncodingFamily {
            vectors: ring.input_family(
                "encoding-input-vectors",
                encoding_params.max_layer_width.clone(),
                (1, 4),
            ),
            public_keys: BggPublicKeyFamily {
                matrices: ring.input_family(
                    "encoding-input-public-keys",
                    encoding_params.max_layer_width.clone(),
                    (1, 4),
                ),
                reveal_plaintext: true,
            },
            plaintexts: ring.input_family(
                "encoding-input-plaintexts",
                encoding_params.max_layer_width.clone(),
                (1, 1),
            ),
        };
        let encoding_output = evaluate_boolean_encoding_layers(
            &encoding_context,
            &encoding_params,
            encoding_circuit,
            encoding_inputs,
            one_encoding,
            BggEncodingCompiler { public_key },
        )
        .unwrap();
        let encoding_graph = encoding_context
            .family_output("vector", encoding_output.vectors)
            .unwrap()
            .family_output("public-key", encoding_output.public_keys.matrices)
            .unwrap()
            .family_output("plaintext", encoding_output.plaintexts)
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
