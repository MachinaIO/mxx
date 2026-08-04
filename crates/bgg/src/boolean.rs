//! BGG+ handlers for parameterized dynamic Boolean circuit families.

use crate::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{
    BodyTraceRemapper, DslContext, DslError, Family, GatherConstructionTrace, Int,
    LoopConstructionTrace, LoopIndex, Mat, Parallel, RemapConstructionTrace,
    SelectConstructionTrace, Sequential,
};
use mxx_gadgets::circuit::{
    BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanLayerGate, GateSlot,
    LayerMetadataConstructionTrace, MatrixBooleanLayerConstructionTrace,
    evaluate_boolean_matrix_family,
};
use mxx_ir_core::ValueHandle;
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

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct PublicKeyGateConstructionTrace {
    pub one: ValueHandle,
    pub left: ValueHandle,
    pub right: ValueHandle,
    pub zero: ValueHandle,
    pub not: ValueHandle,
    pub right_decomposition: ValueHandle,
    pub right_decomposition_materialized: ValueHandle,
    pub product: ValueHandle,
    pub sum: ValueHandle,
    pub two_scalar: ValueHandle,
    pub two_product: ValueHandle,
    pub xor: ValueHandle,
}

impl RemapConstructionTrace for PublicKeyGateConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            one: self.one.remap_current_body(map)?,
            left: self.left.remap_current_body(map)?,
            right: self.right.remap_current_body(map)?,
            zero: self.zero.remap_current_body(map)?,
            not: self.not.remap_current_body(map)?,
            right_decomposition: self.right_decomposition.remap_current_body(map)?,
            right_decomposition_materialized: self
                .right_decomposition_materialized
                .remap_current_body(map)?,
            product: self.product.remap_current_body(map)?,
            sum: self.sum.remap_current_body(map)?,
            two_scalar: self.two_scalar.remap_current_body(map)?,
            two_product: self.two_product.remap_current_body(map)?,
            xor: self.xor.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct PublicKeyBooleanConstructionTrace {
    pub layers: MatrixBooleanLayerConstructionTrace<PublicKeyGateConstructionTrace>,
    pub one: ValueHandle,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixBinaryConstructionTrace {
    pub left: ValueHandle,
    pub right: ValueHandle,
    pub output: ValueHandle,
}

impl RemapConstructionTrace for MatrixBinaryConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            left: self.left.remap_current_body(map)?,
            right: self.right.remap_current_body(map)?,
            output: self.output.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DecompositionConstructionTrace {
    pub input: ValueHandle,
    pub decomposition: ValueHandle,
    pub materialized: ValueHandle,
}

impl RemapConstructionTrace for DecompositionConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            input: self.input.remap_current_body(map)?,
            decomposition: self.decomposition.remap_current_body(map)?,
            materialized: self.materialized.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingMultiplicationConstructionTrace {
    pub right_public_key_decompositions: LoopConstructionTrace<DecompositionConstructionTrace>,
    pub public_keys: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub left_vectors_times_right_decompositions:
        LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub right_vectors_times_left_plaintexts: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub vectors: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub plaintexts: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingComponentConstructionTrace {
    pub vectors: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub public_keys: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
    pub plaintexts: LoopConstructionTrace<MatrixBinaryConstructionTrace>,
}

impl RemapConstructionTrace for EncodingComponentConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            vectors: self.vectors.remap_current_body(map)?,
            public_keys: self.public_keys.remap_current_body(map)?,
            plaintexts: self.plaintexts.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingGatherConstructionTrace {
    pub vectors: LoopConstructionTrace<GatherConstructionTrace>,
    pub public_keys: LoopConstructionTrace<GatherConstructionTrace>,
    pub plaintexts: LoopConstructionTrace<GatherConstructionTrace>,
}

impl RemapConstructionTrace for EncodingGatherConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            vectors: self.vectors.remap_current_body(map)?,
            public_keys: self.public_keys.remap_current_body(map)?,
            plaintexts: self.plaintexts.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingSelectionConstructionTrace {
    pub vectors: LoopConstructionTrace<SelectConstructionTrace>,
    pub public_keys: LoopConstructionTrace<SelectConstructionTrace>,
    pub plaintexts: LoopConstructionTrace<SelectConstructionTrace>,
}

impl RemapConstructionTrace for EncodingSelectionConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            vectors: self.vectors.remap_current_body(map)?,
            public_keys: self.public_keys.remap_current_body(map)?,
            plaintexts: self.plaintexts.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RepeatedEncodingConstructionTrace {
    pub vectors: LoopConstructionTrace<ValueHandle>,
    pub public_keys: LoopConstructionTrace<ValueHandle>,
    pub plaintexts: LoopConstructionTrace<ValueHandle>,
}

impl RemapConstructionTrace for RepeatedEncodingConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            vectors: self.vectors.remap_current_body(map)?,
            public_keys: self.public_keys.remap_current_body(map)?,
            plaintexts: self.plaintexts.remap_current_body(map)?,
        })
    }
}

impl RemapConstructionTrace for EncodingMultiplicationConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            right_public_key_decompositions: self
                .right_public_key_decompositions
                .remap_current_body(map)?,
            public_keys: self.public_keys.remap_current_body(map)?,
            left_vectors_times_right_decompositions: self
                .left_vectors_times_right_decompositions
                .remap_current_body(map)?,
            right_vectors_times_left_plaintexts: self
                .right_vectors_times_left_plaintexts
                .remap_current_body(map)?,
            vectors: self.vectors.remap_current_body(map)?,
            plaintexts: self.plaintexts.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingBooleanLayerBodyConstructionTrace {
    pub body_vectors: ValueHandle,
    pub body_public_keys: ValueHandle,
    pub body_plaintexts: ValueHandle,
    pub body_active_gate_counts: ValueHandle,
    pub body_gate_kinds: ValueHandle,
    pub body_left_sources: ValueHandle,
    pub body_right_sources: ValueHandle,
    pub active_gate_count: GatherConstructionTrace,
    pub metadata: LayerMetadataConstructionTrace,
    pub left_gather: EncodingGatherConstructionTrace,
    pub left_vectors: ValueHandle,
    pub left_public_keys: ValueHandle,
    pub left_plaintexts: ValueHandle,
    pub right_gather: EncodingGatherConstructionTrace,
    pub right_vectors: ValueHandle,
    pub right_public_keys: ValueHandle,
    pub right_plaintexts: ValueHandle,
    pub one_repetition: RepeatedEncodingConstructionTrace,
    pub one_vectors: ValueHandle,
    pub one_public_keys: ValueHandle,
    pub one_plaintexts: ValueHandle,
    pub zero_vectors: ValueHandle,
    pub zero_public_keys: ValueHandle,
    pub zero_plaintexts: ValueHandle,
    pub zero_operations: EncodingComponentConstructionTrace,
    pub not_vectors: ValueHandle,
    pub not_public_keys: ValueHandle,
    pub not_plaintexts: ValueHandle,
    pub not_operations: EncodingComponentConstructionTrace,
    pub multiplication: EncodingMultiplicationConstructionTrace,
    pub sum_vectors: ValueHandle,
    pub sum_public_keys: ValueHandle,
    pub sum_plaintexts: ValueHandle,
    pub sum_operations: EncodingComponentConstructionTrace,
    pub two_product_vectors: ValueHandle,
    pub two_product_public_keys: ValueHandle,
    pub two_product_plaintexts: ValueHandle,
    pub two_product_operations: EncodingComponentConstructionTrace,
    pub xor_vectors: ValueHandle,
    pub xor_public_keys: ValueHandle,
    pub xor_plaintexts: ValueHandle,
    pub xor_operations: EncodingComponentConstructionTrace,
    pub selected_vectors: ValueHandle,
    pub selected_public_keys: ValueHandle,
    pub selected_plaintexts: ValueHandle,
    pub candidate_selection: EncodingSelectionConstructionTrace,
    pub active_mask: ValueHandle,
    pub active_mask_loop: LoopConstructionTrace<ValueHandle>,
    pub output_vectors: ValueHandle,
    pub output_public_keys: ValueHandle,
    pub output_plaintexts: ValueHandle,
    pub active_selection: EncodingSelectionConstructionTrace,
}

impl RemapConstructionTrace for EncodingBooleanLayerBodyConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            body_vectors: self.body_vectors.remap_current_body(map)?,
            body_public_keys: self.body_public_keys.remap_current_body(map)?,
            body_plaintexts: self.body_plaintexts.remap_current_body(map)?,
            body_active_gate_counts: self.body_active_gate_counts.remap_current_body(map)?,
            body_gate_kinds: self.body_gate_kinds.remap_current_body(map)?,
            body_left_sources: self.body_left_sources.remap_current_body(map)?,
            body_right_sources: self.body_right_sources.remap_current_body(map)?,
            active_gate_count: self.active_gate_count.remap_current_body(map)?,
            metadata: self.metadata.remap_current_body(map)?,
            left_gather: self.left_gather.remap_current_body(map)?,
            left_vectors: self.left_vectors.remap_current_body(map)?,
            left_public_keys: self.left_public_keys.remap_current_body(map)?,
            left_plaintexts: self.left_plaintexts.remap_current_body(map)?,
            right_gather: self.right_gather.remap_current_body(map)?,
            right_vectors: self.right_vectors.remap_current_body(map)?,
            right_public_keys: self.right_public_keys.remap_current_body(map)?,
            right_plaintexts: self.right_plaintexts.remap_current_body(map)?,
            one_repetition: self.one_repetition.remap_current_body(map)?,
            one_vectors: self.one_vectors.remap_current_body(map)?,
            one_public_keys: self.one_public_keys.remap_current_body(map)?,
            one_plaintexts: self.one_plaintexts.remap_current_body(map)?,
            zero_vectors: self.zero_vectors.remap_current_body(map)?,
            zero_public_keys: self.zero_public_keys.remap_current_body(map)?,
            zero_plaintexts: self.zero_plaintexts.remap_current_body(map)?,
            zero_operations: self.zero_operations.remap_current_body(map)?,
            not_vectors: self.not_vectors.remap_current_body(map)?,
            not_public_keys: self.not_public_keys.remap_current_body(map)?,
            not_plaintexts: self.not_plaintexts.remap_current_body(map)?,
            not_operations: self.not_operations.remap_current_body(map)?,
            multiplication: self.multiplication.remap_current_body(map)?,
            sum_vectors: self.sum_vectors.remap_current_body(map)?,
            sum_public_keys: self.sum_public_keys.remap_current_body(map)?,
            sum_plaintexts: self.sum_plaintexts.remap_current_body(map)?,
            sum_operations: self.sum_operations.remap_current_body(map)?,
            two_product_vectors: self.two_product_vectors.remap_current_body(map)?,
            two_product_public_keys: self.two_product_public_keys.remap_current_body(map)?,
            two_product_plaintexts: self.two_product_plaintexts.remap_current_body(map)?,
            two_product_operations: self.two_product_operations.remap_current_body(map)?,
            xor_vectors: self.xor_vectors.remap_current_body(map)?,
            xor_public_keys: self.xor_public_keys.remap_current_body(map)?,
            xor_plaintexts: self.xor_plaintexts.remap_current_body(map)?,
            xor_operations: self.xor_operations.remap_current_body(map)?,
            selected_vectors: self.selected_vectors.remap_current_body(map)?,
            selected_public_keys: self.selected_public_keys.remap_current_body(map)?,
            selected_plaintexts: self.selected_plaintexts.remap_current_body(map)?,
            candidate_selection: self.candidate_selection.remap_current_body(map)?,
            active_mask: self.active_mask.remap_current_body(map)?,
            active_mask_loop: self.active_mask_loop.remap_current_body(map)?,
            output_vectors: self.output_vectors.remap_current_body(map)?,
            output_public_keys: self.output_public_keys.remap_current_body(map)?,
            output_plaintexts: self.output_plaintexts.remap_current_body(map)?,
            active_selection: self.active_selection.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct EncodingBooleanConstructionTrace {
    pub initial_vectors: ValueHandle,
    pub initial_public_keys: ValueHandle,
    pub initial_plaintexts: ValueHandle,
    pub one_vector: ValueHandle,
    pub one_public_key: ValueHandle,
    pub one_plaintext: ValueHandle,
    pub active_gate_counts: ValueHandle,
    pub gate_kinds: ValueHandle,
    pub left_sources: ValueHandle,
    pub right_sources: ValueHandle,
    pub layers: LoopConstructionTrace<EncodingBooleanLayerBodyConstructionTrace>,
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

    fn gather_traced(
        self,
        indices: Family<mxx_dsl::Int>,
    ) -> Result<(Self, EncodingGatherConstructionTrace), DynamicBooleanBggError> {
        self.validate()?;
        let (vectors, vector_trace) = self.vectors.parallel_gather_traced(indices.clone())?;
        let (public_keys, public_key_trace) =
            self.public_keys.matrices.parallel_gather_traced(indices.clone())?;
        let (plaintexts, plaintext_trace) = self.plaintexts.parallel_gather_traced(indices)?;
        Ok((
            Self {
                vectors,
                public_keys: BggPublicKeyFamily {
                    matrices: public_keys,
                    reveal_plaintext: self.public_keys.reveal_plaintext,
                },
                plaintexts,
            },
            EncodingGatherConstructionTrace {
                vectors: vector_trace,
                public_keys: public_key_trace,
                plaintexts: plaintext_trace,
            },
        ))
    }
}

pub fn evaluate_boolean_public_key_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggPublicKeyFamily,
    one: BggPublicKeyWire,
    compiler: BggPublicKeyCompiler,
) -> Result<(BggPublicKeyFamily, PublicKeyBooleanConstructionTrace), DynamicBooleanBggError> {
    if !preceding.reveal_plaintext || !one.reveal_plaintext {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let one_handle = one.matrix.value_handle().clone();
    let (matrices, layers) = evaluate_boolean_matrix_family(
        context,
        params,
        circuit,
        preceding.matrices,
        PublicKeyBooleanGate { compiler, one },
    )?;
    Ok((
        BggPublicKeyFamily { matrices, reveal_plaintext: true },
        PublicKeyBooleanConstructionTrace { layers, one: one_handle },
    ))
}

pub fn evaluate_boolean_encoding_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggEncodingFamily,
    one: BggEncodingWire,
    compiler: BggEncodingCompiler,
) -> Result<(BggEncodingFamily, EncodingBooleanConstructionTrace), DynamicBooleanBggError> {
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
    let initial_vectors = preceding.vectors.value_handle().clone();
    let initial_public_keys = preceding.public_keys.matrices.value_handle().clone();
    let initial_plaintexts = preceding.plaintexts.value_handle().clone();
    let one_vector = one.vector.value_handle().clone();
    let one_public_key = one.pubkey.matrix.value_handle().clone();
    let one_plaintext = one.plaintext.as_ref().expect("checked above").value_handle().clone();
    let active_gate_counts_handle = active_gate_counts.value_handle().clone();
    let gate_kinds_handle = gate_kinds.value_handle().clone();
    let left_sources_handle = left_sources.value_handle().clone();
    let right_sources_handle = right_sources.value_handle().clone();
    let invariants = (active_gate_counts, (gate_kinds, (left_sources, right_sources)));
    let initial = (preceding.vectors, preceding.public_keys.matrices, preceding.plaintexts);
    let ((vectors, public_keys, plaintexts), layers) = Sequential::range(params.depth.clone())
        .scan_traced(
            initial,
            invariants,
            |layer,
             (vectors, public_keys, plaintexts),
             (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
                let preceding = BggEncodingFamily {
                    vectors,
                    public_keys: BggPublicKeyFamily {
                        matrices: public_keys,
                        reveal_plaintext: true,
                    },
                    plaintexts,
                };
                let body_vectors = preceding.vectors.value_handle().clone();
                let body_public_keys = preceding.public_keys.matrices.value_handle().clone();
                let body_plaintexts = preceding.plaintexts.value_handle().clone();
                let body_active_gate_counts = active_gate_counts.value_handle().clone();
                let body_gate_kinds = gate_kinds.value_handle().clone();
                let body_left_sources = left_sources.value_handle().clone();
                let body_right_sources = right_sources.value_handle().clone();
                let active_count_index = layer.as_int();
                let active_count_index_handle = active_count_index.value_handle().clone();
                let active_count_source = active_gate_counts.value_handle().clone();
                let active_count = active_gate_counts.get(active_count_index);
                let active_gate_count = GatherConstructionTrace {
                    index: active_count_index_handle,
                    sources: vec![active_count_source],
                    outputs: vec![active_count.value_handle().clone()],
                };
                let (_, kinds, left_indices, right_indices, metadata) = layer_metadata(
                    context,
                    params,
                    &layer,
                    gate_kinds,
                    left_sources,
                    right_sources,
                )?;
                let (left, left_gather) =
                    scan_result(preceding.clone().gather_traced(left_indices))?;
                let (right, right_gather) = scan_result(preceding.gather_traced(right_indices))?;
                let (one_family, one_repetition) = scan_result(repeated_encoding(params, &one))?;
                let (zero, zero_operations) = scan_result(encoding_binary(
                    &compiler,
                    &one_family,
                    &one_family,
                    EncodingOp::Sub,
                ))?;
                let (not, not_operations) =
                    scan_result(encoding_binary(&compiler, &one_family, &left, EncodingOp::Sub))?;
                let (product, multiplication) =
                    scan_result(encoding_multiply(&compiler, &left, &right))?;
                let (sum, sum_operations) =
                    scan_result(encoding_binary(&compiler, &left, &right, EncodingOp::Add))?;
                let (two_product, two_product_operations) = scan_result(encoding_scalar(
                    &compiler,
                    &product,
                    compiler.public_key.ring.polynomial([2.into()]),
                ))?;
                let (xor, xor_operations) =
                    scan_result(encoding_binary(&compiler, &sum, &two_product, EncodingOp::Sub))?;
                let (active, active_mask_loop) = Parallel::range(params.max_layer_width.clone())
                    .map_values_traced(|slot| {
                        let output = slot
                            .as_int()
                            .less_equal(active_count.clone().sub(Int::constant(1)))
                            .to_int();
                        (output.clone(), output.value_handle().clone())
                    })?;

                let (selected_vectors, selected_vectors_trace) =
                    kinds.clone().parallel_select_mats_traced(vec![
                        zero.vectors.clone(),
                        one_family.vectors.clone(),
                        left.vectors.clone(),
                        not.vectors.clone(),
                        product.vectors.clone(),
                        xor.vectors.clone(),
                    ])?;
                let (selected_public_keys, selected_public_keys_trace) =
                    kinds.clone().parallel_select_mats_traced(vec![
                        zero.public_keys.matrices.clone(),
                        one_family.public_keys.matrices.clone(),
                        left.public_keys.matrices.clone(),
                        not.public_keys.matrices.clone(),
                        product.public_keys.matrices.clone(),
                        xor.public_keys.matrices.clone(),
                    ])?;
                let (selected_plaintexts, selected_plaintexts_trace) =
                    kinds.clone().parallel_select_mats_traced(vec![
                        zero.plaintexts.clone(),
                        one_family.plaintexts.clone(),
                        left.plaintexts.clone(),
                        not.plaintexts.clone(),
                        product.plaintexts.clone(),
                        xor.plaintexts.clone(),
                    ])?;
                let candidate_selection = EncodingSelectionConstructionTrace {
                    vectors: selected_vectors_trace,
                    public_keys: selected_public_keys_trace,
                    plaintexts: selected_plaintexts_trace,
                };
                let (output_vectors, output_vectors_trace) =
                    active.clone().parallel_select_mats_traced(vec![
                        zero.vectors.clone(),
                        selected_vectors.clone(),
                    ])?;
                let (output_public_keys, output_public_keys_trace) =
                    active.clone().parallel_select_mats_traced(vec![
                        zero.public_keys.matrices.clone(),
                        selected_public_keys.clone(),
                    ])?;
                let (output_plaintexts, output_plaintexts_trace) =
                    active.clone().parallel_select_mats_traced(vec![
                        zero.plaintexts.clone(),
                        selected_plaintexts.clone(),
                    ])?;
                let active_selection = EncodingSelectionConstructionTrace {
                    vectors: output_vectors_trace,
                    public_keys: output_public_keys_trace,
                    plaintexts: output_plaintexts_trace,
                };
                let body_trace = EncodingBooleanLayerBodyConstructionTrace {
                    body_vectors,
                    body_public_keys,
                    body_plaintexts,
                    body_active_gate_counts,
                    body_gate_kinds,
                    body_left_sources,
                    body_right_sources,
                    active_gate_count,
                    metadata,
                    left_gather,
                    left_vectors: left.vectors.value_handle().clone(),
                    left_public_keys: left.public_keys.matrices.value_handle().clone(),
                    left_plaintexts: left.plaintexts.value_handle().clone(),
                    right_gather,
                    right_vectors: right.vectors.value_handle().clone(),
                    right_public_keys: right.public_keys.matrices.value_handle().clone(),
                    right_plaintexts: right.plaintexts.value_handle().clone(),
                    one_repetition,
                    one_vectors: one_family.vectors.value_handle().clone(),
                    one_public_keys: one_family.public_keys.matrices.value_handle().clone(),
                    one_plaintexts: one_family.plaintexts.value_handle().clone(),
                    zero_vectors: zero.vectors.value_handle().clone(),
                    zero_public_keys: zero.public_keys.matrices.value_handle().clone(),
                    zero_plaintexts: zero.plaintexts.value_handle().clone(),
                    zero_operations,
                    not_vectors: not.vectors.value_handle().clone(),
                    not_public_keys: not.public_keys.matrices.value_handle().clone(),
                    not_plaintexts: not.plaintexts.value_handle().clone(),
                    not_operations,
                    multiplication,
                    sum_vectors: sum.vectors.value_handle().clone(),
                    sum_public_keys: sum.public_keys.matrices.value_handle().clone(),
                    sum_plaintexts: sum.plaintexts.value_handle().clone(),
                    sum_operations,
                    two_product_vectors: two_product.vectors.value_handle().clone(),
                    two_product_public_keys: two_product
                        .public_keys
                        .matrices
                        .value_handle()
                        .clone(),
                    two_product_plaintexts: two_product.plaintexts.value_handle().clone(),
                    two_product_operations,
                    xor_vectors: xor.vectors.value_handle().clone(),
                    xor_public_keys: xor.public_keys.matrices.value_handle().clone(),
                    xor_plaintexts: xor.plaintexts.value_handle().clone(),
                    xor_operations,
                    selected_vectors: selected_vectors.value_handle().clone(),
                    selected_public_keys: selected_public_keys.value_handle().clone(),
                    selected_plaintexts: selected_plaintexts.value_handle().clone(),
                    candidate_selection,
                    active_mask: active.value_handle().clone(),
                    active_mask_loop,
                    output_vectors: output_vectors.value_handle().clone(),
                    output_public_keys: output_public_keys.value_handle().clone(),
                    output_plaintexts: output_plaintexts.value_handle().clone(),
                    active_selection,
                };
                Ok(((output_vectors, output_public_keys, output_plaintexts), body_trace))
            },
        )?;
    let trace = EncodingBooleanConstructionTrace {
        initial_vectors,
        initial_public_keys,
        initial_plaintexts,
        one_vector,
        one_public_key,
        one_plaintext,
        active_gate_counts: active_gate_counts_handle,
        gate_kinds: gate_kinds_handle,
        left_sources: left_sources_handle,
        right_sources: right_sources_handle,
        layers,
    };
    Ok((
        BggEncodingFamily {
            vectors,
            public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
            plaintexts,
        },
        trace,
    ))
}

#[derive(Clone)]
struct PublicKeyBooleanGate {
    compiler: BggPublicKeyCompiler,
    one: BggPublicKeyWire,
}

impl BooleanLayerGate<Mat> for PublicKeyBooleanGate {
    type ConstructionTrace = PublicKeyGateConstructionTrace;

    fn candidates(
        &self,
        _slot: GateSlot,
        left: Mat,
        right: Mat,
    ) -> Result<([Mat; 6], Self::ConstructionTrace), DslError> {
        let left = BggPublicKeyWire { matrix: left, reveal_plaintext: true };
        let right = BggPublicKeyWire { matrix: right, reveal_plaintext: true };
        let zero = self.compiler.sub(&self.one, &self.one);
        let not = self.compiler.sub(&self.one, &left);
        let right_decomposition = right
            .matrix
            .clone()
            .decompose(self.compiler.base.clone(), self.compiler.digit_count.clone());
        let right_decomposition_handle = right_decomposition.value_handle().clone();
        let right_decomposition = right_decomposition.as_mat();
        let right_decomposition_materialized = right_decomposition.value_handle().clone();
        let product = self.compiler.mul_with_decomposition(&left, &right, right_decomposition);
        let sum = self.compiler.add(&left, &right);
        let two_scalar = self.compiler.ring.polynomial([2.into()]);
        let two_product = self.compiler.small_scalar_mul(&product, &two_scalar);
        let xor = self.compiler.sub(&sum, &two_product);
        let trace = PublicKeyGateConstructionTrace {
            one: self.one.matrix.value_handle().clone(),
            left: left.matrix.value_handle().clone(),
            right: right.matrix.value_handle().clone(),
            zero: zero.matrix.value_handle().clone(),
            not: not.matrix.value_handle().clone(),
            right_decomposition: right_decomposition_handle,
            right_decomposition_materialized,
            product: product.matrix.value_handle().clone(),
            sum: sum.matrix.value_handle().clone(),
            two_scalar: two_scalar.value_handle().clone(),
            two_product: two_product.matrix.value_handle().clone(),
            xor: xor.matrix.value_handle().clone(),
        };
        Ok((
            [
                zero.matrix,
                self.one.matrix.clone(),
                left.matrix,
                not.matrix,
                product.matrix,
                xor.matrix,
            ],
            trace,
        ))
    }
}

fn layer_metadata(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    layer: &LoopIndex,
    gate_kinds: Family<Int>,
    left_sources: Family<Int>,
    right_sources: Family<Int>,
) -> Result<
    (Family<Int>, Family<Int>, Family<Int>, Family<Int>, LayerMetadataConstructionTrace),
    DslError,
> {
    let (flattened, flattened_trace) = Parallel::range(params.max_layer_width.clone())
        .map_values_traced(|slot| {
            context.evaluate_int_traced(mxx_ir_core::IntExpr::Add(
                Box::new(mxx_ir_core::IntExpr::Mul(
                    Box::new(layer.expression()),
                    Box::new(params.max_layer_width.clone()),
                )),
                Box::new(slot.expression()),
            ))
        })?;
    let (kinds, kinds_trace) = gate_kinds.parallel_gather_traced(flattened.clone())?;
    let (left, left_trace) = left_sources.parallel_gather_traced(flattened.clone())?;
    let (right, right_trace) = right_sources.parallel_gather_traced(flattened.clone())?;
    Ok((
        flattened,
        kinds,
        left,
        right,
        LayerMetadataConstructionTrace {
            flattened_indices: flattened_trace,
            gate_kinds: kinds_trace,
            left_sources: left_trace,
            right_sources: right_trace,
        },
    ))
}

fn repeated_encoding(
    params: &BooleanCircuitFamilyParams,
    one: &BggEncodingWire,
) -> Result<(BggEncodingFamily, RepeatedEncodingConstructionTrace), DynamicBooleanBggError> {
    let plaintext = one.plaintext.clone().ok_or(DynamicBooleanBggError::PlaintextRequired)?;
    let (vectors, vector_trace) = Parallel::range(params.max_layer_width.clone())
        .map_values_traced(|_| {
            let output = one.vector.clone();
            (output.clone(), output.value_handle().clone())
        })?;
    let (public_keys, public_key_trace) = Parallel::range(params.max_layer_width.clone())
        .map_values_traced(|_| {
            let output = one.pubkey.matrix.clone();
            (output.clone(), output.value_handle().clone())
        })?;
    let (plaintexts, plaintext_trace) = Parallel::range(params.max_layer_width.clone())
        .map_values_traced(|_| {
            let output = plaintext.clone();
            (output.clone(), output.value_handle().clone())
        })?;
    Ok((
        BggEncodingFamily {
            vectors,
            public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
            plaintexts,
        },
        RepeatedEncodingConstructionTrace {
            vectors: vector_trace,
            public_keys: public_key_trace,
            plaintexts: plaintext_trace,
        },
    ))
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
) -> Result<(BggPublicKeyFamily, LoopConstructionTrace<MatrixBinaryConstructionTrace>), DslError> {
    let compiler = compiler.clone();
    let (matrices, trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (left.matrices.clone(), right.matrices.clone()),
        move |_, (left_matrix, right_matrix)| {
            let left_handle = left_matrix.value_handle().clone();
            let right_handle = right_matrix.value_handle().clone();
            let left = BggPublicKeyWire { matrix: left_matrix, reveal_plaintext: true };
            let right = BggPublicKeyWire { matrix: right_matrix, reveal_plaintext: true };
            let output = match operation {
                KeyOp::Add => compiler.add(&left, &right),
                KeyOp::Sub => compiler.sub(&left, &right),
            }
            .matrix;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left: left_handle,
                    right: right_handle,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    Ok((BggPublicKeyFamily { matrices, reveal_plaintext: true }, trace))
}

fn key_scalar(
    compiler: &BggPublicKeyCompiler,
    input: &BggPublicKeyFamily,
    scalar: Mat,
) -> Result<(BggPublicKeyFamily, LoopConstructionTrace<MatrixBinaryConstructionTrace>), DslError> {
    let compiler = compiler.clone();
    let scalar_handle = scalar.value_handle().clone();
    let (matrices, trace) =
        input.matrices.clone().parallel_map_values_traced(move |_, matrix| {
            let left = matrix.value_handle().clone();
            let output = compiler
                .small_scalar_mul(&BggPublicKeyWire { matrix, reveal_plaintext: true }, &scalar)
                .matrix;
            (
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right: scalar_handle.clone(),
                    output: output.value_handle().clone(),
                },
            )
        })?;
    Ok((BggPublicKeyFamily { matrices, reveal_plaintext: true }, trace))
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
) -> Result<(BggEncodingFamily, EncodingComponentConstructionTrace), DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let (plaintexts, plaintext_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        move |_, (left_value, right_value)| {
            let left = left_value.value_handle().clone();
            let right = right_value.value_handle().clone();
            let output = match operation {
                EncodingOp::Add => left_value + right_value,
                EncodingOp::Sub => left_value - right_value,
            };
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let (vectors, public_keys, vector_trace, public_key_trace) = match operation {
        EncodingOp::Add => {
            let (vectors, vector_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| {
                    let left = left_value.value_handle().clone();
                    let right = right_value.value_handle().clone();
                    let output = left_value + right_value;
                    Ok((
                        output.clone(),
                        MatrixBinaryConstructionTrace {
                            left,
                            right,
                            output: output.value_handle().clone(),
                        },
                    ))
                },
            )?;
            let (public_keys, public_key_trace) = key_binary(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Add,
            )?;
            (vectors, public_keys, vector_trace, public_key_trace)
        }
        EncodingOp::Sub => {
            let (vectors, vector_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| {
                    let left = left_value.value_handle().clone();
                    let right = right_value.value_handle().clone();
                    let output = left_value - right_value;
                    Ok((
                        output.clone(),
                        MatrixBinaryConstructionTrace {
                            left,
                            right,
                            output: output.value_handle().clone(),
                        },
                    ))
                },
            )?;
            let (public_keys, public_key_trace) = key_binary(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Sub,
            )?;
            (vectors, public_keys, vector_trace, public_key_trace)
        }
    };
    Ok((
        BggEncodingFamily { vectors, public_keys, plaintexts },
        EncodingComponentConstructionTrace {
            vectors: vector_trace,
            public_keys: public_key_trace,
            plaintexts: plaintext_trace,
        },
    ))
}

fn encoding_multiply(
    compiler: &BggEncodingCompiler,
    left: &BggEncodingFamily,
    right: &BggEncodingFamily,
) -> Result<(BggEncodingFamily, EncodingMultiplicationConstructionTrace), DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let base = compiler.public_key.base.clone();
    let digits = compiler.public_key.digit_count.clone();
    let (decomposed_right, right_public_key_decompositions) =
        right.public_keys.matrices.clone().parallel_map_values_traced(move |_, key| {
            let input = key.value_handle().clone();
            let decomposition = key.decompose(base, digits);
            let decomposition_handle = decomposition.value_handle().clone();
            let materialized = decomposition.as_mat();
            (
                materialized.clone(),
                DecompositionConstructionTrace {
                    input,
                    decomposition: decomposition_handle,
                    materialized: materialized.value_handle().clone(),
                },
            )
        })?;
    let (public_keys, public_keys_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (left.public_keys.matrices.clone(), decomposed_right.clone()),
        |_, (key, decomposition)| {
            let left = key.value_handle().clone();
            let right = decomposition.value_handle().clone();
            let output = key * decomposition;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let (first, first_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (left.vectors.clone(), decomposed_right.clone()),
        |_, (vector, key)| {
            let left = vector.value_handle().clone();
            let right = key.value_handle().clone();
            let output = vector * key;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let (second, second_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (right.vectors.clone(), left.plaintexts.clone()),
        |_, (vector, plaintext)| {
            let left = vector.value_handle().clone();
            let right = plaintext.value_handle().clone();
            let output = vector * plaintext;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let (vectors, vectors_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (first.clone(), second.clone()),
        |_, (left_value, right_value)| {
            let left = left_value.value_handle().clone();
            let right = right_value.value_handle().clone();
            let output = left_value + right_value;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let (plaintexts, plaintexts_trace) = mxx_dsl::parallel_zip_bundle_result_traced(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        |_, (left_value, right_value)| {
            let left = left_value.value_handle().clone();
            let right = right_value.value_handle().clone();
            let output = left_value * right_value;
            Ok((
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right,
                    output: output.value_handle().clone(),
                },
            ))
        },
    )?;
    let trace = EncodingMultiplicationConstructionTrace {
        right_public_key_decompositions,
        public_keys: public_keys_trace,
        left_vectors_times_right_decompositions: first_trace,
        right_vectors_times_left_plaintexts: second_trace,
        vectors: vectors_trace,
        plaintexts: plaintexts_trace,
    };
    Ok((
        BggEncodingFamily {
            vectors,
            public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
            plaintexts,
        },
        trace,
    ))
}

fn encoding_scalar(
    compiler: &BggEncodingCompiler,
    input: &BggEncodingFamily,
    scalar: Mat,
) -> Result<(BggEncodingFamily, EncodingComponentConstructionTrace), DynamicBooleanBggError> {
    input.validate()?;
    let (public_keys, public_key_trace) =
        key_scalar(&compiler.public_key, &input.public_keys, scalar.clone())?;
    let scalar_handle = scalar.value_handle().clone();
    let (vectors, vector_trace) = input.vectors.clone().parallel_map_values_traced({
        let scalar = scalar.clone();
        let scalar_handle = scalar_handle.clone();
        move |_, value| {
            let left = value.value_handle().clone();
            let output = value * scalar;
            (
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right: scalar_handle,
                    output: output.value_handle().clone(),
                },
            )
        }
    })?;
    let (plaintexts, plaintext_trace) =
        input.plaintexts.clone().parallel_map_values_traced(move |_, value| {
            let left = value.value_handle().clone();
            let output = value * scalar;
            (
                output.clone(),
                MatrixBinaryConstructionTrace {
                    left,
                    right: scalar_handle.clone(),
                    output: output.value_handle().clone(),
                },
            )
        })?;
    Ok((
        BggEncodingFamily { vectors, public_keys, plaintexts },
        EncodingComponentConstructionTrace {
            vectors: vector_trace,
            public_keys: public_key_trace,
            plaintexts: plaintext_trace,
        },
    ))
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
        let (public_output, public_trace) = evaluate_boolean_public_key_layers(
            &public_context,
            &public_params,
            public_circuit,
            public_inputs,
            one_key.clone(),
            public_key.clone(),
        )
        .unwrap();
        let (public_graph, public_freeze_map) = public_context
            .family_output("output", public_output.matrices)
            .unwrap()
            .build_with_freeze_map()
            .unwrap();
        public_graph.validate(&bindings()).unwrap();
        public_freeze_map
            .resolve_unique(&public_trace.layers.layer_scan.outputs[0])
            .expect("public-key layer scan output resolves exactly");
        public_freeze_map
            .resolve_unique(
                &public_trace
                    .layers
                    .layer_scan
                    .scope
                    .body
                    .gate_slots
                    .scope
                    .body
                    .gate
                    .right_decomposition,
            )
            .expect("nested public-key decomposition resolves exactly");

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
        let (encoding_output, encoding_trace) = evaluate_boolean_encoding_layers(
            &encoding_context,
            &encoding_params,
            encoding_circuit,
            encoding_inputs,
            one_encoding,
            BggEncodingCompiler { public_key },
        )
        .unwrap();
        let (encoding_graph, encoding_freeze_map) = encoding_context
            .family_output("vector", encoding_output.vectors)
            .unwrap()
            .family_output("public-key", encoding_output.public_keys.matrices)
            .unwrap()
            .family_output("plaintext", encoding_output.plaintexts)
            .unwrap()
            .build_with_freeze_map()
            .unwrap();
        encoding_graph.validate(&bindings()).unwrap();
        encoding_freeze_map
            .resolve_unique(&encoding_trace.layers.outputs[0])
            .expect("encoding layer scan output resolves exactly");
        encoding_freeze_map
            .resolve_unique(
                &encoding_trace
                    .layers
                    .scope
                    .body
                    .multiplication
                    .right_public_key_decompositions
                    .scope
                    .body
                    .decomposition,
            )
            .expect("nested encoding decomposition resolves exactly");
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
