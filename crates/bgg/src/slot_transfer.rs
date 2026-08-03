//! Slot transfer and reduction over declarative indexed families.

use crate::{NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire};
use mxx_dsl::{DslError, Family, Mat, Ring};
use mxx_ir_core::IntExpr;
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SlotFamilyCompileError {
    #[error("slot transfer requires at least one destination slot")]
    EmptyTransfer,
    #[error("slot reduction requires at least one input family and one source slot")]
    EmptyReduction,
    #[error("slot reduction input count exceeds its source-slot count")]
    TooManyReductionInputs,
    #[error("slot reduction requires homogeneous public-key reveal metadata")]
    RevealMetadataMismatch,
    #[error("slot reduction requires homogeneous plaintext availability")]
    PlaintextAvailabilityMismatch,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

#[derive(Clone, Debug, Default)]
pub struct NaiveBggSlotTransferCompiler;

impl NaiveBggSlotTransferCompiler {
    pub fn transfer_public_keys(
        &self,
        input: &NaiveBggPublicKeyVecWire,
        source_slots: &[(u32, Option<u32>)],
    ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
        Ok(NaiveBggPublicKeyVecWire {
            matrices: transfer_matrix_family(&input.matrices, source_slots)?,
            reveal_plaintext: input.reveal_plaintext,
        })
    }

    pub fn reduce_public_keys(
        &self,
        inputs: &[NaiveBggPublicKeyVecWire],
        source_slot_count: usize,
    ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
        let Some(first) = inputs.first() else {
            return Err(SlotFamilyCompileError::EmptyReduction);
        };
        if inputs.par_iter().any(|input| input.reveal_plaintext != first.reveal_plaintext) {
            return Err(SlotFamilyCompileError::RevealMetadataMismatch);
        }
        Ok(NaiveBggPublicKeyVecWire {
            matrices: reduce_matrix_families(
                &inputs.par_iter().map(|input| input.matrices.clone()).collect::<Vec<_>>(),
                source_slot_count,
            )?,
            reveal_plaintext: first.reveal_plaintext,
        })
    }

    pub fn transfer_encodings(
        &self,
        input: &NaiveBggEncodingVecWire,
        source_slots: &[(u32, Option<u32>)],
    ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
        Ok(NaiveBggEncodingVecWire {
            vectors: transfer_matrix_family(&input.vectors, source_slots)?,
            pubkeys: transfer_matrix_family(&input.pubkeys, source_slots)?,
            pubkey_reveal_plaintext: input.pubkey_reveal_plaintext,
            plaintexts: input
                .plaintexts
                .as_ref()
                .map(|plaintexts| transfer_matrix_family(plaintexts, source_slots))
                .transpose()?,
        })
    }

    pub fn reduce_encodings(
        &self,
        inputs: &[NaiveBggEncodingVecWire],
        source_slot_count: usize,
    ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
        let Some(first) = inputs.first() else {
            return Err(SlotFamilyCompileError::EmptyReduction);
        };
        if inputs
            .par_iter()
            .any(|input| input.pubkey_reveal_plaintext != first.pubkey_reveal_plaintext)
        {
            return Err(SlotFamilyCompileError::RevealMetadataMismatch);
        }
        let has_plaintexts = first.plaintexts.is_some();
        if inputs.par_iter().any(|input| input.plaintexts.is_some() != has_plaintexts) {
            return Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch);
        }
        let vectors = inputs.par_iter().map(|input| input.vectors.clone()).collect::<Vec<_>>();
        let pubkeys = inputs.par_iter().map(|input| input.pubkeys.clone()).collect::<Vec<_>>();
        let plaintexts = has_plaintexts
            .then(|| {
                inputs
                    .par_iter()
                    .map(|input| input.plaintexts.as_ref().expect("checked").clone())
                    .collect::<Vec<_>>()
            })
            .map(|families| reduce_matrix_families(&families, source_slot_count))
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors: reduce_matrix_families(&vectors, source_slot_count)?,
            pubkeys: reduce_matrix_families(&pubkeys, source_slot_count)?,
            pubkey_reveal_plaintext: first.pubkey_reveal_plaintext,
            plaintexts,
        })
    }
}

fn transfer_matrix_family(
    input: &Family<Mat>,
    source_slots: &[(u32, Option<u32>)],
) -> Result<Family<Mat>, SlotFamilyCompileError> {
    if source_slots.is_empty() {
        return Err(SlotFamilyCompileError::EmptyTransfer);
    }
    let descriptors = source_slots
        .par_iter()
        .map(|(source, scalar)| (usize::try_from(*source).expect("u32 fits usize"), *scalar))
        .collect::<Vec<_>>();
    let ty = input.element_type();
    let ring = Ring::new(ty.modulus.clone(), ty.ring_dimension.clone());
    let outputs = descriptors
        .into_iter()
        .map(|(source, scalar)| {
            let selected = input.get_static(source);
            scalar.map_or(selected.clone(), |scalar| {
                selected * ring.polynomial([IntExpr::constant(scalar)])
            })
        })
        .collect();
    Ok(Family::pack(outputs)?)
}

fn reduce_matrix_families(
    inputs: &[Family<Mat>],
    source_slot_count: usize,
) -> Result<Family<Mat>, SlotFamilyCompileError> {
    if inputs.is_empty() || source_slot_count == 0 {
        return Err(SlotFamilyCompileError::EmptyReduction);
    }
    if inputs.len() > source_slot_count {
        return Err(SlotFamilyCompileError::TooManyReductionInputs);
    }
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let ty = input.element_type();
        let ring = Ring::new(ty.modulus.clone(), ty.ring_dimension.clone());
        let mut terms = (0..source_slot_count).map(|source| {
            input.get_static(source) *
                ring.polynomial(
                    (0..=source).map(|index| IntExpr::constant(usize::from(index == source))),
                )
        });
        let first = terms.next().expect("nonzero source slot count");
        outputs.push(terms.fold(first, |sum, term| sum + term));
    }
    Ok(Family::pack(outputs)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::{DslContext, Family};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn runtime_transfer_and_reduce_match_the_slotwise_primitive_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let mut inputs = BTreeMap::new();
        let mut source_values = BTreeMap::<String, Vec<DCRTPolyMatrix>>::new();
        let mut make_encoding = |prefix: &str| {
            let mut make_family = |component: &str, width: usize, component_index: usize| {
                let values = (0..3)
                    .map(|slot| {
                        row(
                            &parameters,
                            width,
                            usize::from(prefix == "second") * 9 + component_index * 3 + slot,
                        )
                    })
                    .collect::<Vec<_>>();
                let wires = values
                    .iter()
                    .enumerate()
                    .map(|(slot, value)| {
                        let name = format!("{prefix}_{component}_{slot}");
                        inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                        ring.input(name, (1, width))
                    })
                    .collect();
                source_values.insert(format!("{prefix}_{component}"), values);
                Family::pack(wires).expect("three-member input family")
            };
            NaiveBggEncodingVecWire {
                vectors: make_family("vector", columns, 0),
                pubkeys: make_family("pubkey", columns, 1),
                pubkey_reveal_plaintext: true,
                plaintexts: Some(make_family("plaintext", 1, 2)),
            }
        };
        let first = make_encoding("first");
        let second = make_encoding("second");
        let compiler = NaiveBggSlotTransferCompiler;
        let transferred = compiler
            .transfer_encodings(&first, &[(2, Some(3)), (0, None)])
            .expect("valid transfer");
        let reduced = compiler.reduce_encodings(&[first, second], 3).expect("valid reduction");

        let mut context = DslContext::new("naive-slot-transfer-runtime");
        for (prefix, output, count) in
            [("transferred", &transferred, 2usize), ("reduced", &reduced, 2usize)]
        {
            for slot in 0..count {
                for (component, family) in [
                    ("vector", &output.vectors),
                    ("pubkey", &output.pubkeys),
                    ("plaintext", output.plaintexts.as_ref().expect("plaintext family")),
                ] {
                    context = context
                        .output(format!("{prefix}_{component}_{slot}"), family.get_static(slot))
                        .expect("matrix output");
                }
            }
        }
        let result =
            execute_graph(context.build().expect("runtime graph"), parameters.clone(), inputs);

        let transfer_scalar = DCRTPoly::from_u32s(&parameters, &[3]);
        for component in ["vector", "pubkey", "plaintext"] {
            let source = &source_values[&format!("first_{component}")];
            let expected = [source[2].clone() * transfer_scalar.clone(), source[0].clone()];
            for (slot, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    matrix_output(&result, &format!("transferred_{component}_{slot}")),
                    &expected
                );
            }
        }
        for (output_slot, prefix) in ["first", "second"].into_iter().enumerate() {
            for component in ["vector", "pubkey", "plaintext"] {
                let source = &source_values[&format!("{prefix}_{component}")];
                let expected = source
                    .iter()
                    .enumerate()
                    .map(|(slot, value)| {
                        let mut coefficients = vec![0u32; slot + 1];
                        coefficients[slot] = 1;
                        value.clone() * DCRTPoly::from_u32s(&parameters, &coefficients)
                    })
                    .reduce(|left, right| left + right)
                    .expect("three source slots");
                assert_eq!(
                    matrix_output(&result, &format!("reduced_{component}_{output_slot}")),
                    &expected
                );
            }
        }
    }

    #[test]
    fn reduction_rejects_oversized_rotations_during_validation() {
        let ring = Ring::new(17, 8);
        let input = NaiveBggPublicKeyVecWire {
            matrices: Family::pack(
                (0..9).map(|slot| ring.input(format!("slot-{slot}"), (1, 2))).collect(),
            )
            .unwrap(),
            reveal_plaintext: true,
        };
        let output = NaiveBggSlotTransferCompiler
            .reduce_public_keys(&[input], 9)
            .expect("construction leaves rotation validation to ir-core");
        let graph = DslContext::new("oversized-slot-reduction")
            .output("output", output.matrices.get_static(0))
            .unwrap()
            .build()
            .unwrap();
        let error = graph
            .validate(&ParamEnv::default())
            .expect_err("rotation exponent exceeds ring dimension");
        assert!(error.to_string().contains("constant polynomial exceeds the ring dimension"));
    }

    #[test]
    fn reduction_rejects_heterogeneous_family_metadata() {
        let ring = Ring::new(17, 8);
        let rows =
            Family::pack(vec![ring.input("row-0", (1, 2)), ring.input("row-1", (1, 2))]).unwrap();
        let scalars =
            Family::pack(vec![ring.input("scalar-0", (1, 1)), ring.input("scalar-1", (1, 1))])
                .unwrap();
        let public = NaiveBggPublicKeyVecWire { matrices: rows.clone(), reveal_plaintext: true };
        let hidden = NaiveBggPublicKeyVecWire { matrices: rows.clone(), reveal_plaintext: false };
        assert!(matches!(
            NaiveBggSlotTransferCompiler.reduce_public_keys(&[public, hidden], 2),
            Err(SlotFamilyCompileError::RevealMetadataMismatch)
        ));

        let revealed = NaiveBggEncodingVecWire {
            vectors: rows.clone(),
            pubkeys: rows.clone(),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(scalars),
        };
        let unavailable = NaiveBggEncodingVecWire {
            vectors: rows.clone(),
            pubkeys: rows,
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        };
        assert!(matches!(
            NaiveBggSlotTransferCompiler.reduce_encodings(&[revealed, unavailable], 2),
            Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch)
        ));
    }
}
