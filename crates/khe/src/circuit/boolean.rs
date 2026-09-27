//! Bounded Boolean circuits whose active widths, wiring, and output are public runtime data.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::PolyCircuit;
use crate::Poly;

const GATE_RECORD_BYTES: usize = 9;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BooleanCircuitShape {
    pub instance_width: usize,
    pub witness_width: usize,
    pub depth: usize,
    pub max_layer_width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BooleanCircuitData {
    pub layers: Vec<Vec<BooleanGateData>>,
    pub output_source: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BooleanGateData {
    pub kind: BooleanGateKind,
    pub left: usize,
    pub right: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BooleanGateKind {
    ConstantFalse = 0,
    ConstantTrue = 1,
    Copy = 2,
    Not = 3,
    And = 4,
    Xor = 5,
}

impl TryFrom<u8> for BooleanGateKind {
    type Error = BooleanCircuitError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ConstantFalse),
            1 => Ok(Self::ConstantTrue),
            2 => Ok(Self::Copy),
            3 => Ok(Self::Not),
            4 => Ok(Self::And),
            5 => Ok(Self::Xor),
            opcode => Err(BooleanCircuitError::InvalidOpcode(opcode)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanCircuitAnalysis {
    pub depth: usize,
    pub maximum_layer_width: usize,
    pub gate_count: usize,
    pub multiplicative_depth: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BooleanCircuitError {
    #[error("a Boolean circuit shape must have at least one logical input")]
    EmptyInputs,
    #[error("a Boolean circuit bound must permit at least one layer")]
    EmptyLayers,
    #[error("Boolean circuit maximum layer width is zero")]
    EmptyLayer { layer: usize },
    #[error("Boolean circuit width {width} exceeds the canonical u32 index range")]
    WidthTooLarge { width: usize },
    #[error("Boolean circuit input width overflowed usize")]
    InputWidthOverflow,
    #[error("output source {slot} is outside the active final layer width {width}")]
    InvalidOutputSlot { slot: usize, width: usize },
    #[error("circuit has {actual} layers but its shape requires {expected}")]
    LayerCount { expected: usize, actual: usize },
    #[error("circuit layer {layer} has {actual} gates but the maximum is {expected}")]
    LayerWidth { layer: usize, expected: usize, actual: usize },
    #[error("gate ({layer}, {slot}) has a noncanonical unused {field} index")]
    NoncanonicalUnusedIndex { layer: usize, slot: usize, field: &'static str },
    #[error("gate ({layer}, {slot}) predecessor {index} is outside the preceding width {width}")]
    PredecessorOutOfRange { layer: usize, slot: usize, index: usize, width: usize },
    #[error("Boolean input arity is {actual}, expected {expected}")]
    InputArity { expected: usize, actual: usize },
    #[error("unknown Boolean gate opcode {0}")]
    InvalidOpcode(u8),
    #[error("canonical Boolean circuit data is truncated")]
    TruncatedEncoding,
    #[error("canonical Boolean circuit data contains trailing bytes")]
    TrailingEncoding,
    #[error("a canonical predecessor index does not fit in u32")]
    IndexTooLarge,
}

impl BooleanCircuitShape {
    pub fn validate(&self) -> Result<(), BooleanCircuitError> {
        let input_width = self
            .instance_width
            .checked_add(self.witness_width)
            .ok_or(BooleanCircuitError::InputWidthOverflow)?;
        if input_width == 0 {
            return Err(BooleanCircuitError::EmptyInputs);
        }
        check_width(input_width)?;
        if self.depth == 0 {
            return Err(BooleanCircuitError::EmptyLayers);
        }
        if self.max_layer_width == 0 {
            return Err(BooleanCircuitError::EmptyLayer { layer: 0 });
        }
        check_width(self.max_layer_width)?;
        if input_width > self.max_layer_width {
            return Err(BooleanCircuitError::LayerWidth {
                layer: 0,
                expected: self.max_layer_width,
                actual: input_width,
            });
        }
        Ok(())
    }

    pub fn input_width(&self) -> Result<usize, BooleanCircuitError> {
        self.validate()?;
        Ok(self.instance_width + self.witness_width)
    }

    pub fn analyze(&self) -> Result<BooleanCircuitAnalysis, BooleanCircuitError> {
        self.validate()?;
        Ok(BooleanCircuitAnalysis {
            depth: self.depth,
            maximum_layer_width: self.max_layer_width,
            gate_count: self
                .depth
                .checked_mul(self.max_layer_width)
                .ok_or(BooleanCircuitError::InputWidthOverflow)?,
            multiplicative_depth: self.depth,
        })
    }
}

impl BooleanCircuitData {
    pub fn validate(&self, shape: &BooleanCircuitShape) -> Result<(), BooleanCircuitError> {
        shape.validate()?;
        if self.layers.len() != shape.depth {
            return Err(BooleanCircuitError::LayerCount {
                expected: shape.depth,
                actual: self.layers.len(),
            });
        }
        let mut preceding_width = shape.instance_width + shape.witness_width;
        for (layer, gates) in self.layers.iter().enumerate() {
            if gates.is_empty() || gates.len() > shape.max_layer_width {
                return Err(BooleanCircuitError::LayerWidth {
                    layer,
                    expected: shape.max_layer_width,
                    actual: gates.len(),
                });
            }
            for (slot, gate) in gates.iter().enumerate() {
                validate_gate(*gate, layer, slot, preceding_width)?;
            }
            preceding_width = gates.len();
        }
        if self.output_source >= preceding_width {
            return Err(BooleanCircuitError::InvalidOutputSlot {
                slot: self.output_source,
                width: preceding_width,
            });
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        shape: &BooleanCircuitShape,
        instance: &[bool],
        witness: &[bool],
    ) -> Result<bool, BooleanCircuitError> {
        self.validate(shape)?;
        if instance.len() != shape.instance_width {
            return Err(BooleanCircuitError::InputArity {
                expected: shape.instance_width,
                actual: instance.len(),
            });
        }
        if witness.len() != shape.witness_width {
            return Err(BooleanCircuitError::InputArity {
                expected: shape.witness_width,
                actual: witness.len(),
            });
        }
        let mut preceding = instance.iter().chain(witness).copied().collect::<Vec<_>>();
        for layer in &self.layers {
            preceding = layer.par_iter().map(|gate| evaluate_gate(*gate, &preceding)).collect();
        }
        Ok(preceding[self.output_source])
    }

    pub fn analyze(
        &self,
        shape: &BooleanCircuitShape,
    ) -> Result<BooleanCircuitAnalysis, BooleanCircuitError> {
        self.validate(shape)?;
        let mut preceding_depth = vec![0usize; shape.instance_width + shape.witness_width];
        for layer in &self.layers {
            preceding_depth = layer
                .iter()
                .map(|gate| match gate.kind {
                    BooleanGateKind::ConstantFalse | BooleanGateKind::ConstantTrue => 0,
                    BooleanGateKind::Copy | BooleanGateKind::Not => preceding_depth[gate.left],
                    BooleanGateKind::And | BooleanGateKind::Xor => {
                        preceding_depth[gate.left].max(preceding_depth[gate.right]) + 1
                    }
                })
                .collect();
        }
        let mut analysis = shape.analyze()?;
        analysis.gate_count = self.layers.iter().map(Vec::len).sum();
        analysis.multiplicative_depth = preceding_depth[self.output_source];
        Ok(analysis)
    }

    pub fn to_canonical_bytes(
        &self,
        shape: &BooleanCircuitShape,
    ) -> Result<Vec<u8>, BooleanCircuitError> {
        self.validate(shape)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &u32::try_from(self.output_source)
                .map_err(|_| BooleanCircuitError::IndexTooLarge)?
                .to_le_bytes(),
        );
        for layer in &self.layers {
            bytes.extend_from_slice(
                &u32::try_from(layer.len())
                    .map_err(|_| BooleanCircuitError::IndexTooLarge)?
                    .to_le_bytes(),
            );
            for gate in layer {
                bytes.push(gate.kind as u8);
                bytes.extend_from_slice(
                    &u32::try_from(gate.left)
                        .map_err(|_| BooleanCircuitError::IndexTooLarge)?
                        .to_le_bytes(),
                );
                bytes.extend_from_slice(
                    &u32::try_from(gate.right)
                        .map_err(|_| BooleanCircuitError::IndexTooLarge)?
                        .to_le_bytes(),
                );
            }
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(
        shape: &BooleanCircuitShape,
        bytes: &[u8],
    ) -> Result<Self, BooleanCircuitError> {
        shape.validate()?;
        let mut offset = 0usize;
        let output_source = read_u32(bytes, &mut offset)?;
        let mut layers = Vec::with_capacity(shape.depth);
        for _ in 0..shape.depth {
            let width = read_u32(bytes, &mut offset)?;
            let layer = (0..width)
                .map(|_| {
                    let end = offset
                        .checked_add(GATE_RECORD_BYTES)
                        .ok_or(BooleanCircuitError::InputWidthOverflow)?;
                    let record =
                        bytes.get(offset..end).ok_or(BooleanCircuitError::TruncatedEncoding)?;
                    offset = end;
                    Ok(BooleanGateData {
                        kind: BooleanGateKind::try_from(record[0])?,
                        left: u32::from_le_bytes(record[1..5].try_into().unwrap()) as usize,
                        right: u32::from_le_bytes(record[5..9].try_into().unwrap()) as usize,
                    })
                })
                .collect::<Result<Vec<_>, BooleanCircuitError>>()?;
            layers.push(layer);
        }
        if offset != bytes.len() {
            return Err(BooleanCircuitError::TrailingEncoding);
        }
        let circuit = Self { layers, output_source };
        circuit.validate(shape)?;
        Ok(circuit)
    }
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<usize, BooleanCircuitError> {
    let end = offset.checked_add(4).ok_or(BooleanCircuitError::InputWidthOverflow)?;
    let value = bytes.get(*offset..end).ok_or(BooleanCircuitError::TruncatedEncoding)?;
    *offset = end;
    Ok(u32::from_le_bytes(value.try_into().unwrap()) as usize)
}

pub fn to_poly_circuit<P: Poly>(
    shape: &BooleanCircuitShape,
    circuit: &BooleanCircuitData,
) -> Result<PolyCircuit<P>, BooleanCircuitError> {
    circuit.validate(shape)?;
    let mut result = PolyCircuit::new();
    let mut preceding = result.input(shape.instance_width + shape.witness_width).to_vec();
    for layer in &circuit.layers {
        let mut current = Vec::with_capacity(layer.len());
        for gate in layer {
            let output = match gate.kind {
                BooleanGateKind::ConstantFalse => result.const_zero_gate(),
                BooleanGateKind::ConstantTrue => result.const_one_gate(),
                BooleanGateKind::Copy => preceding[gate.left].into(),
                BooleanGateKind::Not => result.not_gate(preceding[gate.left]),
                BooleanGateKind::And => {
                    result.and_gate(preceding[gate.left], preceding[gate.right])
                }
                BooleanGateKind::Xor => {
                    result.xor_gate(preceding[gate.left], preceding[gate.right])
                }
            };
            current.push(output.as_single_wire());
        }
        preceding = current;
    }
    result.output([preceding[circuit.output_source]]);
    Ok(result)
}

fn check_width(width: usize) -> Result<(), BooleanCircuitError> {
    if width > u32::MAX as usize {
        Err(BooleanCircuitError::WidthTooLarge { width })
    } else {
        Ok(())
    }
}

fn validate_gate(
    gate: BooleanGateData,
    layer: usize,
    slot: usize,
    preceding_width: usize,
) -> Result<(), BooleanCircuitError> {
    match gate.kind {
        BooleanGateKind::ConstantFalse | BooleanGateKind::ConstantTrue => {
            require_zero(gate.left, layer, slot, "left")?;
            require_zero(gate.right, layer, slot, "right")?;
        }
        BooleanGateKind::Copy | BooleanGateKind::Not => {
            require_index(gate.left, layer, slot, preceding_width)?;
            require_zero(gate.right, layer, slot, "right")?;
        }
        BooleanGateKind::And | BooleanGateKind::Xor => {
            require_index(gate.left, layer, slot, preceding_width)?;
            require_index(gate.right, layer, slot, preceding_width)?;
        }
    }
    Ok(())
}

fn require_zero(
    index: usize,
    layer: usize,
    slot: usize,
    field: &'static str,
) -> Result<(), BooleanCircuitError> {
    if index == 0 {
        Ok(())
    } else {
        Err(BooleanCircuitError::NoncanonicalUnusedIndex { layer, slot, field })
    }
}

fn require_index(
    index: usize,
    layer: usize,
    slot: usize,
    width: usize,
) -> Result<(), BooleanCircuitError> {
    if index < width {
        Ok(())
    } else {
        Err(BooleanCircuitError::PredecessorOutOfRange { layer, slot, index, width })
    }
}

fn evaluate_gate(gate: BooleanGateData, preceding: &[bool]) -> bool {
    match gate.kind {
        BooleanGateKind::ConstantFalse => false,
        BooleanGateKind::ConstantTrue => true,
        BooleanGateKind::Copy => preceding[gate.left],
        BooleanGateKind::Not => !preceding[gate.left],
        BooleanGateKind::And => preceding[gate.left] && preceding[gate.right],
        BooleanGateKind::Xor => preceding[gate.left] != preceding[gate.right],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> BooleanCircuitShape {
        BooleanCircuitShape { instance_width: 1, witness_width: 1, depth: 2, max_layer_width: 3 }
    }

    fn circuit() -> BooleanCircuitData {
        BooleanCircuitData {
            layers: vec![
                vec![
                    BooleanGateData { kind: BooleanGateKind::Copy, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 },
                    BooleanGateData { kind: BooleanGateKind::ConstantTrue, left: 0, right: 0 },
                ],
                vec![
                    BooleanGateData { kind: BooleanGateKind::Not, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Xor, left: 1, right: 2 },
                ],
            ],
            output_source: 1,
        }
    }

    #[test]
    fn validates_and_evaluates_layers() {
        assert!(!circuit().evaluate(&shape(), &[true], &[true]).unwrap());
        assert!(circuit().evaluate(&shape(), &[false], &[true]).unwrap());
        assert_eq!(circuit().analyze(&shape()).unwrap().multiplicative_depth, 2);
    }

    #[test]
    fn canonical_encoding_round_trips_with_dynamic_widths_and_output() {
        let bytes = circuit().to_canonical_bytes(&shape()).unwrap();
        assert_eq!(bytes.len(), 12 + 5 * GATE_RECORD_BYTES);
        assert_eq!(BooleanCircuitData::from_canonical_bytes(&shape(), &bytes).unwrap(), circuit());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            BooleanCircuitData::from_canonical_bytes(&shape(), &trailing),
            Err(BooleanCircuitError::TrailingEncoding)
        );
    }

    #[test]
    fn rejects_noncanonical_and_out_of_range_inputs() {
        let mut data = circuit();
        data.layers[0][0].right = 1;
        assert!(matches!(
            data.validate(&shape()),
            Err(BooleanCircuitError::NoncanonicalUnusedIndex { .. })
        ));
        let mut data = circuit();
        data.layers[1][1].left = 3;
        assert!(matches!(
            data.validate(&shape()),
            Err(BooleanCircuitError::PredecessorOutOfRange { .. })
        ));
    }

    #[test]
    fn evaluates_every_opcode_and_preserves_fanout() {
        let shape = BooleanCircuitShape {
            instance_width: 2,
            witness_width: 0,
            depth: 2,
            max_layer_width: 6,
        };
        let circuit = BooleanCircuitData {
            layers: vec![
                vec![
                    BooleanGateData { kind: BooleanGateKind::ConstantFalse, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::ConstantTrue, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Copy, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Not, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 },
                    BooleanGateData { kind: BooleanGateKind::Xor, left: 0, right: 1 },
                ],
                vec![
                    BooleanGateData { kind: BooleanGateKind::Copy, left: 5, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Xor, left: 5, right: 5 },
                ],
            ],
            output_source: 1,
        };
        assert!(!circuit.evaluate(&shape, &[false, false], &[]).unwrap());
        assert!(!circuit.evaluate(&shape, &[false, true], &[]).unwrap());
        assert!(!circuit.evaluate(&shape, &[true, false], &[]).unwrap());
        assert!(!circuit.evaluate(&shape, &[true, true], &[]).unwrap());
        let converted =
            to_poly_circuit::<crate::poly::dcrt::poly::DCRTPoly>(&shape, &circuit).unwrap();
        assert_eq!(converted.num_input(), shape.instance_width);
        assert_eq!(converted.num_output(), 1);
    }

    #[test]
    fn rejects_all_malformed_shape_and_encoding_classes() {
        assert_eq!(
            BooleanCircuitShape {
                instance_width: 0,
                witness_width: 0,
                depth: 1,
                max_layer_width: 1,
            }
            .validate(),
            Err(BooleanCircuitError::EmptyInputs)
        );
        assert!(matches!(
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 0,
                depth: 0,
                max_layer_width: 1,
            }
            .validate(),
            Err(BooleanCircuitError::EmptyLayers)
        ));
        assert!(matches!(
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 0,
                depth: 1,
                max_layer_width: 0,
            }
            .validate(),
            Err(BooleanCircuitError::EmptyLayer { .. })
        ));
        assert!(matches!(
            BooleanCircuitData {
                layers: vec![vec![BooleanGateData {
                    kind: BooleanGateKind::Copy,
                    left: 0,
                    right: 0,
                }]],
                output_source: 1,
            }
            .validate(&BooleanCircuitShape {
                instance_width: 1,
                witness_width: 0,
                depth: 1,
                max_layer_width: 1,
            }),
            Err(BooleanCircuitError::InvalidOutputSlot { .. })
        ));

        let bytes = circuit().to_canonical_bytes(&shape()).unwrap();
        assert_eq!(
            BooleanCircuitData::from_canonical_bytes(&shape(), &bytes[..bytes.len() - 1]),
            Err(BooleanCircuitError::TruncatedEncoding)
        );
        let mut invalid_opcode = bytes;
        invalid_opcode[8] = 6;
        assert_eq!(
            BooleanCircuitData::from_canonical_bytes(&shape(), &invalid_opcode),
            Err(BooleanCircuitError::InvalidOpcode(6))
        );
    }
}
