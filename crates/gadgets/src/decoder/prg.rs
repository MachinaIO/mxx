//! Protocol boundary for mask PRG expansion.

/// A conceptual range of mask PRG outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskPrgRange {
    pub conceptual_output_bits: usize,
    pub range_start: usize,
    pub range_len: usize,
}

impl MaskPrgRange {
    pub fn new(conceptual_output_bits: usize, range_start: usize, range_len: usize) -> Self {
        assert!(conceptual_output_bits > 0, "conceptual_output_bits must be positive");
        assert!(range_len > 0, "range_len must be positive");
        assert!(
            range_start + range_len <= conceptual_output_bits,
            "mask PRG range must fit in conceptual output"
        );
        Self { conceptual_output_bits, range_start, range_len }
    }
}

/// Evaluates a protocol-owned mask PRG over already-selected seed wires.
///
/// Protocols keep ownership of seed evolution, selected-half branching, and
/// domain separation. Decoder code only calls this boundary when it needs a
/// specific mask output range.
pub trait MaskPrgEvaluator<E> {
    fn evaluate_mask_prg_range(&self, seed_wires: Vec<E>, range: MaskPrgRange) -> Vec<E>;
}

/// Returns how many PRG outputs should be generated in debug-single-output
/// mode versus normal mode.
pub fn generated_mask_output_bits(
    mask_output_bits: usize,
    debug_reuse_single_sample: bool,
) -> usize {
    assert!(mask_output_bits > 0, "mask_output_bits must be positive");
    if debug_reuse_single_sample { 1 } else { mask_output_bits }
}

/// Expands one debug PRG sample per wire into the normal `mask_bits * wire`
/// layout expected by mask decrypt circuits.
pub fn expand_debug_reused_mask_wires<E: Clone>(
    mask_output_wires: Vec<E>,
    mask_output_bits: usize,
    wire_count: usize,
    debug_reuse_single_sample: bool,
) -> Vec<E> {
    if !debug_reuse_single_sample {
        return mask_output_wires;
    }
    assert_eq!(
        mask_output_wires.len(),
        wire_count,
        "debug PRG mask reuse expects one generated wire per logical output"
    );
    let mut expanded = Vec::with_capacity(
        mask_output_bits.checked_mul(wire_count).expect("mask wire count overflow"),
    );
    for _ in 0..mask_output_bits {
        expanded.extend(mask_output_wires.iter().cloned());
    }
    expanded
}

#[cfg(test)]
mod tests {
    use super::{expand_debug_reused_mask_wires, generated_mask_output_bits};

    #[test]
    fn generated_mask_output_bits_respects_debug_reuse() {
        assert_eq!(generated_mask_output_bits(7, true), 1);
        assert_eq!(generated_mask_output_bits(7, false), 7);
    }

    #[test]
    fn debug_reused_mask_wires_expand_by_mask_bits() {
        let expanded = expand_debug_reused_mask_wires(vec![1, 2], 3, 2, true);
        assert_eq!(expanded, vec![1, 2, 1, 2, 1, 2]);
    }
}
