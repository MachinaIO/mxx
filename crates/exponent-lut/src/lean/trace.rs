//! Concrete native trace indices and their kernel-checked odd-residue permutation.

use crate::ExpLayout;

pub(super) fn trace_indices(layout: &ExpLayout, n: usize) -> Result<String, String> {
    layout.validate(n).map_err(|error| error.to_string())?;
    // Match readout::full_trace_layout: operation width is replaced by the
    // complete native subgroup width while the exponent stride is retained.
    let full = ExpLayout {
        stride_log2: layout.stride_log2,
        width_log2: n.trailing_zeros() - layout.stride_log2,
    };
    let indices = full.automorphism_indices(n).map_err(|error| error.to_string())?;
    let width = indices.len();
    let period = width.checked_mul(2).ok_or("native trace period overflow")?;
    let mut positions = Vec::with_capacity(width);
    let mut inverse = vec![None; width];
    // The inverse is constructed once, with duplicate/missing residues rejected.
    // Mutating its disjoint slots follows the canonical ascending source order.
    for (index, sigma) in indices.iter().copied().enumerate() {
        let residue = sigma % period;
        if residue % 2 != 1 {
            return Err("native trace index has an even residue".into());
        }
        let position = residue / 2;
        if inverse[position].replace(index).is_some() {
            return Err("native trace indices repeat an odd residue".into());
        }
        positions.push(position);
    }
    let inverse = inverse
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or("native trace indices omit an odd residue")?;
    let indices = indices.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    let positions = positions
        .iter()
        .map(|position| format!("⟨{position}, by decide⟩"))
        .collect::<Vec<_>>()
        .join(", ");
    let inverse = inverse
        .iter()
        .map(|position| format!("⟨{position}, by decide⟩"))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        r#"import ExponentTrace

namespace TraceIndices

def ringDimension : Nat := {n}
def stride : Nat := {stride}
def widthLog : Nat := {width_log}
def width : Nat := {width}

def indices : Fin width → Nat := ![{indices}]
def residuePosition : Fin width → Fin width := ![{positions}]
def inversePosition : Fin width → Fin width := ![{inverse}]

def permutation : Fin width ≃ Fin width where
  toFun := residuePosition
  invFun := inversePosition
  left_inv := by decide
  right_inv := by decide

theorem dimension : ringDimension = stride * width := by decide
theorem width_power : width = 2 ^ widthLog := by decide
theorem all_odd : ∀ i, Odd (indices i) := by decide
theorem indices_valid : ∀ i, 0 < indices i ∧ indices i < 2 * ringDimension := by decide
theorem index_correspondence : ∀ i, indices i % (2 * width) =
    2 * (permutation i).val + 1 := by decide
theorem identity_index : indices ⟨0, by decide⟩ = 1 := by decide

end TraceIndices
"#,
        stride = full.stride(),
        width_log = full.width_log2,
    ))
}
