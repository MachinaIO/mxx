//! The generated backend module: the regular gadget layout of every ring a protocol's gadget
//! nodes use, and the `MxxRuntime.BackendContext` that generated relations read them from.
//!
//! The layouts come from the graphs themselves (see [`super::BackendLayout`]), so a protocol's
//! export needs no runtime parameters. Lean checks every layout obligation: coprime towers whose
//! product is the modulus, and enough digits to cover each tower.

use super::BackendLayout;
use std::fmt::Write;

/// Renders the module [`super::claim::BACKEND_MODULE`] holding `layouts`, which must have
/// distinct `(modulus, ring_dimension)` keys.
pub fn render_backend(layouts: &[BackendLayout]) -> String {
    let mut source =
        "import MxxRuntime\n\nnamespace Backend\n\nnoncomputable section\n\n".to_owned();
    for (index, layout) in layouts.iter().enumerate() {
        let moduli = layout.crt_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
        let base_bits = layout.base.bits() - 1;
        let digits = layout.digits_per_tower;
        let dropped = layout.dropped_towers.unwrap_or(0);
        write!(
            source,
            r#"def moduli{index} : List Nat := [{moduli}]
def layout{index} : MxxRuntime.RegularLayout {q} :=
  {{ crtModuli := moduli{index}
    droppedModuli := {dropped}
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := {base_bits}
    base := {base}
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := {digits}
    digits_pos := by norm_num
    capacity := by decide }}

"#,
            q = layout.modulus,
            base = layout.base,
        )
        .expect("writing to a string");
    }
    // Without layouts the lookup ignores its ring, so its binders are anonymous.
    let binders = if layouts.is_empty() { "_ _" } else { "q n" };
    source.push_str(&format!(
        "def backend : MxxRuntime.BackendContext where\n  regularLayout {binders} :=\n"
    ));
    for (index, layout) in layouts.iter().enumerate() {
        writeln!(
            source,
            "    if h : q = {} ∧ n = {} then\n      some (h.1.symm ▸ layout{index})\n    else",
            layout.modulus, layout.ring_dimension
        )
        .expect("writing to a string");
    }
    source.push_str("    none\n\nend\nend Backend\n");
    source
}
