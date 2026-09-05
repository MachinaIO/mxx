//! Emit a concrete Lean regular-gadget layout from the DCRT parameters used by setup.

use crate::lean::{export_dcrt_layouts, render_backend_context};
use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

fn render_fixture() -> String {
    let parameters = DCRTPolyParams::new(2, 1, 10, 5);
    let layout = export_dcrt_layouts([&parameters]).expect("concrete DCRT layout").remove(0);
    let q = layout.modulus.to_str_radix(10);
    let modulus_list = layout.crt_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
    let base = 1u64 << layout.base_bits;
    let digits = layout.regular_digit_count / layout.crt_depth;
    let regular_digits = layout.regular_digit_count;
    let n = layout.ring_dimension;

    let multi_parameters = DCRTPolyParams::new(2, 2, 10, 5);
    let multi_layout =
        export_dcrt_layouts([&multi_parameters]).expect("multi-tower layout").remove(0);
    let multi_q = multi_layout.modulus.to_string();
    let context = render_backend_context(
        &[layout.clone(), multi_layout],
        "GeneratedBackend",
        "GeneratedBackend",
    )
    .expect("render concrete backend context");
    let context = context.source();
    format!(
        r#"{context}

namespace GeneratedConcreteRegular

noncomputable section

def q : Nat := {q}
def ringDimension : Nat := {n}
def base : Nat := {base}
def digitsPerTower : Nat := {digits}
def regularDigits : Nat := {regular_digits}
def moduli : List Nat := [{modulus_list}]

def concreteLayout : MxxRuntime.RegularLayout q :=
  {{ crtModuli := moduli
    crtModuli_nonempty := by simp [moduli]
    modulus_pos := by
      intro tower; fin_cases tower <;> norm_num [moduli]
    pairwise_coprime := by simp [moduli]
    product_eq := by norm_num [q, moduli]
    baseBits := {base_bits}
    base := base
    base_eq := by norm_num [base]
    base_gt_one := by norm_num [base]
    base_even := by refine ⟨16, by norm_num [base]⟩
    digitsPerTower := digitsPerTower
    digits_pos := by norm_num [digitsPerTower]
    capacity := by
      intro tower; fin_cases tower <;> norm_num [moduli, base, digitsPerTower] }}

def publicMatrix : Mxx.Primitives.ExactMatrix q ringDimension 1 2 :=
  MxxRuntime.regularGadgetMatrix concreteLayout
def target : Mxx.Primitives.ExactMatrix q ringDimension 1 1 := fun _ _ => 1
def trapdoor : MxxRuntime.TrapdoorValue
    (Mxx.Primitives.ExactMatrix q ringDimension 1 2) Unit :=
  MxxRuntime.regularGadgetTrapdoor concreteLayout 0

def preimage : Mxx.Primitives.ExactMatrix q ringDimension 2 1 :=
  MxxRuntime.regularDecomposeMatrix concreteLayout target

theorem generated_layout_capacity :
    ∀ tower : Fin moduli.length, moduli.get tower ≤ base ^ digitsPerTower := by
  exact concreteLayout.capacity

theorem generated_digit_bound
    (value : Mxx.Primitives.ExactPoly q ringDimension)
    (limb : MxxRuntime.RegularLimb concreteLayout)
    (coefficient : Fin ringDimension) :
    (MxxRuntime.regularDigitCoefficient concreteLayout value limb coefficient).natAbs ≤ 16 := by
  simpa [MxxRuntime.regularDigitCoefficient, base, concreteLayout] using
    (Mxx.Primitives.balancedDigit_abs_le concreteLayout.base
      concreteLayout.base_gt_one concreteLayout.base_even _)

theorem generated_public_preimage_fixture :
    MxxRuntime.publicGadgetPreimageRuns publicMatrix trapdoor target preimage := by
  exact MxxRuntime.regularGadgetTrapdoor_preimage concreteLayout 0 target

theorem generated_arbitrary_target_reconstruction
    (value : Mxx.Primitives.ExactMatrix q ringDimension 1 1) :
    MxxRuntime.regularGadgetMatrix (n := ringDimension) (rows := 1) concreteLayout *
      MxxRuntime.regularDecomposeMatrix concreteLayout value = value := by
  exact MxxRuntime.regularGadgetMatrix_reconstruct concreteLayout value
    (by norm_num [q]) (by norm_num [ringDimension])

theorem generated_preimage_bound : Mxx.Primitives.PreimageWithin preimage 16 := by
  exact MxxRuntime.regularDecomposeMatrix_bounded concreteLayout target
    (by norm_num [q]) (by norm_num [ringDimension])

theorem generated_backend_lookup :
    GeneratedBackend.backend.regularLayout q ringDimension =
      some GeneratedBackend.layout0 := by
  simp [GeneratedBackend.backend, q, ringDimension]

theorem generated_backend_decomposition
    (value : Mxx.Primitives.ExactMatrix q ringDimension 1 1) :
    MxxRuntime.gadgetDecomposeRuns GeneratedBackend.backend base regularDigits value
      (MxxRuntime.regularDecomposeMatrix GeneratedBackend.layout0 value) := by
  refine ⟨GeneratedBackend.layout0, generated_backend_lookup, ?_, ?_, rfl, rfl⟩
  · norm_num [base, GeneratedBackend.layout0]
  · norm_num [regularDigits, GeneratedBackend.layout0, GeneratedBackend.moduli0]

theorem generated_multitower_lookup :
    GeneratedBackend.backend.regularLayout {multi_q} {n} =
      some GeneratedBackend.layout1 := by
  simp [GeneratedBackend.backend]

theorem generated_multitower_reconstruction
    (value : Mxx.Primitives.ExactMatrix {multi_q} {n} 1 1) :
    MxxRuntime.regularGadgetMatrix (n := {n}) (rows := 1) GeneratedBackend.layout1 *
      MxxRuntime.regularDecomposeMatrix GeneratedBackend.layout1 value = value := by
  exact MxxRuntime.regularGadgetMatrix_reconstruct GeneratedBackend.layout1 value
    (by norm_num) (by norm_num)

end
end GeneratedConcreteRegular
"#,
        q = q,
        context = context,
        multi_q = multi_q,
        n = n,
        base = base,
        digits = digits,
        regular_digits = regular_digits,
        modulus_list = modulus_list,
        base_bits = layout.base_bits,
    )
}

#[test]
fn export_concrete_regular_fixture() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/lean_runtime_fixture");
    std::fs::create_dir_all(&directory).expect("create fixture directory");
    std::fs::write(directory.join("Generated.lean"), render_fixture()).expect("write Lean fixture");
}
