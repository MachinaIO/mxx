import MxxRuntime

namespace FheBackend

noncomputable section

def moduli0 : List Nat := [2147377153]
def layout0 : MxxRuntime.RegularLayout 2147377153 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 4
    digits_pos := by norm_num
    capacity := by decide }

def moduli1 : List Nat := [2147389441]
def layout1 : MxxRuntime.RegularLayout 2147389441 :=
  { crtModuli := moduli1
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 4
    digits_pos := by norm_num
    capacity := by decide }

def moduli2 : List Nat := [2147389441, 2147377153]
def layout2 : MxxRuntime.RegularLayout 4611255024196841473 :=
  { crtModuli := moduli2
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 4
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 2147377153 ∧ n = 2048 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 2147389441 ∧ n = 2048 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 4611255024196841473 ∧ n = 2048 then
      some (h.1.symm ▸ layout2)
    else
    none

end
end FheBackend
