import MxxRuntime

namespace FheBackend

noncomputable section

def moduli0 : List Nat := [65537]
def layout0 : MxxRuntime.RegularLayout 65537 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 6
    base := 64
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 3
    digits_pos := by norm_num
    capacity := by decide }

def moduli1 : List Nat := [79873]
def layout1 : MxxRuntime.RegularLayout 79873 :=
  { crtModuli := moduli1
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 6
    base := 64
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 3
    digits_pos := by norm_num
    capacity := by decide }

def moduli2 : List Nat := [65537, 79873]
def layout2 : MxxRuntime.RegularLayout 5234636801 :=
  { crtModuli := moduli2
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 6
    base := 64
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 3
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 65537 ∧ n = 1024 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 79873 ∧ n = 1024 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 5234636801 ∧ n = 1024 then
      some (h.1.symm ▸ layout2)
    else
    none

end
end FheBackend
