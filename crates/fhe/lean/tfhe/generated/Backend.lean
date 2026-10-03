import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [65537, 79873]
def layout0 : MxxRuntime.RegularLayout 5234636801 :=
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

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 5234636801 ∧ n = 1024 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
