import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [65537, 79873]
def layout0 : MxxRuntime.RegularLayout 5234636801 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide +kernel
    crtModuli_nonempty := by decide +kernel
    modulus_pos := by decide +kernel
    pairwise_coprime := by unfold Pairwise; decide +kernel
    product_eq := by decide +kernel
    baseBits := 6
    base := 64
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide +kernel
    digitsPerTower := 3
    digits_pos := by norm_num
    capacity := by decide +kernel }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 5234636801 ∧ n = 1024 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
