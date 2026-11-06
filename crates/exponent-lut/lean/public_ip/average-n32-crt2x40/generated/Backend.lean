import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1099511626049, 1099511625409]
def layout0 : MxxRuntime.RegularLayout 1208925815113228574679041 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1208925815113228574679041 ∧ n = 32 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
