import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1073692673, 1073643521, 1073479681, 1073430529, 1073299457]
def layout0 : MxxRuntime.RegularLayout 1425702062627056211233337151529541201382227969 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 15
    base := 32768
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1425702062627056211233337151529541201382227969 ∧ n = 8192 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
