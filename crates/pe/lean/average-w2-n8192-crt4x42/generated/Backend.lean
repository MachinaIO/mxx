import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [4398046150657, 4398045708289, 4398045511681, 4398045380609]
def layout0 : MxxRuntime.RegularLayout 374144139004015249260017124107828540725582728380417 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 21
    base := 2097152
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 374144139004015249260017124107828540725582728380417 ∧ n = 8192 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
