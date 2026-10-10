import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1073479681, 1072496641, 1071513601, 1070727169, 1069219841, 1068564481]
def layout0 : MxxRuntime.RegularLayout 1509155612652322963894981558308021422531187968728956929 :=
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
    if h : q = 1509155612652322963894981558308021422531187968728956929 ∧ n = 32768 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
