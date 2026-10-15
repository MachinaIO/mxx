import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1125899906842273, 1125899906842177, 1125899906842033, 1125899906841713, 1125899906841377]
def layout0 : MxxRuntime.RegularLayout 1809251394327365744250316798798350396453362684746139641520435240552807944993 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide +kernel
    crtModuli_nonempty := by decide +kernel
    modulus_pos := by decide +kernel
    pairwise_coprime := by unfold Pairwise; decide +kernel
    product_eq := by decide +kernel
    baseBits := 12
    base := 4096
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide +kernel
    digitsPerTower := 5
    digits_pos := by norm_num
    capacity := by decide +kernel }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1809251394327365744250316798798350396453362684746139641520435240552807944993 ∧ n = 8 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
