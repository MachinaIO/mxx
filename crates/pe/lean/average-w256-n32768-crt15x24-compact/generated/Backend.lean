import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [16580609, 16515073, 16384001, 16121857, 14942209, 14155777, 13631489, 13565953, 13238273, 12451841, 11599873, 11468801, 11272193, 10223617, 10027009]
def layout0 : MxxRuntime.RegularLayout 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489 :=
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
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide +kernel }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489 ∧ n = 32768 then
      some (h.1.symm ▸ layout0)
    else
    none

end
end Backend
