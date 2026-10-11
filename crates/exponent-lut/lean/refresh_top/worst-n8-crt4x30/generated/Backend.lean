import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1073741441, 1073741329]
def layout0 : MxxRuntime.RegularLayout 1152920561861715089 :=
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

def moduli1 : List Nat := [1073741441, 1073741329, 1073740609]
def layout1 : MxxRuntime.RegularLayout 1237937626222020133447349201 :=
  { crtModuli := moduli1
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

def moduli2 : List Nat := [1073741441, 1073741329, 1073740609, 1073740529]
def layout2 : MxxRuntime.RegularLayout 1329223801648636169536407324729467329 :=
  { crtModuli := moduli2
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

def moduli3 : List Nat := [1073741441, 1073741329, 1073740609, 1073740529, 1073740177]
def layout3 : MxxRuntime.RegularLayout 1427241000054819492486624008799114726956177233 :=
  { crtModuli := moduli3
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

def moduli4 : List Nat := [1073741441, 1073741329, 1073740609, 1073740529, 1073740177, 1073740049]
def layout4 : MxxRuntime.RegularLayout 1532485821333670884548742795052537878078547363014104417 :=
  { crtModuli := moduli4
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

def moduli5 : List Nat := [1073741441, 1073741329, 1073740609, 1073740529, 1073740177, 1073740049, 1073739937]
def layout5 : MxxRuntime.RegularLayout 1645491229252209031554101362188915932898173126614280606821001729 :=
  { crtModuli := moduli5
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
    if h : q = 1152920561861715089 ∧ n = 8 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 1237937626222020133447349201 ∧ n = 8 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 1329223801648636169536407324729467329 ∧ n = 8 then
      some (h.1.symm ▸ layout2)
    else
    if h : q = 1427241000054819492486624008799114726956177233 ∧ n = 8 then
      some (h.1.symm ▸ layout3)
    else
    if h : q = 1532485821333670884548742795052537878078547363014104417 ∧ n = 8 then
      some (h.1.symm ▸ layout4)
    else
    if h : q = 1645491229252209031554101362188915932898173126614280606821001729 ∧ n = 8 then
      some (h.1.symm ▸ layout5)
    else
    none

end
end Backend
